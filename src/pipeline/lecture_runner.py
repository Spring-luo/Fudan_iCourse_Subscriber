"""Per-lecture state machine: prefetch → ASR → OCR drain → summarize → release.

One ``LectureRunner`` instance drives one lecture from "started" to either
"summary saved" or "deliberately skipped".  The class is single-use — make a
new instance per lecture so error state can't leak across runs.

Phases (named like the original ``main.process_lecture`` for diff-friendly
log greps):

  A  short-circuit ``summary already exists`` → mark processed, return.
  B  ``PPTPipeline.submit`` with ``defer_ocr=True``: stages 1-3 (fetch,
     register, dedup) run inline; the OCR jobs are held on the returned
     ``PPTAsyncHandle`` and only enter the pool at drain time, so ASR in
     Phase D gets the CPU to itself.
  C  schedule **next** lecture's prefetch (images always; audio unless
     a transcript already exists) so its download
     overlaps with the current lecture's ASR.
  D  transcript: cached → ASR → official iCourse transcript (config-gated)
     if audio fails.  For ASR, ``Scheduler.audio_downloader
     .get`` blocks for the ffmpeg spawn scheduled earlier, then
     ``Transcriber.transcribe_tail`` reads PCM from the disk file with
     tail-f semantics while ffmpeg keeps writing.
  E  ``handle.drain()`` submits the deferred OCR jobs and blocks for them.
  E2 ``PPTPipeline.prefetch_and_ocr`` spawns a background thread that
     collects + dedups + OCRs the next lecture's pages, overlapping with
     this lecture's LLM wait; leftovers are absorbed by the next
     ``submit``.
  F  ``bucketer.assemble`` builds the prompt; ``Summarizer.summarize``
     calls the LLM round-robin until one succeeds.
  G  persist ``update_summary``, ``mark_processed``, ``clear_error``.
  H  ``audio_downloader.release`` kills ffmpeg + deletes scratch file.

The runner never owns the WebVPN session check; the orchestrator
(LectureRunner's caller) calls ``_check_session`` between lectures so all
threads pick up refreshed cookies through the shared ``ICourseClient``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from src.ai import bucketer
from src.pipeline.ppt_pipeline import PPTPipeline
from src.ai.transcriber import IncompleteAudioError, NoAudioStreamError
from src.runtime import config

if TYPE_CHECKING:
    from src.data.database import Database
    from src.api.icourse import ICourseClient
    from src.runtime.reporter import Reporter
    from src.runtime.scheduler import Scheduler
    from src.ai.summarizer import Summarizer
    from src.ai.transcriber import Transcriber


class LectureRunner:
    """Run one lecture end-to-end.  Construct once per orchestration session,
    call ``run`` for each lecture in order."""

    def __init__(self, client: "ICourseClient", db: "Database",
                 scheduler: "Scheduler", transcriber: "Transcriber",
                 summarizer: "Summarizer", reporter: "Reporter"):
        self._client = client
        self._db = db
        self._scheduler = scheduler
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._reporter = reporter
        self._ppt = PPTPipeline(db, scheduler, reporter)

    PARTIAL_OFFICIAL_NOTICE = (
        "【资料范围说明：音频提取失败，官方字幕存在时间缺口；"
        "以下只依据可见字幕和PPT，不代表完整课堂内容。】"
    )

    # ── Public entry point ──────────────────────────────────────────────

    def run(self, course_id: str, course_title: str, lecture: dict,
            next_info: Optional[tuple[str, str]] = None) -> Optional[str]:
        """Process one lecture.  Returns the summary text or None.

        ``next_info``: ``(course_id, sub_id)`` of the next lecture, used to
        kick off its prefetch concurrently.  Pass ``None`` for the last
        lecture in the batch.
        """
        sub_id = str(lecture["sub_id"])
        sub_title = lecture.get("sub_title", sub_id)
        date = lecture.get("date", "")
        t_start = time.time()
        self._reporter.lecture_start(course_title, sub_title, date)

        existing = self._db.get_lecture(sub_id)
        # ── Phase A — short-circuit if a summary already exists ─────────
        if self._has_summary(existing):
            self._reporter.lecture_skip_v2_done(
                sub_title, len(existing["summary"])
            )
            self._schedule_next(next_info)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            # The return value feeds the email batch — suppress it when
            # this summary already went out so it isn't re-sent.
            if existing.get("emailed_at"):
                return None
            return existing["summary"]

        # ── Phase B — submit PPT pipeline (fetch + dedup, no OCR yet) ──
        # OCR is deferred (defer_ocr=True) so ASR in Phase D gets exclusive
        # CPU.  OCR will be submitted in Phase E (handle.drain()).
        ppt_handle = self._ppt.submit(
            self._client, course_id, sub_id, defer_ocr=True,
        )

        # ── Phase C — schedule next lecture's prefetch ─────────────────
        # Done BEFORE ASR so the next audio download can start filling its
        # AudioDownloader slot while we transcribe.  Both audio + image
        # prefetches are idempotent so this is safe to call any time.
        self._schedule_next(next_info)

        # ── Phase D — ASR, then official-subtitle fallback ────────────
        transcript, transcript_segments, partial_official = self._get_transcript(
            existing, course_id, sub_id,
        )
        if transcript is None:
            # _get_transcript already logged + persisted the skip reason.
            # Still drain the PPT handle: with defer_ocr the OCR jobs are
            # only submitted at drain() time, so skipping it would leave
            # the pages 'pending' forever and force the retry run to redo
            # download + dedup from scratch.
            ppt_handle.drain()
            return None

        # ── Phase E — drain remaining OCR work ─────────────────────────
        ppt_stats = ppt_handle.drain()
        _ = ppt_stats  # stats are emitted by PPTAsyncHandle.drain via reporter

        # ── Phase E2 — kick off next lecture's OCR (runs during LLM) ───
        # Images were prefetched in Phase C; prefetch_and_ocr spawns a
        # background thread that collects them, dedups and submits OCR,
        # so the whole thing genuinely overlaps with this lecture's LLM
        # wait.  Whatever isn't finished when the LLM returns is absorbed
        # by the next lecture's own PPTPipeline.submit().
        if next_info:
            next_course, next_sub = next_info
            self._ppt.prefetch_and_ocr(self._client, next_course, next_sub)

        # ── Phase F — bucketed-prompt LLM summary ──────────────────────
        if not transcript.strip():
            self._reporter.info("    Empty transcript, skipping summary.")
            self._release_audio(sub_id)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            return None

        summary = self._summarize(
            sub_id, course_title, transcript, transcript_segments,
            partial_official=partial_official,
        )
        if summary is None:
            self._release_audio(sub_id)
            return None

        # ── Phase G — persist + clear errors ───────────────────────────
        self._db.mark_processed(sub_id)
        self._db.clear_error(sub_id)

        # ── Phase H — release audio resources (ffmpeg + file) ──────────
        self._release_audio(sub_id)

        elapsed = time.time() - t_start
        self._reporter.lecture_done(course_title, sub_title, elapsed)
        return summary

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _has_summary(existing: dict | None) -> bool:
        return bool(
            existing
            and existing.get("summary")
        )

    def prefetch_first(self, course_id: str, sub_id: str) -> None:
        """Prefetch for the first lecture in the batch — same decision
        logic (skip the audio download when transcription won't need it)
        as the in-loop Phase C prefetch."""
        self._schedule_next((course_id, sub_id))

    def _schedule_next(self, next_info: Optional[tuple[str, str]]):
        if next_info is None:
            return
        next_course, next_sub = next_info
        # Image prefetch is always useful; the audio download — a full
        # ffmpeg pull of the lecture — only when the next lecture will
        # actually be ASR-transcribed.  Both are idempotent so repeated
        # invocations are no-ops; the audio side blocks on its semaphore
        # until a download slot frees.
        self._scheduler.prefetch_lecture(
            self._client, next_course, next_sub,
            audio=self._needs_audio(next_course, next_sub),
        )

    def _needs_audio(self, _course_id: str, sub_id: str) -> bool:
        """Prefetch audio unless a transcript is already stored."""
        existing = self._db.get_lecture(sub_id)
        return not bool(existing and existing.get("transcript"))

    @staticmethod
    def _official_transcript_usable(segments: list[dict] | None,
                                    max_gap_minutes: int = 20,
                                    duration_hint_s: int = 0) -> bool:
        """True if the official transcript is complete enough to use.

        Looks for a >``max_gap_minutes`` hole in three places: before the
        first segment (head truncation), between segments, and — when a
        duration hint is known (the last PPT screenshot offset, a lower
        bound on lecture length) — after the last segment (tail
        truncation)."""
        if not segments:
            return False
        max_gap_ms = max_gap_minutes * 60_000
        if segments[0]["start_ms"] > max_gap_ms:
            return False
        prev_end = segments[0]["end_ms"]
        for seg in segments[1:]:
            if seg["start_ms"] - prev_end > max_gap_ms:
                return False
            prev_end = max(prev_end, seg["end_ms"])
        if duration_hint_s and duration_hint_s * 1000 - prev_end > max_gap_ms:
            return False
        return True

    def _get_transcript(self, existing: dict | None, course_id: str,
                        sub_id: str) -> tuple[Optional[str], Optional[list], bool]:
        """Transcribe video audio first; use official subtitles on failure."""
        if existing and existing.get("transcript"):
            self._reporter.info(
                f"    Transcript exists "
                f"({len(existing['transcript'])} chars), "
                f"skipping transcription."
            )
            return (existing["transcript"], None,
                    existing["transcript"].startswith(self.PARTIAL_OFFICIAL_NOTICE))

        # Pull the audio handle.  ``schedule`` is idempotent — usually the
        # previous lecture already kicked it off (Phase C), but for the
        # first lecture in the batch we still need to fire it ourselves.
        downloader = self._scheduler.audio_downloader
        audio_error: Exception | None = None
        handle = None
        audio_result = None
        try:
            downloader.schedule(self._client, course_id, sub_id)
            handle = downloader.get(sub_id, timeout=120)
            if handle is not None:
                transcript, segments = self._transcriber.transcribe_tail(
                    handle.path, handle.process, handle.stderr_chunks,
                )
                if transcript.strip():
                    audio_result = transcript, segments
                else:
                    audio_error = RuntimeError("Audio transcription was empty")
        except Exception as e:
            audio_error = e
        if audio_result is not None:
            transcript, segments = audio_result
            self._db.update_transcript(sub_id, transcript)
            return transcript, segments, False
        self._release_audio(sub_id)

        if config.use_official_transcript_for(course_id):
            official_result = None
            try:
                official = self._client.get_transcript_segments(sub_id)
                if official:
                    duration_hint = self._db.get_max_ppt_created_sec(sub_id)
                    partial = not self._official_transcript_usable(
                        official, duration_hint_s=duration_hint,
                    )
                    text = " ".join(s["text"] for s in official)
                    if partial:
                        text = f"{self.PARTIAL_OFFICIAL_NOTICE}\n{text}"
                    self._reporter.info(
                        f"    Using official transcript after audio failure "
                        f"({len(text)} chars, {len(official)} segments"
                        f"{', incomplete' if partial else ''})"
                    )
                    official_result = text, official, partial
            except Exception as e:
                self._reporter.info(
                    f"    [Official transcript] fallback failed: "
                    f"{type(e).__name__}: {e}"
                )
            if official_result is not None:
                text, official, partial = official_result
                self._db.update_transcript(sub_id, text)
                return text, official, partial

        if handle is None and audio_error is None:
            # No playable video URL. Keep the frontend's gray no-video hint.
            self._reporter.lecture_skip_no_video(
                existing.get("sub_title", sub_id) if existing else sub_id
            )
            self._db.update_error(sub_id, "no_video", "no playable video URL")
            return None, None, False
        if isinstance(audio_error, NoAudioStreamError):
            self._reporter.info(f"    [SKIP] Video-only (no audio stream): {audio_error}")
            self._db.update_error(sub_id, "transcribe", str(audio_error))
            self._db.mark_processed(sub_id)
            return None, None, False
        if isinstance(audio_error, IncompleteAudioError):
            self._reporter.info(
                f"    [SKIP] Incomplete audio, will retry next run: {audio_error}"
            )
            self._db.update_error(sub_id, "transcribe", str(audio_error))
            return None, None, False
        assert audio_error is not None
        self._reporter.info(
            f"    [FAIL] Transcription error: {type(audio_error).__name__}: "
            f"{audio_error}"
        )
        self._db.update_error(sub_id, "transcribe", str(audio_error))
        raise audio_error

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None,
                   partial_official: bool = False) -> Optional[str]:
        try:
            kept_pages = self._db.get_done_ppt_pages(sub_id)
            prompt_text, mode = bucketer.assemble(
                transcript, transcript_segments, kept_pages,
            )
            if partial_official:
                prompt_text = (
                    self.PARTIAL_OFFICIAL_NOTICE + "\n"
                    "请只总结可见字幕与PPT支持的内容，明确缺失部分，"
                    "不要推测未提供的讲课内容。\n\n" + prompt_text
                )
            self._reporter.info(
                f"    [Time] Generating summary at "
                f"{time.strftime('%H:%M:%S')}"
                f" — mode={mode}, prompt={len(prompt_text)} chars"
            )
            summary, model_used = self._summarizer.summarize(
                course_title, prompt_text,
            )
            if partial_official:
                summary = (
                    "### 资料范围说明\n"
                    "本节音频不可用，官方字幕存在时间缺口；"
                    "以下笔记仅依据可见字幕和PPT。\n\n" + summary
                )
            self._reporter.info(
                f"    [OK] Summary by {model_used}: {len(summary)} chars"
            )
            self._db.update_summary(sub_id, summary, model_used)
            return summary
        except Exception as e:
            self._reporter.info(
                f"    [FAIL] Summarization error: {type(e).__name__}: {e}"
            )
            self._db.update_error(sub_id, "summarize", str(e))
            raise

    def _release_audio(self, sub_id: str):
        try:
            self._scheduler.audio_downloader.release(sub_id)
        except Exception as e:
            self._reporter.info(
                f"    [WARN] audio release failed: {type(e).__name__}: {e}"
            )


