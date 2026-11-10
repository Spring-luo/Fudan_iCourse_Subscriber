"""Offline checks for the UIS second-factor response contract."""

import os
import unittest
from unittest.mock import patch

import pyotp

from src.api.webvpn import (
    SecondFactorRequiredError,
    _complete_password_auth,
)


PASSWORD_PAYLOAD = {
    "authChainCode": "first",
    "entityId": "https://course.example.test",
    "requestType": "chain_type",
    "lck": "lock",
    "authPara": {"loginName": "student", "password": "encrypted"},
}
MFA_RESPONSE = {
    "code": 200,
    "pageLevelNo": 2,
    "second": True,
    "moduleCodes": ["userAndOtp"],
    "requestNumber": "request-1",
    "authChainCode": "second",
}


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"code": 200, "loginToken": "second-token"}


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response()


class TotpAuthTests(unittest.TestCase):
    def test_direct_token_needs_no_second_request(self):
        session = _Session()
        token = _complete_password_auth(
            session, {"code": 200, "loginToken": "direct"},
            PASSWORD_PAYLOAD, "https://id.example.test/authExecute", {}, "iCourse",
        )
        self.assertEqual(token, "direct")
        self.assertEqual(session.calls, [])

    def test_mfa_posts_generated_code_and_challenge_fields(self):
        secret = "JBSWY3DPEHPK3PXP"
        session = _Session()
        with patch.dict(os.environ, {"UIS_TOTP_SECRET": secret}):
            token = _complete_password_auth(
                session, MFA_RESPONSE, PASSWORD_PAYLOAD,
                "https://id.example.test/authExecute", {}, "iCourse",
            )
        self.assertEqual(token, "second-token")
        challenge = session.calls[0][1]["json"]
        self.assertEqual(challenge["authModuleCode"], "userAndOtp")
        self.assertEqual(challenge["authChainCode"], "second")
        self.assertEqual(challenge["requestNumber"], "request-1")
        self.assertEqual(challenge["authPara"]["loginName"], "student")
        self.assertTrue(pyotp.TOTP(secret).verify(challenge["authPara"]["otpCode"]))
        self.assertNotIn("password", challenge["authPara"])

    def test_mfa_without_secret_fails_before_second_request(self):
        session = _Session()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SecondFactorRequiredError, "UIS_TOTP_SECRET"):
                _complete_password_auth(
                    session, MFA_RESPONSE, PASSWORD_PAYLOAD,
                    "https://id.example.test/authExecute", {}, "iCourse",
                )
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
