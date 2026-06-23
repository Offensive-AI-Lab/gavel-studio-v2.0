"""Tests for user authentication: registration, login, JWT validation."""
import pytest
import time


class TestRegistration:
    """User registration endpoint tests."""

    def test_register_success(self, client):
        suffix = int(time.time() * 1000) % 1000000
        res = client.post("/user/register", json={
            "username": f"newuser_{suffix}",
            "email": f"newuser_{suffix}@test.com",
            "password": "SecurePass123!",
        })
        assert res.status_code == 200
        data = res.json()
        assert "user_id" in data

    def test_register_duplicate_username(self, client, test_user):
        res = client.post("/user/register", json={
            "username": test_user["username"],
            "email": "different@test.com",
            "password": "SecurePass123!",
        })
        assert res.status_code in (400, 409, 500)

    def test_register_duplicate_email(self, client, test_user):
        res = client.post("/user/register", json={
            "username": "completely_different",
            "email": test_user["email"],
            "password": "SecurePass123!",
        })
        assert res.status_code in (400, 409, 500)

    def test_register_missing_fields(self, client):
        res = client.post("/user/register", json={"username": "incomplete"})
        assert res.status_code == 422

    def test_register_empty_username(self, client):
        res = client.post("/user/register", json={
            "username": "",
            "email": "empty@test.com",
            "password": "Pass123!",
        })
        assert res.status_code in (400, 422)

    def test_register_short_username(self, client):
        res = client.post("/user/register", json={
            "username": "ab",
            "email": "short@test.com",
            "password": "Pass123!",
        })
        assert res.status_code in (400, 422)


class TestLogin:
    """User login endpoint tests."""

    def test_login_success(self, client, test_user):
        res = client.post("/user/login", json={
            "email": test_user["email"],
            "password": test_user["password"],
        })
        assert res.status_code == 200
        data = res.json()
        assert "token" in data
        assert "user_id" in data

    def test_login_wrong_password(self, client, test_user):
        res = client.post("/user/login", json={
            "email": test_user["email"],
            "password": "WrongPassword!",
        })
        assert res.status_code in (400, 401, 403)

    def test_login_nonexistent_email(self, client):
        res = client.post("/user/login", json={
            "email": "nonexistent@nowhere.com",
            "password": "anything",
        })
        assert res.status_code in (400, 401, 404)

    def test_login_missing_fields(self, client):
        res = client.post("/user/login", json={"email": "x@x.com"})
        assert res.status_code == 422


class TestJWT:
    """JWT token validation tests."""

    def test_valid_token_access(self, client, auth_headers):
        res = client.get("/user/me", headers=auth_headers)
        assert res.status_code == 200

    def test_missing_token(self, client):
        res = client.get("/user/me")
        assert res.status_code in (401, 403)

    def test_invalid_token(self, client):
        res = client.get("/user/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert res.status_code == 401

    def test_expired_token(self, client):
        from utils.auth import create_access_token
        from datetime import timedelta

        expired = create_access_token({"sub": "1"}, expires_delta=timedelta(hours=-1))
        res = client.get("/user/me", headers={"Authorization": f"Bearer {expired}"})
        assert res.status_code == 401
