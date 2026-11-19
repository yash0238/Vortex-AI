"""Tests for the VORTEX Money backend. No network: MONEY_OFFLINE=1 and a pre-filled cache."""
import importlib
import json
import os
import sqlite3
import time

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MONEY_DB", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("MONEY_SECRET", "test-secret")
    monkeypatch.setenv("MONEY_OFFLINE", "1")
    monkeypatch.setenv("MONEY_DIST", str(tmp_path / "nodist"))
    import money_api.app as m
    m = importlib.reload(m)
    from fastapi.testclient import TestClient
    c = TestClient(m.app)
    c.mod = m
    return c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.headers["x-content-type-options"] == "nosniff"


def test_register_login_state_roundtrip(client):
    r = client.post("/api/auth/register", json={"email": "A@x.com", "password": "longpassword"})
    assert r.status_code == 200
    tok = r.json()["token"]
    assert client.post("/api/auth/register", json={"email": "a@x.com", "password": "longpassword"}).status_code == 409
    assert client.post("/api/auth/login", json={"email": "a@x.com", "password": "wrongpassword"}).status_code == 401
    tok2 = client.post("/api/auth/login", json={"email": "a@x.com", "password": "longpassword"}).json()["token"]
    h = {"Authorization": f"Bearer {tok2}"}
    assert client.get("/api/me/state", headers=h).json() == {"state": None, "version": 0, "updated": None}
    assert client.put("/api/me/state", headers=h, json={"state": {"profile": {"name": "A"}}, "version": 0}).json() == {"version": 1}
    # a second device holding version 0 must not overwrite
    assert client.put("/api/me/state", headers={"Authorization": f"Bearer {tok}"}, json={"state": {}, "version": 0}).status_code == 409
    got = client.get("/api/me/state", headers=h).json()
    assert got["state"]["profile"]["name"] == "A" and got["version"] == 1
    assert client.delete("/api/me", headers=h).json() == {"deleted": True}
    assert client.get("/api/me/state", headers=h).status_code == 401


def test_auth_rejects_bad_input_and_tokens(client):
    assert client.post("/api/auth/register", json={"email": "nope", "password": "longpassword"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "b@x.com", "password": "short"}).status_code == 422
    assert client.get("/api/me/state").status_code == 401
    assert client.get("/api/me/state", headers={"Authorization": "Bearer abc.def"}).status_code == 401
    m = client.mod
    expired = m.make_token(1)
    body, sig = expired.split(".")
    forged = m.b64(json.dumps({"uid": 1, "exp": time.time() + 99}).encode()) + "." + sig
    assert m.read_token(forged) is None


def test_password_hash_is_salted():
    import money_api.app as m
    a, b = m.hash_pw("secretpass"), m.hash_pw("secretpass")
    assert a != b and m.check_pw("secretpass", a) and not m.check_pw("other", a)


def test_fund_proxy_uses_cache_offline(client):
    m = client.mod
    with sqlite3.connect(m.S.db) as con:
        con.execute("INSERT INTO cache (key, body, fetched) VALUES (?, ?, ?)",
                    ("fund:120503", json.dumps({"meta": {"scheme_name": "Test Fund"}, "data": [{"date": "01-01-2024", "nav": "10"}]}), time.time()))
    r = client.get("/api/funds/120503")
    assert r.status_code == 200 and r.json()["meta"]["scheme_name"] == "Test Fund"
    assert client.get("/api/funds/12").status_code == 422
    assert client.get("/api/funds/999999").status_code == 503          # offline, nothing cached
    assert client.get("/api/funds/search?q=ab").status_code == 422


def test_rate_limit(client):
    codes = [client.post("/api/auth/login", json={"email": "z@x.com", "password": "longpassword"}).status_code for _ in range(12)]
    assert 429 in codes
