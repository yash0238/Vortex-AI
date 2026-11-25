"""VORTEX Money production backend.

    .venv\\Scripts\\python.exe -m uvicorn money_api.app:app --host 127.0.0.1 --port 8100

Serves the built consumer app (app/dist) and a small JSON API:
  GET  /api/health                       liveness + version
  GET  /api/engine                       the scenario engine (CG-NSDE output), cacheable
  GET  /api/funds/search?q=              mutual fund search (AMFI data via mfapi.in), cached
  GET  /api/funds/{code}                 a fund's daily NAV history + metadata, cached 12 h
  POST /api/auth/register {email, password}   -> {token, user}
  POST /api/auth/login    {email, password}   -> {token, user}
  GET  /api/me/state                     the user's saved plan (profile + progress), with a version number
  PUT  /api/me/state {state, version}    save; rejects stale writes (409) so two devices cannot clobber each other
  DELETE /api/me                         delete the account and its data

Security: scrypt password hashing (stdlib), HMAC-SHA256 signed expiring tokens, per-IP rate limits, strict CORS
(same origin by default), security headers, request-size limits, parameterised SQL only.  Storage: SQLite (one file,
path from MONEY_DB).  Configuration through environment variables (see Settings).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent


class Settings:
    def __init__(self) -> None:
        self.db = Path(os.environ.get("MONEY_DB", ROOT / "money_api" / "data" / "money.sqlite3"))
        self.dist = Path(os.environ.get("MONEY_DIST", ROOT / "app" / "dist"))
        self.engine = Path(os.environ.get("MONEY_ENGINE", ROOT / "app" / "public" / "money" / "engine.json"))
        self.origins = [o for o in os.environ.get("MONEY_CORS", "").split(",") if o]
        self.token_days = float(os.environ.get("MONEY_TOKEN_DAYS", "30"))
        self.fund_ttl = float(os.environ.get("MONEY_FUND_TTL_H", "12")) * 3600
        self.mfapi = os.environ.get("MONEY_MFAPI", "https://api.mfapi.in")
        self.offline = os.environ.get("MONEY_OFFLINE", "") == "1"         # tests: never call the network
        secret = os.environ.get("MONEY_SECRET")
        if not secret:
            f = self.db.parent / ".secret"
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists():
                f.write_text(secrets.token_hex(32))
            secret = f.read_text().strip()
        self.secret = secret.encode()


S = Settings()


# ------------------------------------------------------------------ storage
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw TEXT NOT NULL,
                                  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS state (uid INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                                  body TEXT NOT NULL, version INTEGER NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched REAL NOT NULL);
"""
_lock = threading.Lock()


@contextmanager
def db():
    S.db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(S.db, timeout=10)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


with db() as _c:
    _c.executescript(SCHEMA)


# ------------------------------------------------------------------ passwords and tokens
def hash_pw(pw: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_pw(pw: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        return hmac.compare_digest(hash_pw(pw, bytes.fromhex(salt)).split("$")[2], h)
    except ValueError:
        return False


def b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(uid: int) -> str:
    body = b64(json.dumps({"uid": uid, "exp": time.time() + S.token_days * 86400}).encode())
    sig = b64(hmac.new(S.secret, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_token(tok: str) -> int | None:
    try:
        body, sig = tok.split(".")
        if not hmac.compare_digest(sig, b64(hmac.new(S.secret, body.encode(), hashlib.sha256).digest())):
            return None
        p = json.loads(unb64(body))
        return int(p["uid"]) if p["exp"] > time.time() else None
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def current_user(authorization: str = Header(default="")) -> int:
    uid = read_token(authorization.removeprefix("Bearer ").strip()) if authorization.startswith("Bearer ") else None
    if uid is None:
        raise HTTPException(401, "sign in required")
    with db() as c:
        if not c.execute("SELECT 1 FROM users WHERE id = ?", (uid,)).fetchone():
            raise HTTPException(401, "account not found")
    return uid


# ------------------------------------------------------------------ rate limiting (per IP, sliding window)
_hits: dict[tuple[str, str], deque] = defaultdict(deque)


def limit(bucket: str, per_min: int):
    def dep(request: Request) -> None:
        ip = request.client.host if request.client else "?"
        q, now = _hits[(bucket, ip)], time.time()
        with _lock:
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= per_min:
                raise HTTPException(429, "too many requests, try again in a minute")
            q.append(now)
    return dep


# ------------------------------------------------------------------ app
app = FastAPI(title="VORTEX Money API", version=VERSION, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=1024)
if S.origins:
    app.add_middleware(CORSMiddleware, allow_origins=S.origins, allow_methods=["GET", "POST", "PUT", "DELETE"],
                       allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def guard(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > 256_000:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return resp


@app.get("/api/health")
def health():
    return {"ok": True, "version": VERSION, "engine": S.engine.exists()}


@app.get("/api/engine")
def engine(response: Response):
    if not S.engine.exists():
        raise HTTPException(503, "scenario engine not built: run python -m crisisbench.export_money")
    response.headers["Cache-Control"] = "public, max-age=3600"
    return FileResponse(S.engine, media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})


# ------------------------------------------------------------------ mutual funds (cached proxy)
def cached_fetch(key: str, url: str, ttl: float):
    with db() as c:
        row = c.execute("SELECT body, fetched FROM cache WHERE key = ?", (key,)).fetchone()
    if row and time.time() - row[1] < ttl:
        return json.loads(row[0])
    if S.offline:
        if row:
            return json.loads(row[0])
        raise HTTPException(503, "offline mode and nothing cached")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vortex-money/1.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode())
    except Exception as e:                                          # serve stale data rather than fail
        if row:
            return json.loads(row[0])
        raise HTTPException(502, f"fund data source unavailable ({type(e).__name__})")
    with db() as c:
        c.execute("INSERT OR REPLACE INTO cache (key, body, fetched) VALUES (?, ?, ?)", (key, json.dumps(data), time.time()))
    return data


@app.get("/api/funds/search", dependencies=[Depends(limit("funds", 60))])
def fund_search(q: str = ""):
    q = q.strip()
    if len(q) < 3 or len(q) > 60:
        raise HTTPException(422, "query must be 3-60 characters")
    data = cached_fetch(f"search:{q.lower()}", f"{S.mfapi}/mf/search?q={urllib.parse.quote(q)}", 24 * 3600)
    return data[:25] if isinstance(data, list) else []


@app.get("/api/funds/{code}", dependencies=[Depends(limit("funds", 60))])
def fund(code: int):
    if not (100 <= code <= 9_999_999):
        raise HTTPException(422, "invalid scheme code")
    data = cached_fetch(f"fund:{code}", f"{S.mfapi}/mf/{code}", S.fund_ttl)
    if not isinstance(data, dict) or not data.get("data"):
        raise HTTPException(404, "fund not found")
    return data


# ------------------------------------------------------------------ accounts and sync
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")


class Creds(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)


class StateIn(BaseModel):
    state: dict
    version: int = Field(ge=0)


@app.post("/api/auth/register", dependencies=[Depends(limit("auth", 10))])
def register(c: Creds):
    email = c.email.strip().lower()
    if not EMAIL.match(email):
        raise HTTPException(422, "enter a valid email")
    try:
        with db() as con:
            cur = con.execute("INSERT INTO users (email, pw, created) VALUES (?, ?, ?)", (email, hash_pw(c.password), time.time()))
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "an account with this email already exists")
    return {"token": make_token(uid), "user": {"id": uid, "email": email}}


@app.post("/api/auth/login", dependencies=[Depends(limit("auth", 10))])
def login(c: Creds):
    email = c.email.strip().lower()
    with db() as con:
        row = con.execute("SELECT id, pw FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not check_pw(c.password, row[1]):
        raise HTTPException(401, "wrong email or password")
    return {"token": make_token(row[0]), "user": {"id": row[0], "email": email}}


@app.get("/api/me/state")
def get_state(uid: int = Depends(current_user)):
    with db() as con:
        row = con.execute("SELECT body, version, updated FROM state WHERE uid = ?", (uid,)).fetchone()
    return {"state": json.loads(row[0]) if row else None, "version": row[1] if row else 0, "updated": row[2] if row else None}


@app.put("/api/me/state")
def put_state(s: StateIn, uid: int = Depends(current_user)):
    body = json.dumps(s.state, separators=(",", ":"))
    if len(body) > 200_000:
        raise HTTPException(413, "state too large")
    with _lock, db() as con:
        row = con.execute("SELECT version FROM state WHERE uid = ?", (uid,)).fetchone()
        cur = row[0] if row else 0
        if s.version != cur:
            raise HTTPException(409, f"stale version {s.version}, server has {cur}")
        con.execute("INSERT OR REPLACE INTO state (uid, body, version, updated) VALUES (?, ?, ?, ?)",
                    (uid, body, cur + 1, time.time()))
    return {"version": cur + 1}


@app.delete("/api/me")
def delete_me(uid: int = Depends(current_user)):
    with db() as con:
        con.execute("DELETE FROM state WHERE uid = ?", (uid,))
        con.execute("DELETE FROM users WHERE id = ?", (uid,))
    return {"deleted": True}


# ------------------------------------------------------------------ the web app itself
if S.dist.exists():
    app.mount("/assets", StaticFiles(directory=S.dist / "assets"), name="assets")
    if (S.dist / "money").exists():
        app.mount("/money", StaticFiles(directory=S.dist / "money"), name="money")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(S.dist / "money.html", headers={"Cache-Control": "no-cache"})

    @app.get("/{name}", include_in_schema=False)
    def static_file(name: str):
        f = (S.dist / name).resolve()
        if f.parent != S.dist.resolve() or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)
