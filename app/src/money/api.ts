// Backend client. The app works fully offline (browser storage + direct public fund API); when the VORTEX Money
// backend is reachable (same origin /api), it adds accounts, cross-device sync and a cached fund-data proxy.

const BASE = "/api";
let backend: boolean | null = null;

export async function hasBackend(): Promise<boolean> {
  if (backend !== null) return backend;
  try {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), 1500);
    const r = await fetch(`${BASE}/health`, { signal: c.signal });
    clearTimeout(t);
    backend = r.ok && (await r.json()).ok === true;
  } catch { backend = false; }
  return backend;
}

const TOK = "vortexMoney.v1.token";
export const session = {
  get: (): { token: string; email: string } | null => { try { return JSON.parse(localStorage.getItem(TOK) || "null"); } catch { return null; } },
  set: (v: { token: string; email: string } | null) => { try { if (v) localStorage.setItem(TOK, JSON.stringify(v)); else localStorage.removeItem(TOK); } catch { /* ignore */ } },
};

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const s = session.get();
  const r = await fetch(`${BASE}${path}`, {
    ...init, headers: { "Content-Type": "application/json", ...(s ? { Authorization: `Bearer ${s.token}` } : {}), ...(init.headers || {}) },
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep */ }
    throw Object.assign(new Error(String(msg)), { status: r.status });
  }
  return r.json() as Promise<T>;
}

export async function auth(kind: "login" | "register", email: string, password: string) {
  const r = await call<{ token: string; user: { email: string } }>(`/auth/${kind}`, { method: "POST", body: JSON.stringify({ email, password }) });
  session.set({ token: r.token, email: r.user.email });
  return r;
}

export const pullState = () => call<{ state: unknown; version: number; updated: number | null }>("/me/state");
export const pushState = (state: unknown, version: number) =>
  call<{ version: number }>("/me/state", { method: "PUT", body: JSON.stringify({ state, version }) });
export const deleteAccount = () => call<{ deleted: boolean }>("/me", { method: "DELETE" });

// fund data: backend proxy (cached, rate-limited) or the public API directly
export async function fundJson(path: string): Promise<unknown> {
  if (await hasBackend()) {
    const p = path.startsWith("/mf/search") ? `/funds/search${path.slice("/mf/search".length)}` : `/funds/${path.split("/").pop()}`;
    return call(p);
  }
  const r = await fetch(`https://api.mfapi.in${path}`);
  if (!r.ok) throw new Error(`fund data: HTTP ${r.status}`);
  return r.json();
}
