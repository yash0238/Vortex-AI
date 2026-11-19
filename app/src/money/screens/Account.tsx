import { useEffect, useState } from "react";
import { Cloud, LogOut, Trash2, X } from "lucide-react";
import { auth, deleteAccount, hasBackend, session } from "../api";
import { Sheet } from "../ui";

export function AccountSheet({ open, onClose, cloud, onSignedOut }:
  { open: boolean; onClose: () => void; cloud: { on: boolean; email?: string; status: string }; onSignedOut: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("register");
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => { if (open) { hasBackend().then(setOnline); setMsg(null); } }, [open]);

  const submit = async () => {
    setBusy(true); setMsg(null);
    try { await auth(mode, email, pw); onClose(); }
    catch (e) { setMsg((e as Error).message); }
    setBusy(false);
  };

  return (
    <Sheet open={open} onClose={onClose}>
      <div className="row between"><h1 className="title" style={{ fontSize: 24 }}><Cloud size={20} style={{ verticalAlign: -2 }} /> Sync across devices</h1>
        <button className="btn small ghost" onClick={onClose}><X size={16} /></button></div>
      {online === false && <p className="sub" style={{ marginTop: 12 }}>You're using VORTEX Money offline: everything is saved on this device only.
        Start the full app with <b>start_money.bat</b> to enable accounts and sync.</p>}
      {online && cloud.on && (
        <div className="stack" style={{ marginTop: 12 }}>
          <p className="sub">Signed in as <b>{cloud.email}</b> · {cloud.status}</p>
          <div className="row">
            <button className="btn ghost" onClick={() => { session.set(null); onSignedOut(); onClose(); }}><LogOut size={16} /> Sign out</button>
            <button className="btn ghost" onClick={async () => { if (confirm("Delete your account and all synced data? This cannot be undone.")) { await deleteAccount(); session.set(null); onSignedOut(); onClose(); } }}>
              <Trash2 size={16} /> Delete account</button>
          </div>
        </div>
      )}
      {online && !cloud.on && (
        <div className="stack" style={{ marginTop: 12 }}>
          <div className="seg"><button className={mode === "register" ? "on" : ""} onClick={() => setMode("register")}>Create account</button>
            <button className={mode === "login" ? "on" : ""} onClick={() => setMode("login")}>Sign in</button></div>
          <div className="field"><label>Email</label><input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" /></div>
          <div className="field"><label>Password (8+ characters)</label><input className="input" type="password" value={pw} onChange={(e) => setPw(e.target.value)}
            autoComplete={mode === "login" ? "current-password" : "new-password"} onKeyDown={(e) => e.key === "Enter" && submit()} /></div>
          <button className="btn" disabled={busy || pw.length < 8 || !email.includes("@")} onClick={submit}>{busy ? "Please wait…" : mode === "register" ? "Create account" : "Sign in"}</button>
          {msg && <p className="sub badtext">{msg}</p>}
          <p className="muted">Your plan is stored on this app's own server; passwords are hashed with scrypt. No bank or broker access is ever requested.</p>
        </div>
      )}
    </Sheet>
  );
}
