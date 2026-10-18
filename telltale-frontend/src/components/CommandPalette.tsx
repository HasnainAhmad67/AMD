import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { MachineSummary } from "../api/types";
import { label } from "../lib/format";
export default function CommandPalette({ open, onClose, machines }: { open: boolean; onClose: () => void; machines: MachineSummary[] }) {
  const nav = useNavigate(); const [q, setQ] = useState(""); const [i, setI] = useState(0); const ref = useRef<HTMLInputElement>(null);
  useEffect(() => { if (open) { setQ(""); setI(0); setTimeout(() => ref.current?.focus(), 30); } }, [open]);
  if (!open) return null;
  const all = [{ l: "Fleet overview", h: "Page", to: "/" }, { l: "Alerts", h: "Page", to: "/alerts" }, { l: "New assessment", h: "Page", to: "/assess" }, { l: "Settings", h: "Page", to: "/settings" },
    ...machines.map((m) => ({ l: m.name, h: `${m.line} · ${label(m.health)}`, to: `/machines/${m.id}` }))];
  const items = all.filter((x) => x.l.toLowerCase().includes(q.toLowerCase()));
  const go = (to: string) => { nav(to); onClose(); };
  const key = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") onClose();
    else if (e.key === "ArrowDown") { e.preventDefault(); setI((x) => Math.min(items.length - 1, x + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setI((x) => Math.max(0, x - 1)); }
    else if (e.key === "Enter" && items[i]) go(items[i].to);
  };
  return (<div className="pal-bg" onClick={onClose}><div className="pal" role="dialog" aria-label="Command palette" onClick={(e) => e.stopPropagation()}>
    <input ref={ref} value={q} onChange={(e) => { setQ(e.target.value); setI(0); }} onKeyDown={key} placeholder="Search pages and machines…" aria-label="Search" />
    <ul>{items.map((x, k) => <li key={x.to} className={k === i ? "on" : ""} onMouseEnter={() => setI(k)} onClick={() => go(x.to)}><span>{x.l}</span><span className="lbl">{x.h}</span></li>)}{!items.length && <li className="lbl">No results</li>}</ul></div></div>);
}
