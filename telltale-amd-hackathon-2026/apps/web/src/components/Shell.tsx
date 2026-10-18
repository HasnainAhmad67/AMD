import { useEffect, useRef, useState } from "react";
import { NavLink, Link, Outlet, useLocation } from "react-router-dom";
import { Activity, BellRing, ClipboardCheck, Cpu, LayoutDashboard, Search, Settings } from "lucide-react";
import { useHealth, useMachines, useOverview } from "../api/hooks";
import CommandPalette from "./CommandPalette";
import Toaster from "./Toaster";
export default function Shell() {
  const health = useHealth(); const overview = useOverview(); const machines = useMachines();
  const [open, setOpen] = useState(false); const [now, setNow] = useState(new Date());
  const { pathname } = useLocation(); const root = useRef<HTMLDivElement>(null);
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(t); }, []);
  useEffect(() => {
    let raf = 0; const mv = (e: MouseEvent) => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { root.current?.style.setProperty("--mx", `${e.clientX}px`); root.current?.style.setProperty("--my", `${e.clientY}px`); }); };
    const kd = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setOpen((o) => !o); } };
    window.addEventListener("mousemove", mv); window.addEventListener("keydown", kd);
    return () => { window.removeEventListener("mousemove", mv); window.removeEventListener("keydown", kd); cancelAnimationFrame(raf); };
  }, []);
  const ms = machines.data ?? [];
  const worst = [...ms].sort((a, b) => (b.failure_probability ?? -1) - (a.failure_probability ?? -1))[0];
  const open_ = overview.data?.open_alerts ?? 0; const online = health.isSuccess;
  const boot = health.data?.sensor_bootstrap;
  return (<div className="app" ref={root}>
    <div className="bg" aria-hidden><i className="o1" /><i className="o2" /><i className="spot" /></div>
    <aside className="side">
      <div className="logo"><Activity color="#22D3EE" /><span className="gt">Telltale</span></div>
      <div className="grp">Monitor</div>
      <nav className="navs">
        <NavLink to="/" end className="nav"><LayoutDashboard size={18} />Fleet overview</NavLink>
        {worst && <NavLink to={`/machines/${worst.id}`} className="nav"><Cpu size={18} />Machine detail</NavLink>}
        <NavLink to="/alerts" className="nav"><BellRing size={18} />Alerts</NavLink></nav>
      <div className="grp">Intelligence</div>
      <nav className="navs"><NavLink to="/assess" className="nav"><ClipboardCheck size={18} />New assessment</NavLink></nav>
      <div className="grp">System</div>
      <nav className="navs"><NavLink to="/settings" className="nav"><Settings size={18} />Settings</NavLink></nav>
      <div className="stat"><div><i className="dot" style={{ background: online ? "#34D399" : "#F43F5E", boxShadow: `0 0 8px ${online ? "#34D399" : "#F43F5E"}` }} />{online ? "Backend online" : health.isLoading ? "Connecting…" : "Backend offline"}</div>
        <div className="lbl" style={{ textTransform: "none" }}>{online ? `API v${health.data?.version}` : "Check VITE_API_PROXY"}</div></div>
    </aside>
    <div className="col">
      <header className="top">
        <button className="srch" onClick={() => setOpen(true)} aria-label="Open search"><Search size={16} />Search pages and machines…<kbd>Ctrl K</kbd></button>
        <div className="tr"><span className="mono" style={{ fontSize: 13, color: "var(--mu)" }}>{now.toLocaleTimeString()}</span>
          <Link to="/alerts" className={`ib ${open_ ? "rg" : ""}`} aria-label={`${open_} open alerts`}><BellRing size={18} />{open_ > 0 && <b>{open_}</b>}</Link></div>
      </header>
      <main className="main"><div key={pathname} className="page">
        {boot?.status === "running" && <div className="banner"><i /><b>Getting the risk model ready…</b><span>{boot.message || "Training on first start. This page updates by itself."}</span></div>}
        <Outlet /></div></main>
    </div>
    <CommandPalette open={open} onClose={() => setOpen(false)} machines={ms} /><Toaster />
  </div>);
}
