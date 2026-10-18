import { useState } from "react";
import { Link } from "react-router-dom";
import { Area, AreaChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BellRing, Boxes, ClipboardCheck, Cpu, Eye, ShieldCheck, Siren } from "lucide-react";
import { useMachines, useOverview } from "../api/hooks";
import type { Health } from "../api/types";
import { ErrorCard, Skeletons } from "../components/QueryState";
import RiskRing from "../components/RiskRing";
import Tilt from "../components/Tilt";
import { useCountUp } from "../lib/hooks";
import { label, pct, tone, toneColor } from "../lib/format";

const Count = ({ n }: { n: number }) => <>{useCountUp(n)}</>;
const Spark = ({ data, color }: { data: number[]; color: string }) => {
  if (data.length < 2) return null;
  const W = 110, H = 30, mx = Math.max(...data, 0.001);
  const pts = data.map((v, i) => `${((i / (data.length - 1)) * W).toFixed(1)},${(H - (v / mx) * (H - 4) - 2).toFixed(1)}`).join(" ");
  return <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden><polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" /></svg>;
};
type F = "all" | Health;

export default function FleetPage() {
  const machines = useMachines(); const overview = useOverview(); const [f, setF] = useState<F>("all");
  if (machines.isError) return <ErrorCard error={machines.error} onRetry={() => machines.refetch()} title="Could not load the fleet" />;
  if (!machines.data || !overview.data) return <Skeletons heights={[110, 96, 320]} />;
  const ms = machines.data, by = overview.data.by_health;
  const sorted = [...ms].sort((a, b) => (b.failure_probability ?? -1) - (a.failure_probability ?? -1));
  const shown = sorted.filter((m) => f === "all" || m.health === f);
  const worst = sorted[0]; const attention = sorted.filter((m) => m.health === "critical" || m.health === "watch").slice(0, 4);
  const head = by.critical ? `${by.critical} machine${by.critical > 1 ? "s need" : " needs"} attention` : by.watch ? `${by.watch} machine${by.watch > 1 ? "s" : ""} to watch` : "All machines healthy";
  const len = ms.length ? Math.min(...ms.map((m) => m.sparkline.length)) : 0;
  const avg = Array.from({ length: len }, (_, i) => ({ i, p: Math.round((ms.reduce((s, m) => s + m.sparkline[m.sparkline.length - len + i], 0) / ms.length) * 1000) / 10 }));
  const kpis = [["Monitored", overview.data.machines, "#22D3EE", Boxes], ["Healthy", by.ok, "#34D399", ShieldCheck], ["Watch", by.watch, "#F59E0B", Eye], ["Critical", by.critical, "#F43F5E", Siren]] as const;
  const donut = ([["Healthy", by.ok, "#34D399"], ["Watch", by.watch, "#F59E0B"], ["Critical", by.critical, "#F43F5E"], ["Unknown", by.unknown, "#8B9AAB"]] as const).filter((d) => d[1] > 0);
  return (<>
    <header className="ph"><div><h1>Fleet overview</h1><p>Machine health from sensor and sound signals</p></div>
      <div className="ph-r"><span className="badge healthy live">Live · refreshes every 3s</span><Link to="/assess" className="btn"><ClipboardCheck size={18} />New assessment</Link></div></header>
    <div className="card hero reveal"><div><div className="lbl">Fleet health</div><h2 className="gt" style={{ margin: "8px 0 6px", fontSize: 30 }}>{head}</h2>
      <div style={{ color: "var(--mu)" }}>{by.critical} critical · {by.watch} watch · {by.ok} healthy{by.unknown ? ` · ${by.unknown} unknown` : ""} · {overview.data.open_alerts} open alert{overview.data.open_alerts === 1 ? "" : "s"}</div></div>
      <div className="eq" aria-hidden>{Array.from({ length: 34 }, (_, i) => <i key={i} style={{ animationDelay: `${(i * 83) % 900}ms`, animationDuration: `${800 + ((i * 131) % 700)}ms` }} />)}</div></div>
    <div className="g12">{kpis.map(([l, v, c, Icon], i) => (<Tilt key={l} className="card c3 k reveal" style={{ animationDelay: `${i * 70}ms` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><div><div className="lbl">{l}</div><div style={{ fontSize: 34, fontWeight: 600, marginTop: 6, color: c }}><Count n={v} /></div></div><span className="ico" style={{ background: c + "22", color: c }}><Icon size={20} /></span></div>
      <div className="bar" style={{ margin: "14px 0 0" }}><i style={{ width: `${(v / Math.max(1, overview.data.machines)) * 100}%`, background: c }} /></div></Tilt>))}</div>
    <div className="g12">
      <div className="c8 stack"><div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}><span className="lbl">Machines · {shown.length}</span>
        <div className="chips">{(["all", "critical", "watch", "ok"] as F[]).map((k) => <button key={k} className={`chip ${f === k ? "on" : ""}`} onClick={() => setF(k)} style={{ height: 34 }}>{k === "all" ? "All" : label(k)}</button>)}</div></div>
        <div className="g2" style={{ flex: 1, alignContent: "stretch" }}>{shown.map((m, i) => {
          const t = tone(m.risk_level ?? m.health);
          return (<Tilt to={`/machines/${m.id}`} className={`card reveal ${m.health === "critical" ? "crit" : ""}`} style={{ animationDelay: `${i * 60}ms`, gap: 14 }} key={m.id}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}><div><b>{m.name}</b><div className="lbl" style={{ marginTop: 4 }}>{m.line}</div></div><span className={`badge ${tone(m.health)}`}>{label(m.health)}</span></div>
            <div style={{ display: "flex", gap: 16, alignItems: "center" }}><RiskRing p={m.failure_probability} tone={t} />
              <div className="mono" style={{ fontSize: 13, lineHeight: 1.8, color: "var(--mu)" }}>Failure risk {pct(m.failure_probability)}<br />Sound {m.acoustic ? label(m.acoustic.status) : "not analysed"}{m.tool_life ? <><br />Tool {Math.round(m.tool_life.remaining_use_min)} min left</> : null}</div></div>
            <Spark data={m.sparkline} color={toneColor[t]} /></Tilt>);
        })}{!shown.length && <div className="lbl">No machines in this filter</div>}</div></div>
      <div className="c4 stack">
        <div className="card reveal"><div className="lbl" style={{ marginBottom: 6 }}>Health distribution</div>
          <div style={{ position: "relative", height: 170 }}><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={donut.map((d) => ({ n: d[0], v: d[1] }))} dataKey="v" nameKey="n" innerRadius={52} outerRadius={76} paddingAngle={3} stroke="none">{donut.map((d) => <Cell key={d[0]} fill={d[2]} />)}</Pie><Tooltip contentStyle={{ background: "#121923", border: "1px solid #1F2A37" }} /></PieChart></ResponsiveContainer>
            <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", pointerEvents: "none" }}><div style={{ textAlign: "center" }}><div style={{ fontSize: 26, fontWeight: 600 }}>{overview.data.machines}</div><div className="lbl">machines</div></div></div></div></div>
        <div className="card reveal"><div className="row"><span className="lbl">Needs attention</span><Link to="/alerts" style={{ color: "var(--ac)", fontSize: 13 }}>View alerts</Link></div>
          {attention.length ? attention.map((m) => <Link key={m.id} to={`/machines/${m.id}`} className="qa" style={{ marginTop: 8 }}><i className="dot" style={{ background: toneColor[tone(m.health)], margin: 0 }} /><span style={{ flex: 1 }}>{m.name}<div className="lbl" style={{ marginTop: 2, textTransform: "none" }}>Risk {pct(m.failure_probability)} · {m.top_mode ?? "no failure mode flagged"}</div></span></Link>) : <span className="lbl" style={{ marginTop: 8 }}>Nothing needs attention</span>}</div></div>
      <div className="card c8 reveal"><div className="lbl" style={{ marginBottom: 6 }}>Average failure risk across the fleet (% over recent readings)</div>
        <div style={{ height: 190 }}>{avg.length > 1 ? <ResponsiveContainer width="100%" height="100%"><AreaChart data={avg}><defs><linearGradient id="fg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#22D3EE" stopOpacity={0.4} /><stop offset="100%" stopColor="#22D3EE" stopOpacity={0} /></linearGradient></defs><XAxis dataKey="i" hide /><YAxis hide domain={[0, "dataMax+2"]} /><Tooltip formatter={(v: number) => `${v}%`} contentStyle={{ background: "#121923", border: "1px solid #1F2A37" }} /><Area type="monotone" dataKey="p" stroke="#22D3EE" strokeWidth={2.4} fill="url(#fg)" /></AreaChart></ResponsiveContainer> : <span className="sub">Collecting readings…</span>}</div></div>
      <div className="card c4 reveal"><div className="lbl" style={{ marginBottom: 12 }}>Quick actions</div><div style={{ display: "grid", gap: 10 }}>
        <Link to="/assess" className="qa"><ClipboardCheck size={18} />Run a new assessment</Link>
        {worst && <Link to={`/machines/${worst.id}`} className="qa"><Cpu size={18} />Open the highest-risk machine</Link>}
        <Link to="/alerts" className="qa"><BellRing size={18} />Review alerts</Link></div></div>
    </div></>);
}
