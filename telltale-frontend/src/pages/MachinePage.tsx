import { Link, useParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ClipboardCheck } from "lucide-react";
import { api } from "../api/client";
import { useMachine, useSensorSchema } from "../api/hooks";
import type { MaintenanceAction, SensorField } from "../api/types";
import { ErrorCard, Skeletons } from "../components/QueryState";
import Recs from "../components/Recs";
import RiskRing from "../components/RiskRing";
import { ago, label, pct, title, tone, toneColor } from "../lib/format";
import { toast } from "../lib/toast";

const FALLBACK: Record<SensorField, [number, number]> = { air_temperature_k: [295, 310], process_temperature_k: [305, 315], rotational_speed_rpm: [1200, 2800], torque_nm: [5, 75], tool_wear_min: [0, 250] };
const SIGNALS: { k: SensorField; name: string; unit: string }[] = [
  { k: "air_temperature_k", name: "Air temperature", unit: "K" }, { k: "process_temperature_k", name: "Process temperature", unit: "K" },
  { k: "rotational_speed_rpm", name: "Rotational speed", unit: "rpm" }, { k: "torque_nm", name: "Torque", unit: "Nm" }, { k: "tool_wear_min", name: "Tool wear", unit: "min" }];
const ACTIONS: { id: MaintenanceAction; name: string }[] = [{ id: "replace_tool", name: "Replace tool" }, { id: "service_cooling", name: "Service cooling" }, { id: "reduce_load", name: "Reduce load" }, { id: "inspect_drive", name: "Inspect drive" }];

const Bar = ({ n, v, p, c }: { n: string; v: string; p: number; c: string }) => (<><div className="row"><span>{n}</span><span className="mono" style={{ color: c }}>{v}</span></div><div className="bar"><i style={{ width: `${Math.min(100, Math.max(2, p * 100))}%`, background: c }} /></div></>);

export default function MachinePage() {
  const { id = "" } = useParams(); const q = useMachine(id); const schema = useSensorSchema(); const qc = useQueryClient();
  const log = useMutation({ mutationFn: (a: MaintenanceAction) => api.maintenance(id, a), onSuccess: (_d, a) => { toast(`Logged: ${title(a)}`); qc.invalidateQueries({ queryKey: ["machine", id] }); qc.invalidateQueries({ queryKey: ["machines"] }); }, onError: (e) => toast(e instanceof Error ? e.message : "Could not log the action") });
  if (q.isError) return <ErrorCard error={q.error} onRetry={() => q.refetch()} title="Could not load this machine" />;
  if (!q.data) return <Skeletons heights={[60, 300, 300]} />;
  const m = q.data, a = m.assessment, t = tone(m.risk_level ?? m.health), r = m.latest_reading;
  const hist = m.history.filter((h) => h.failure_probability != null).map((h, i) => ({ i, t: new Date(h.t).toLocaleTimeString(), p: Math.round((h.failure_probability as number) * 1000) / 10 }));
  const top = a ? [...a.contributions].sort((x, y) => Math.abs(y.contribution) - Math.abs(x.contribution)).slice(0, 6) : [];
  const mx = Math.max(0.001, ...top.map((c) => Math.abs(c.contribution)));
  return (<>
    <header className="ph"><div><div className="lbl">Fleet / {m.line}</div><h1>{m.name}</h1><p>Quality {m.product_quality} · simulated scenario: {title(m.simulated_scenario)}</p></div>
      <div className="ph-r"><span className={`badge ${tone(m.health)}`}>{label(m.health)}</span><Link to={`/assess?machine=${m.id}`} className="btn"><ClipboardCheck size={18} />Run assessment</Link></div></header>
    {m.health !== "ok" && m.health !== "unknown" && <div className="banner reveal" style={{ borderColor: toneColor[tone(m.health)] + "99", background: toneColor[tone(m.health)] + "1a" }}><i style={{ background: toneColor[tone(m.health)] }} /><b style={{ color: toneColor[tone(m.health)] }}>{label(m.health)}</b><span>Failure risk {pct(m.failure_probability)}{m.top_mode ? ` · most likely mode: ${m.top_mode}` : ""}</span></div>}
    <div className="g12">
      <div className="card c3 ctr reveal"><div className="lbl">Failure risk</div><RiskRing p={m.failure_probability} tone={t} size={160} /><span className={`badge ${t}`} style={{ alignSelf: "center" }}>{label(m.risk_level)}</span>
        <div className="sub">{a ? `Alert threshold ${pct(a.decision_threshold)}` : "Waiting for the first reading"}</div></div>
      <div className="card c9 reveal"><div className="lbl" style={{ marginBottom: 8 }}>Failure risk over time (%)</div>
        <div style={{ flex: 1, minHeight: 200 }}>{hist.length > 1 ? <ResponsiveContainer width="100%" height="100%"><AreaChart data={hist}><defs><linearGradient id="hg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={toneColor[t]} stopOpacity={0.4} /><stop offset="100%" stopColor={toneColor[t]} stopOpacity={0} /></linearGradient></defs><XAxis dataKey="t" hide /><YAxis hide domain={[0, "dataMax+2"]} /><Tooltip formatter={(v: number) => `${v}%`} contentStyle={{ background: "#121923", border: "1px solid #1F2A37" }} /><Area type="monotone" dataKey="p" stroke={toneColor[t]} strokeWidth={2.4} fill="url(#hg)" /></AreaChart></ResponsiveContainer> : <span className="sub">Collecting readings…</span>}</div></div>
      <div className="card c4 reveal"><div className="lbl" style={{ marginBottom: 14 }}>Live signals</div>
        {r ? SIGNALS.map((s) => { const [lo, hi] = schema.data?.feature_ranges?.[s.k] ? [schema.data.feature_ranges[s.k].p01, schema.data.feature_ranges[s.k].p99] : FALLBACK[s.k]; const v = r[s.k];
          return <Bar key={s.k} n={s.name} v={`${v.toFixed(s.k === "rotational_speed_rpm" || s.k === "tool_wear_min" ? 0 : 1)} ${s.unit}`} p={(v - lo) / Math.max(1e-6, hi - lo)} c="#22D3EE" />; }) : <span className="sub">No reading yet.</span>}
        <div className="sub">Bars show where each value sits in the typical training range.</div></div>
      <div className="card c4 reveal"><div className="lbl" style={{ marginBottom: 14 }}>What drives the risk</div>
        {top.length ? top.map((c) => <Bar key={c.feature} n={c.label} v={`${c.contribution > 0 ? "+" : ""}${c.contribution.toFixed(2)}`} p={Math.abs(c.contribution) / mx} c={c.contribution > 0 ? "#F43F5E" : "#34D399"} />) : <span className="sub">No explanation yet.</span>}
        <div className="sub">SHAP contribution per feature (red raises risk, green lowers it).</div></div>
      <div className="card c4 reveal"><div className="lbl" style={{ marginBottom: 14 }}>Failure modes</div>
        {a?.failure_modes.length ? a.failure_modes.map((f) => <Bar key={f.code} n={`${f.name}${f.flagged ? " · flagged" : ""}`} v={pct(f.probability, 1)} p={f.probability} c={f.flagged ? "#F43F5E" : "#8B9AAB"} />) : <span className="sub">None reported.</span>}
        {a?.failure_modes.some((f) => !f.reliable) && <div className="sub">Modes marked unreliable have a weak validation score and never raise flags.</div>}</div>
      <div className="card c6 reveal"><div className="lbl" style={{ marginBottom: 12 }}>Recommended actions</div>{a?.recommendations.length ? <Recs items={a.recommendations} /> : <span className="sub">No actions recommended right now.</span>}</div>
      <div className="c6 stack">
        <div className="card reveal"><div className="lbl" style={{ marginBottom: 10 }}>Tool life</div>{m.tool_life ? <div className="stackc" style={{ gap: 8 }}><Bar n="Wear" v={`${Math.round(m.tool_life.wear_min)} / ${Math.round(m.tool_life.replace_at_min)} min`} p={m.tool_life.wear_min / Math.max(1, m.tool_life.replace_at_min)} c={m.tool_life.remaining_use_min < 30 ? "#F59E0B" : "#34D399"} />
          <div className="sub">{Math.round(m.tool_life.remaining_use_min)} min of use left{m.tool_life.eta_minutes != null ? ` · replace in about ${Math.round(m.tool_life.eta_minutes)} min` : ""}</div></div> : <span className="sub">Not available.</span>}</div>
        <div className="card reveal"><div className="lbl" style={{ marginBottom: 10 }}>Latest sound check</div>{m.acoustic ? <div className="stackc" style={{ gap: 6 }}><div className="gap8"><span className={`badge ${tone(m.acoustic.status)}`}>{label(m.acoustic.status)}</span><span className="sub">score {m.acoustic.score.toFixed(2)} · severity {m.acoustic.severity.toFixed(2)}×</span></div><div className="sub">{m.acoustic.filename ?? "recording"} · {ago(m.acoustic.analyzed_at)}</div></div> : <span className="sub">No sound analysed for this machine yet.</span>}</div>
        <div className="card reveal"><div className="lbl" style={{ marginBottom: 10 }}>Log maintenance</div><div className="gap8">{ACTIONS.map((x) => <button key={x.id} className="ghost" disabled={log.isPending} onClick={() => log.mutate(x.id)}>{x.name}</button>)}</div></div></div>
    </div></>);
}
