import type { AcousticAnalysis, Assessment, FaultDiagnosis, ModalityRecord, SensorAssessment } from "../api/types";
import { ago, label, pct, shortId, title, tone, toneColor } from "../lib/format";
import { toast } from "../lib/toast";
import AiPanel from "./AiPanel";
import { FrameErrorChart, SpectrogramChart, WaveformChart } from "./DataViz";
import Recs from "./Recs";
import RiskRing from "./RiskRing";

const Bar = ({ n, v, p, c }: { n: string; v: string; p: number; c: string }) => (<><div className="row"><span>{n}</span><span className="mono" style={{ color: c }}>{v}</span></div><div className="bar"><i style={{ width: `${Math.min(100, Math.max(2, p * 100))}%`, background: c }} /></div></>);

function Foot({ m }: { m: ModalityRecord }) {
  const id = m.model_identity;
  return (<div className="stackc" style={{ gap: 8 }}>
    <div className="gap8"><span className="badge unknown">{m.source_mode}</span>{m.illustrative && <span className="badge inspect">Illustrative data</span>}
      <span className="sub">{id ? [id.name, id.version && `v${id.version}`, id.training_dataset && `trained on ${id.training_dataset}`].filter(Boolean).join(" · ") : "Model identity not recorded"}</span></div>
    {m.limitations.length > 0 && <details><summary>Model limitations ({m.limitations.length})</summary><ul className="ul">{m.limitations.map((x, i) => <li key={i}>{x}</li>)}</ul></details>}</div>);
}

function Sensor({ m, o }: { m: ModalityRecord; o: SensorAssessment }) {
  const t = tone(o.risk_level); const top = [...o.contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 6);
  const mx = Math.max(0.001, ...top.map((c) => Math.abs(c.contribution))); const flagged = o.failure_modes.filter((f) => f.flagged);
  return (<div className="card reveal" style={{ gap: 14 }}>
    <div className="row" style={{ marginBottom: 0 }}><b>Sensor model</b><span className={`badge ${t}`}>{label(o.risk_level)}</span></div>
    <div className="gap8" style={{ gap: 16 }}><RiskRing p={o.failure_probability} tone={t} size={92} /><div className="sub">Failure probability from the sensor readings. Alert threshold {pct(o.decision_threshold)}.</div></div>
    {flagged.length > 0 && <div className="gap8"><span className="lbl">Flagged failure modes</span>{flagged.map((f) => <span key={f.code} className="badge critical">{f.name} {pct(f.probability)}</span>)}</div>}
    <div><div className="lbl" style={{ marginBottom: 8 }}>What drives the risk (red raises, green lowers)</div>
      {top.map((c) => <Bar key={c.feature} n={c.label} v={`${c.contribution > 0 ? "+" : ""}${c.contribution.toFixed(2)}`} p={Math.abs(c.contribution) / mx} c={c.contribution > 0 ? "#F43F5E" : "#34D399"} />)}</div>
    {o.warnings.length > 0 && <ul className="ul">{o.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}
    <Recs items={o.recommendations} /><Foot m={m} /></div>);
}

function Acoustic({ m, o }: { m: ModalityRecord; o: AcousticAnalysis }) {
  const t = tone(o.status);
  return (<div className="card reveal" style={{ gap: 14 }}>
    <div className="row" style={{ marginBottom: 0 }}><b>Sound anomaly model</b><span className={`badge ${t}`}>{label(o.status)}</span></div>
    <div className="gap8" style={{ gap: 18 }}>
      <div><div className="lbl">Anomaly score</div><div style={{ fontSize: 26, fontWeight: 600, color: toneColor[t] }}>{o.score.toFixed(2)}</div><div className="sub">threshold {o.threshold.toFixed(2)} · severity {o.severity.toFixed(2)}×</div></div>
      {o.p_value != null && <div><div className="lbl">p-value</div><div style={{ fontSize: 26, fontWeight: 600 }}>{o.p_value.toFixed(3)}</div><div className="sub">{o.unusual_percentile != null ? `more unusual than ${o.unusual_percentile.toFixed(0)}% of normal clips` : "vs normal reference"}</div></div>}
      <div><div className="lbl">Clip</div><div className="mono" style={{ fontSize: 14 }}>{o.duration_s.toFixed(1)} s</div><div className="sub">peak at {o.peak_time_s.toFixed(1)} s</div></div></div>
    <div><div className="lbl" style={{ marginBottom: 6 }}>Waveform</div><WaveformChart wave={o.waveform} peak={o.peak_time_s} /></div>
    <div><div className="lbl" style={{ marginBottom: 6 }}>Spectrogram</div><SpectrogramChart spec={o.spectrogram} peak={o.peak_time_s} duration={o.duration_s} /></div>
    <div><div className="lbl" style={{ marginBottom: 6 }}>Anomaly score over time (dashed = alert level)</div><FrameErrorChart series={o.frame_errors} threshold={o.threshold} peak={o.peak_time_s} /></div>
    <Recs items={o.recommendations} /><Foot m={m} /></div>);
}

function Bearing({ m, o }: { m: ModalityRecord; o: FaultDiagnosis }) {
  const top = [...o.probabilities].sort((a, b) => b.probability - a.probability).slice(0, 5);
  return (<div className="card reveal" style={{ gap: 14 }}>
    <div className="row" style={{ marginBottom: 0 }}><b>Bearing fault model</b>{o.uncertain ? <span className="badge inspect">Uncertain</span> : <span className="badge unknown">Classified</span>}</div>
    <div><div className="lbl">Most likely condition</div><div style={{ fontSize: 22, fontWeight: 600 }}>{o.label}</div><div className="sub">confidence {pct(o.confidence)} · segments agree {pct(o.segment_agreement)} ({o.segments})</div></div>
    {o.prediction_set_labels.length > 1 && <div className="gap8"><span className="lbl">Cannot rule out</span>{o.prediction_set_labels.map((l) => <span key={l} className="badge unknown">{l}</span>)}</div>}
    <div>{top.map((p) => <Bar key={p.class_name} n={p.label} v={pct(p.probability, 1)} p={p.probability} c="#22D3EE" />)}</div>
    <div className="sub">Shaft speed used: {o.rpm_used != null ? `${Math.round(o.rpm_used)} rpm` : "unknown"}{o.rpm_assumed ? " (assumed default)" : ""}</div>
    <Recs items={o.recommendations} /><Foot m={m} /></div>);
}

export default function AssessmentView({ a, name }: { a: Assessment; name?: string }) {
  const t = tone(a.overall_status); const c = a.evidence_completeness;
  return (<>
    <header className="ph"><div><div className="lbl">Assessment {shortId(a.assessment_id)}</div><h1>{name ?? a.machine_id}</h1><p>{ago(a.created_at)} · {a.source_mode} data</p></div>
      <div className="ph-r"><span className={`badge ${t}`}>{label(a.overall_status)}</span>
        <button className="ghost" onClick={() => { navigator.clipboard.writeText(a.assessment_id); toast("Assessment ID copied"); }}>Copy ID</button></div></header>
    {a.status_reasons.length > 0 && <div className="banner reveal" style={{ borderColor: toneColor[t] + "99", background: toneColor[t] + "1a" }}><i style={{ background: toneColor[t], boxShadow: `0 0 10px ${toneColor[t]}` }} /><b style={{ color: toneColor[t] }}>{label(a.overall_status)}</b><span>{a.status_reasons.join(" · ")}</span></div>}
    <div className="g12">
      <div className="c4 stack" style={{ alignSelf: "start", position: "relative", minWidth: 0 }}><div className="card reveal" style={{ gap: 12, flex: "none" }}>
        <div className="lbl">Evidence completeness</div><div style={{ fontSize: 30, fontWeight: 600 }}>{c.present}/{c.expected}</div>
        <div className="bar" style={{ margin: 0 }}><i style={{ width: `${c.ratio * 100}%`, background: "#22D3EE" }} /></div>
        <div className="sub">How many of the requested inputs produced results. This is not confidence.</div>
        <div className="gap8">{a.modalities.map((m) => <span key={m.kind} className={`badge cap ${m.availability === "available" ? "healthy" : "inspect"}`}>{m.kind}: {m.availability}</span>)}</div></div>
        <div className="card reveal" style={{ gap: 10, flex: "none" }}><div className="lbl">Quality flags</div>
          {a.quality_flags.length ? a.quality_flags.map((f) => (<div className="rec" key={f.code}><div className="row" style={{ marginBottom: 0 }}><b>{title(f.code.toLowerCase())}</b><span className={`badge ${f.severity === "critical" ? "critical" : f.severity === "warning" ? "inspect" : "unknown"}`}>{f.severity}</span></div><div className="sub">{f.explanation}</div></div>)) : <span className="sub">No quality flags.</span>}</div></div>
      <div className="c8 stackc">{a.modalities.map((m) => {
        if (m.availability !== "available" || !m.output) return (<div className="card reveal" key={m.kind}><div className="row" style={{ marginBottom: 0 }}><b className="cap">{m.kind} model</b><span className="badge inspect">{m.availability}</span></div><div className="sub">{title(m.reason_code)}{m.error_detail ? `: ${m.error_detail}` : ""}</div></div>);
        if (m.kind === "sensor") return <Sensor key={m.kind} m={m} o={m.output as SensorAssessment} />;
        if (m.kind === "acoustic") return <Acoustic key={m.kind} m={m} o={m.output as AcousticAnalysis} />;
        return <Bearing key={m.kind} m={m} o={m.output as FaultDiagnosis} />;
      })}</div>
      <div className="c12"><AiPanel key={a.assessment_id} a={a} /></div>
      <div className="card c12 reveal" style={{ overflowX: "auto" }}><details><summary>Stored evidence ({a.evidence.length})</summary>
        <table className="tbl"><thead><tr><th>Kind</th><th>Value</th><th>Unit</th><th>Source</th><th>Provenance</th></tr></thead><tbody>{a.evidence.map((e) => <tr key={e.evidence_id}><td>{title(e.kind)}</td><td className="mono">{typeof e.value === "number" ? Number(e.value.toFixed(4)) : String(e.value)}</td><td>{e.unit ?? "–"}</td><td className="cap">{e.source}</td><td>{e.source_mode}{e.illustrative ? " (illustrative)" : ""}</td></tr>)}</tbody></table></details></div>
    </div></>);
}
