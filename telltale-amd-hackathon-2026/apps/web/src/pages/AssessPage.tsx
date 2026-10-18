import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { useAssessments, useMachines, useSamples } from "../api/hooks";
import type { SampleClip, SensorField, SensorReading } from "../api/types";
import { ErrorCard } from "../components/QueryState";
import Stepper from "../components/Stepper";
import { ago, label, shortId, tone } from "../lib/format";
import { toast } from "../lib/toast";

const FIELDS: { k: SensorField; label: string; min: number; max: number; step: number }[] = [
  { k: "air_temperature_k", label: "Air temperature (K)", min: 250, max: 350, step: 0.1 },
  { k: "process_temperature_k", label: "Process temperature (K)", min: 250, max: 400, step: 0.1 },
  { k: "rotational_speed_rpm", label: "Rotational speed (rpm)", min: 1, max: 5000, step: 1 },
  { k: "torque_nm", label: "Torque (Nm)", min: 0, max: 200, step: 0.1 },
  { k: "tool_wear_min", label: "Tool wear (min)", min: 0, max: 500, step: 1 },
];

function FileSlot({ title, hint, file, setFile, clips, disabled, onError }: { title: string; hint: string; file: File | null; setFile: (f: File | null) => void; clips: SampleClip[]; disabled?: boolean; onError: (e: unknown) => void }) {
  return (<div className="stackc" style={{ gap: 10, opacity: disabled ? 0.5 : 1 }}>
    <div className="lbl">{title}</div>
    <label className={`drop ${file ? "has" : ""}`} style={{ padding: 22 }}>
      <div style={{ color: "var(--tx)", fontWeight: 600 }}>{file ? file.name : "Drop a file or browse"}</div><div className="sub">{hint}</div>
      <input type="file" hidden disabled={disabled} onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></label>
    <div className="gap8" style={{ flexWrap: "nowrap" }}>
      <select className="in" disabled={disabled || !clips.length} value="" aria-label={`${title}: use a sample clip`} onChange={(e) => { if (e.target.value) api.sampleFile(e.target.value).then(setFile).catch(onError); }}>
        <option value="">{clips.length ? "Or use a sample clip…" : "No sample clips available"}</option>{clips.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}</select>
      {file && <button className="ghost" onClick={() => setFile(null)}>Remove</button>}</div></div>);
}

export default function AssessPage() {
  const machines = useMachines(); const samples = useSamples(); const recent = useAssessments(8);
  const [params] = useSearchParams(); const nav = useNavigate(); const qc = useQueryClient();
  const ms = machines.data ?? []; const [machineId, setMachineId] = useState(params.get("machine") ?? "");
  const machine = ms.find((m) => m.id === machineId) ?? ms[0];
  const [vals, setVals] = useState<Record<string, string>>({}); const [quality, setQuality] = useState("");
  const [useSensor, setUseSensor] = useState(true); const [source, setSource] = useState<"simulated" | "dataset_demo">("simulated");
  const [acoustic, setAcoustic] = useState<File | null>(null); const [bearing, setBearing] = useState<File | null>(null);
  const [stamp, setStamp] = useState(true); const [busy, setBusy] = useState(false); const [error, setError] = useState<unknown>(null);
  const clips = samples.data ?? []; const profile = machine?.acoustic_profile ?? null;
  const val = (k: SensorField) => vals[k] ?? (machine?.latest_reading ? String(machine.latest_reading[k]) : "");
  const soundOnlyBlocked = Boolean(acoustic) && !useSensor && !bearing && profile !== "fan";
  const nothing = !useSensor && !acoustic && !bearing;

  async function submit() {
    if (!machine) return; setError(null);
    let sensor: { reading: SensorReading; sourceMode: "simulated" | "dataset_demo" } | undefined;
    if (useSensor) {
      const nums: Record<string, number> = {};
      for (const f of FIELDS) {
        const n = Number(val(f.k));
        if (!Number.isFinite(n) || val(f.k).trim() === "" || n < f.min || n > f.max) { setError(new Error(`${f.label} must be a number between ${f.min} and ${f.max}.`)); return; }
        nums[f.k] = n;
      }
      sensor = { reading: { product_quality: (quality || machine.latest_reading?.product_quality || machine.product_quality) as SensorReading["product_quality"], ...nums } as SensorReading, sourceMode: source };
    }
    setBusy(true);
    try {
      const a = await api.createAssessment({ machineId: machine.id, sensor, acoustic: acoustic ?? undefined, bearing: bearing ?? undefined, observedAt: stamp ? new Date().toISOString() : undefined });
      qc.invalidateQueries({ queryKey: ["assessments"] }); toast("Assessment created"); nav(`/assessments/${a.assessment_id}`);
    } catch (e) { setError(e); setBusy(false); }
  }

  return (<>
    <header className="ph"><div><h1>New assessment</h1><p>Combine sensor readings and recordings into one stored, explainable assessment</p></div><Stepper steps={["Inputs", "Assess", "Review"]} current={busy ? 2 : 1} /></header>
    <div className="g12">
      <div className="c8 stack">
        <div className="card reveal" style={{ gap: 14 }}>
          <div className="g2"><label className="fld"><span>Machine</span><select className="in" value={machine?.id ?? ""} onChange={(e) => { setMachineId(e.target.value); setVals({}); setQuality(""); setAcoustic(null); }} aria-label="Machine">{ms.map((m) => <option key={m.id} value={m.id}>{m.name} · {m.line}</option>)}</select></label>
            <label className="fld"><span>Data source (for the sensor readings)</span><select className="in" value={source} onChange={(e) => setSource(e.target.value as "simulated" | "dataset_demo")}><option value="simulated">Simulated</option><option value="dataset_demo">Dataset demo</option></select></label></div>
          {machines.isError && <ErrorCard error={machines.error} onRetry={() => machines.refetch()} title="Could not load machines" />}</div>
        <div className="card reveal" style={{ gap: 14 }}>
          <label className="gap8" style={{ cursor: "pointer" }}><input type="checkbox" checked={useSensor} onChange={(e) => setUseSensor(e.target.checked)} /><b>Sensor readings</b><span className="sub">prefilled from the machine's latest reading</span></label>
          {useSensor && <div className="g2">{FIELDS.map((f) => <label className="fld" key={f.k}><span>{f.label}</span><input className="in mono" type="number" inputMode="decimal" min={f.min} max={f.max} step={f.step} value={val(f.k)} onChange={(e) => setVals((v) => ({ ...v, [f.k]: e.target.value }))} /></label>)}
            <label className="fld"><span>Product quality</span><select className="in" value={quality || machine?.latest_reading?.product_quality || machine?.product_quality || "M"} onChange={(e) => setQuality(e.target.value)}><option value="L">L</option><option value="M">M</option><option value="H">H</option></select></label></div>}</div>
        <div className="g2">
          <div className="card reveal"><FileSlot title="Sound recording" hint={profile ? `WAV, FLAC or OGG · ${profile} sound model` : "This machine has no sound profile"} file={acoustic} setFile={setAcoustic} clips={clips.filter((c) => c.machine_type === profile)} disabled={!profile} onError={setError} /></div>
          <div className="card reveal"><FileSlot title="Bearing recording" hint="WAV, FLAC, OGG or MATLAB .mat" file={bearing} setFile={setBearing} clips={clips.filter((c) => c.machine_type === "bearing")} onError={setError} /></div></div>
        <div className="card reveal" style={{ gap: 12 }}>
          <label className="gap8" style={{ cursor: "pointer" }}><input type="checkbox" checked={stamp} onChange={(e) => setStamp(e.target.checked)} /><span>Record the observation time as now</span></label>
          <div className="sub">Without a time the backend flags the assessment as having a missing timestamp.</div>
          {soundOnlyBlocked && <div className="banner" style={{ borderColor: "#F59E0B99", background: "#F59E0B1a" }}><i style={{ background: "#F59E0B" }} /><span>Sound-only assessments currently support fan machines (the CNC Mills). Add sensor readings or a bearing recording, or pick a CNC Mill.</span></div>}
          {nothing && <div className="sub">Add at least one input.</div>}
          {error != null && <ErrorCard error={error} title="The assessment could not be created" />}
          <div><button className="btn" disabled={busy || nothing || soundOnlyBlocked || !machine} onClick={submit}>{busy ? "Assessing…" : "Create assessment"}</button></div></div>
      </div>
      <div className="card c4 reveal" style={{ gap: 12, alignSelf: "start" }}><div className="lbl">Recent assessments</div>
        {recent.isError ? <ErrorCard error={recent.error} onRetry={() => recent.refetch()} title="Could not load assessments" /> : !recent.data ? <div className="skel" style={{ height: 120 }} />
          : recent.data.items.length ? recent.data.items.map((a) => (<Link key={a.assessment_id} to={`/assessments/${a.assessment_id}`} className="qa" style={{ justifyContent: "space-between" }}>
              <span><b>{ms.find((m) => m.id === a.machine_id)?.name ?? a.machine_id}</b><div className="lbl" style={{ marginTop: 2, textTransform: "none" }}>{shortId(a.assessment_id)} · {a.modalities.map((m) => m.kind).join(" + ")} · {ago(a.created_at)}</div></span>
              <span className={`badge ${tone(a.overall_status)}`}>{label(a.overall_status)}</span></Link>)) : <span className="sub">No assessments yet. Create the first one.</span>}</div>
    </div></>);
}
