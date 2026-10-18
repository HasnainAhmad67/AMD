import { useState } from "react";
import { hasApiToken } from "../api/client";
import { useHealth } from "../api/hooks";
import { ErrorCard, Skeletons } from "../components/QueryState";

export default function SettingsPage() {
  const h = useHealth(); const [calm, setCalm] = useState(localStorage.getItem("calm") === "1");
  const flip = () => { const v = !calm; setCalm(v); localStorage.setItem("calm", v ? "1" : "0"); document.documentElement.classList.toggle("calm", v); };
  const yn = (b: boolean) => <span className={`badge ${b ? "healthy" : "inspect"}`}>{b ? "Online" : "Not loaded"}</span>;
  return (<>
    <header className="ph"><div><h1>Settings</h1><p>Display and backend status</p></div></header>
    <div className="g12">
      <div className="card c6 reveal"><div className="lbl" style={{ marginBottom: 8 }}>Display</div>
        <div className="setrow" style={{ borderTop: 0 }}><div><div>Calm mode</div><div className="sub">Turns off animations and 3D effects. Use it on slow laptops or projectors.</div></div><button className={`sw ${calm ? "on" : ""}`} onClick={flip} role="switch" aria-checked={calm} aria-label="Calm mode" /></div>
        <div className="setrow"><div><div>API token</div><div className="sub">Needed for assessments and AI (set VITE_API_TOKEN in apps/web/.env)</div></div><span className={`badge ${hasApiToken ? "healthy" : "inspect"}`}>{hasApiToken ? "Configured" : "Missing"}</span></div></div>
      <div className="card c6 reveal"><div className="lbl" style={{ marginBottom: 8 }}>Backend</div>
        {h.isError ? <ErrorCard error={h.error} onRetry={() => h.refetch()} title="Backend unreachable" /> : !h.data ? <Skeletons heights={[160]} /> : (<>
          <div className="setrow" style={{ borderTop: 0 }}><span>Status</span><span className="badge healthy">{h.data.status} · v{h.data.version}</span></div>
          <div className="setrow"><span>Sensor model</span>{yn(h.data.models.sensor)}</div><div className="setrow"><span>Sound models</span>{yn(h.data.models.acoustic)}</div><div className="setrow"><span>Bearing model</span>{yn(h.data.models.bearing)}</div>
          <div className="setrow"><span>Sound types</span><span className="mono sub">{h.data.acoustic_machine_types.join(", ") || "none"}</span></div>
          <div className="setrow"><span>Fleet simulator</span><span className={`badge ${h.data.simulator ? "healthy" : "unknown"}`}>{h.data.simulator ? "Running" : "Off"}</span></div>
          <div className="setrow"><span>First-start training</span><span className="sub">{h.data.sensor_bootstrap.status}{h.data.sensor_bootstrap.message ? ` · ${h.data.sensor_bootstrap.message}` : ""}</span></div></>)}</div>
    </div></>);
}
