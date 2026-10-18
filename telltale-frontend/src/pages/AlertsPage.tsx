import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { useAlerts } from "../api/hooks";
import { ErrorCard, Skeletons } from "../components/QueryState";
import Tilt from "../components/Tilt";
import { ago } from "../lib/format";
import { toast } from "../lib/toast";

type F = "all" | "open" | "acknowledged" | "resolved";
export default function AlertsPage() {
  const q = useAlerts(); const qc = useQueryClient(); const [f, setF] = useState<F>("open");
  const refresh = () => { for (const k of ["alerts", "overview", "machines"]) qc.invalidateQueries({ queryKey: [k] }); };
  const ack = useMutation({ mutationFn: (id: string) => api.ackAlert(id), onSuccess: () => { toast("Alert acknowledged"); refresh(); }, onError: (e) => toast(e instanceof Error ? e.message : "Failed") });
  const res = useMutation({ mutationFn: (id: string) => api.resolveAlert(id), onSuccess: () => { toast("Alert resolved"); refresh(); }, onError: (e) => toast(e instanceof Error ? e.message : "Failed") });
  if (q.isError) return <ErrorCard error={q.error} onRetry={() => q.refetch()} title="Could not load alerts" />;
  if (!q.data) return <Skeletons heights={[96, 360]} />;
  const all = q.data; const n = (s: F) => (s === "all" ? all.length : all.filter((a) => a.status === s).length);
  const shown = all.filter((a) => f === "all" || a.status === f);
  const kp = [["Open", n("open"), "#22D3EE"], ["Critical & open", all.filter((a) => a.status === "open" && a.severity === "critical").length, "#F43F5E"], ["Acknowledged", n("acknowledged"), "#F59E0B"], ["Resolved", n("resolved"), "#34D399"]] as const;
  return (<>
    <header className="ph"><div><h1>Alerts</h1><p>Raised automatically from sensor and sound signals</p></div></header>
    <div className="g12">{kp.map(([l, v, c], i) => <Tilt key={l} className="card c3 k reveal" style={{ animationDelay: `${i * 70}ms` }}><div className="lbl">{l}</div><div style={{ fontSize: 34, fontWeight: 600, marginTop: 6, color: c }}>{v}</div></Tilt>)}</div>
    <div className="chips">{(["open", "acknowledged", "resolved", "all"] as F[]).map((k) => <button key={k} className={`chip cap ${f === k ? "on" : ""}`} onClick={() => setF(k)}>{k} {n(k)}</button>)}</div>
    <div className="card reveal" style={{ overflowX: "auto" }}><table className="tbl"><thead><tr><th>Severity</th><th>Alert</th><th>Machine</th><th>Status</th><th>When</th><th></th></tr></thead><tbody>
      {shown.map((a, i) => (<tr key={a.id} style={{ animationDelay: `${i * 40}ms` }}><td><span className={`badge ${a.severity === "critical" ? "critical" : "inspect"}`}>{a.severity}</span></td>
        <td><div style={{ fontWeight: 500 }}>{a.title}</div><div className="sub">{a.detail}</div></td><td><Link to={`/machines/${a.machine_id}`} style={{ color: "var(--ac)" }}>{a.machine_name}</Link><div className="lbl cap">{a.source}</div></td>
        <td><span className={`badge cap ${a.status === "open" ? "inspect" : a.status === "resolved" ? "healthy" : "unknown"}`}>{a.status}</span></td><td className="sub">{ago(a.created_at)}</td>
        <td style={{ textAlign: "right" }}><div className="gap8" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
          {a.status === "open" && <button className="ghost" style={{ height: 34 }} disabled={ack.isPending} onClick={() => ack.mutate(a.id)}>Acknowledge</button>}
          {a.status !== "resolved" && <button className="ghost" style={{ height: 34 }} disabled={res.isPending} onClick={() => res.mutate(a.id)}>Resolve</button>}</div></td></tr>))}
      {!shown.length && <tr><td colSpan={6} className="lbl">No alerts in this view</td></tr>}</tbody></table></div></>);
}
