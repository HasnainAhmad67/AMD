import type { Recommendation } from "../api/types";
const TONE = { high: "critical", medium: "inspect", low: "healthy" } as const;
export default function Recs({ items }: { items: Recommendation[] }) {
  if (!items.length) return null;
  return (<div className="stackc">{items.map((r) => (<div className="rec" key={r.title}>
    <div className="row" style={{ marginBottom: 0 }}><b>{r.title}</b><span className={`badge ${TONE[r.priority]}`}>{r.priority}</span></div>
    <div className="sub">{r.reason}</div>
    {r.steps.length > 0 && <ol className="ul" style={{ listStyle: "decimal" }}>{r.steps.map((s) => <li key={s}>{s}</li>)}</ol>}
    <div className="lbl" style={{ textTransform: "none" }}>Source: {r.source}</div></div>))}</div>);
}
