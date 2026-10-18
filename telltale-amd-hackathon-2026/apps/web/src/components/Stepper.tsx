import { Check } from "lucide-react";
export default function Stepper({ steps, current }: { steps: string[]; current: number }) {
  return (<div className="stp">{steps.map((s, i) => (<div key={s} style={{ display: "contents" }}>
    {i > 0 && <div className={`ln ${current > i ? "f" : ""}`}><i /></div>}
    <div className={`s ${current === i + 1 ? "on" : ""} ${current > i + 1 ? "done" : ""}`}><span className="n">{current > i + 1 ? <Check size={15} /> : i + 1}</span>{s}</div></div>))}</div>);
}
