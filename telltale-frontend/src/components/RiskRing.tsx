import { useEffect, useState } from "react";
import { useCountUp } from "../lib/hooks";
import { toneColor, type Tone } from "../lib/format";
// Shows the model's failure probability (0..1). It is a probability, not a "health score": none is invented.
export default function RiskRing({ p, tone, size = 72 }: { p: number | null; tone: Tone; size?: number }) {
  const r = size * 0.4, c = 2 * Math.PI * r, w = size * 0.1, m = size / 2, col = toneColor[tone];
  const target = p == null ? 0 : Math.round(p * 100);
  const [arc, setArc] = useState(0); const shown = useCountUp(target);
  useEffect(() => { const id = requestAnimationFrame(() => setArc(target)); return () => cancelAnimationFrame(id); }, [target]);
  return (<svg className="ring-glow" style={{ ["--g" as string]: col + "99" }} width={size} height={size} role="img" aria-label={p == null ? "No risk score yet" : `Failure risk ${target} percent`}>
    <circle cx={m} cy={m} r={r} fill="none" stroke="#1F2A37" strokeWidth={w} />
    <circle cx={m} cy={m} r={r} fill="none" stroke={col} strokeWidth={w} strokeLinecap="round" strokeDasharray={`${(arc / 100) * c} ${c}`} transform={`rotate(-90 ${m} ${m})`} style={{ transition: "stroke-dasharray 1.2s cubic-bezier(.2,.8,.2,1)" }} />
    <text x={m} y={m + size * 0.07} textAnchor="middle" fill="#E6EDF3" fontSize={size * 0.26} fontWeight={600}>{p == null ? "–" : `${shown}%`}</text></svg>);
}
