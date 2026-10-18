import { useEffect, useRef } from "react";
import type { AcousticAnalysis } from "../api/types";

// Drawn from the arrays the backend returns inside the stored acoustic output (audio itself is never stored).
export function WaveformChart({ wave, peak }: { wave: AcousticAnalysis["waveform"]; peak?: number }) {
  const n = wave.max.length, W = 1000, H = 110, mid = H / 2;
  const amp = Math.max(0.001, ...wave.max.map(Math.abs), ...wave.min.map(Math.abs));
  const d = wave.max.map((mx, i) => { const x = (i / Math.max(1, n - 1)) * W; return `M${x.toFixed(1)} ${(mid - (mx / amp) * (mid - 4)).toFixed(1)}V${(mid - (wave.min[i] / amp) * (mid - 4)).toFixed(1)}`; }).join("");
  const px = peak != null && wave.duration_s > 0 ? (peak / wave.duration_s) * W : null;
  return (<svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" role="img" aria-label="Waveform of the analysed recording">
    {px != null && <rect x={Math.max(0, px - W * 0.02)} width={W * 0.04} y={0} height={H} fill="#F43F5E" opacity=".16" />}
    <path d={d} stroke="#22D3EE" strokeWidth="1.4" vectorEffect="non-scaling-stroke" fill="none" />
    {px != null && <line x1={px} x2={px} y1={0} y2={H} stroke="#F43F5E" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />}</svg>);
}

const STOPS: [number, number[]][] = [[0, [11, 29, 42]], [0.4, [14, 90, 107]], [0.7, [34, 211, 238]], [0.85, [245, 158, 11]], [1, [244, 63, 94]]];
function ramp(t: number) {
  const v = Math.min(1, Math.max(0, t)); let k = 1; while (k < STOPS.length - 1 && v > STOPS[k][0]) k++;
  const [a, A] = STOPS[k - 1], [b, B] = STOPS[k]; const f = (v - a) / (b - a);
  return A.map((x, j) => Math.round(x + (B[j] - x) * f));
}

export function SpectrogramChart({ spec, peak, duration }: { spec: AcousticAnalysis["spectrogram"]; peak?: number; duration?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current; const rows = spec.values.length, cols = spec.values[0]?.length ?? 0;
    if (!cv || !rows || !cols) return;
    cv.width = cols; cv.height = rows; const ctx = cv.getContext("2d"); if (!ctx) return;
    let lo = Infinity, hi = -Infinity;
    for (const r of spec.values) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = Math.max(1e-6, hi - lo); const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const [R, G, B] = ramp((spec.values[r][c] - lo) / span); const k = ((rows - 1 - r) * cols + c) * 4;
      img.data[k] = R; img.data[k + 1] = G; img.data[k + 2] = B; img.data[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [spec]);
  const maxHz = spec.band_hz[spec.band_hz.length - 1] ?? 0;
  const left = peak != null && duration ? `${(peak / duration) * 100}%` : null;
  return (<div style={{ position: "relative" }}>
    <canvas ref={ref} className="canvas" style={{ height: 150 }} role="img" aria-label="Spectrogram of the analysed recording" />
    {left && <div style={{ position: "absolute", top: 0, bottom: 0, left, width: 2, background: "#F43F5E", boxShadow: "0 0 8px #F43F5E" }} />}
    <span className="lbl" style={{ position: "absolute", left: 8, top: 6, color: "#fff" }}>{(maxHz / 1000).toFixed(1)} kHz</span>
    <span className="lbl" style={{ position: "absolute", left: 8, bottom: 6, color: "#fff" }}>0 Hz</span></div>);
}

export function FrameErrorChart({ series, threshold, peak }: { series: AcousticAnalysis["frame_errors"]; threshold: number; peak?: number }) {
  const W = 1000, H = 90, vals = series.values, max = Math.max(threshold, ...vals) * 1.15 || 1;
  const t1 = series.times[series.times.length - 1] || 1;
  const pts = vals.map((v, i) => `${((series.times[i] / t1) * W).toFixed(1)},${(H - (v / max) * (H - 6) - 3).toFixed(1)}`).join(" ");
  const ty = H - (threshold / max) * (H - 6) - 3;
  return (<svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" role="img" aria-label="Anomaly score over time with alert threshold">
    <line x1="0" x2={W} y1={ty} y2={ty} stroke="#F43F5E" strokeDasharray="6 6" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
    <polyline points={pts} fill="none" stroke="#F59E0B" strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    {peak != null && <line x1={(peak / t1) * W} x2={(peak / t1) * W} y1="0" y2={H} stroke="#F43F5E" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />}</svg>);
}
