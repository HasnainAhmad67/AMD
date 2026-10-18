import { useEffect, useState } from "react";
// Counts from 0 to target with an ease-out curve (instant when the user prefers reduced motion).
export function useCountUp(target: number, ms = 900) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setV(target); return; }
    let raf = 0; const t0 = performance.now();
    const tick = (t: number) => { const p = Math.min(1, (t - t0) / ms); setV(Math.round(target * (1 - Math.pow(1 - p, 3)))); if (p < 1) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

// Types text out character by character (instant when reduced motion is on).
export function useTypewriter(text: string, cps = 140) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || document.documentElement.classList.contains("calm")) { setN(text.length); return; }
    setN(0); let raf = 0; const t0 = performance.now();
    const tick = (t: number) => { const k = Math.min(text.length, Math.floor(((t - t0) / 1000) * cps)); setN(k); if (k < text.length) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [text, cps]);
  return text.slice(0, n);
}
