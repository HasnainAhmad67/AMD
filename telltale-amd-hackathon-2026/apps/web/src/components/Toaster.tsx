import { useEffect, useState } from "react";
export default function Toaster() {
  const [t, setT] = useState<{ id: number; m: string }[]>([]);
  useEffect(() => {
    const h = (e: Event) => { const id = Date.now() + Math.random(); setT(x => [...x, { id, m: (e as CustomEvent<string>).detail }]); setTimeout(() => setT(x => x.filter(i => i.id !== id)), 2800); };
    window.addEventListener("toast", h); return () => window.removeEventListener("toast", h);
  }, []);
  return <div className="toasts" role="status" aria-live="polite">{t.map(i => <div className="toast" key={i.id}>{i.m}</div>)}</div>;
}
