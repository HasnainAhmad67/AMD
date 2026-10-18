import { useRef, ReactNode, CSSProperties } from "react";
import { Link } from "react-router-dom";
type P = { to?: string; className?: string; style?: CSSProperties; children: ReactNode };
// Card that tilts in 3D toward the mouse and shows a moving glare.
export default function Tilt({ to, className = "", style, children }: P) {
  const ref = useRef<HTMLElement | null>(null);
  const move = (e: React.MouseEvent) => {
    const el = ref.current; if (!el) return; const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    el.style.setProperty("--ry", `${(x - 0.5) * 10}deg`); el.style.setProperty("--rx", `${(0.5 - y) * 8}deg`);
    el.style.setProperty("--gx", `${x * 100}%`); el.style.setProperty("--gy", `${y * 100}%`);
  };
  const leave = () => { const el = ref.current; if (!el) return; el.style.setProperty("--rx", "0deg"); el.style.setProperty("--ry", "0deg"); };
  const props = { className: `${className} tilt`, style, onMouseMove: move, onMouseLeave: leave };
  return to ? <Link to={to} ref={ref as any} {...props}>{children}</Link> : <div ref={ref as any} {...props}>{children}</div>;
}
