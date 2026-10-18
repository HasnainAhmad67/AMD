import { ApiError, hasApiToken } from "../api/client";

function hint(e: unknown): string | null {
  if (!(e instanceof ApiError)) return null;
  if (e.code === "network") return "Start the backend, then check VITE_API_PROXY in apps/web/.env and restart npm run dev.";
  if (e.status === 401) return hasApiToken
    ? "The backend rejected the token. With TELLTALE_DEMO_MODE=1 it must be demo-token-123; otherwise it must match TELLTALE_API_TOKENS."
    : "No token is configured. Add VITE_API_TOKEN to apps/web/.env (demo-token-123 in demo mode) and restart npm run dev.";
  if (e.status === 503) return "A backend dependency is not ready (for example a model that is still loading or training). Try again in a minute.";
  return null;
}

export function ErrorCard({ error, onRetry, title = "Something went wrong" }: { error: unknown; onRetry?: () => void; title?: string }) {
  const e = error instanceof ApiError ? error : null; const h = hint(error);
  return (<div className="card err" role="alert" style={{ gap: 8 }}>
    <b style={{ color: "#FB7185" }}>{title}</b>
    <div>{error instanceof Error ? error.message : "Unknown error"}</div>
    {h && <div className="sub">{h}</div>}
    {e && (e.code || e.requestId) && <div className="lbl" style={{ textTransform: "none" }}>{e.status ? `HTTP ${e.status}` : ""} {e.code ?? ""} {e.requestId ? `· request ${e.requestId.slice(0, 8)}` : ""}</div>}
    {onRetry && <div><button className="ghost" onClick={onRetry}>Try again</button></div>}</div>);
}

export const Skeletons = ({ heights }: { heights: number[] }) => <>{heights.map((h, i) => <div key={i} className="skel" style={{ height: h }} />)}</>;
