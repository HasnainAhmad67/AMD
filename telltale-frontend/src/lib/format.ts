import type { Health, OverallStatus, RiskLevel, SoundStatus } from "../api/types";

export type Tone = "healthy" | "inspect" | "critical" | "unknown";
type Anyhealth = Health | OverallStatus | RiskLevel | SoundStatus | null | undefined;

export function tone(h: Anyhealth): Tone {
  switch (h) {
    case "ok": case "normal": case "low": return "healthy";
    case "watch": case "medium": return "inspect";
    case "critical": case "high": case "anomalous": return "critical";
    default: return "unknown";
  }
}
export const toneColor: Record<Tone, string> = { healthy: "#34D399", inspect: "#F59E0B", critical: "#F43F5E", unknown: "#8B9AAB" };

const LABELS: Record<string, string> = {
  ok: "Healthy", normal: "Normal", watch: "Watch", critical: "Critical", unknown: "Unknown",
  low: "Low risk", medium: "Medium risk", high: "High risk", anomalous: "Anomalous",
};
export const label = (h: Anyhealth) => (h ? LABELS[h] ?? String(h) : "Unknown");

export const pct = (p: number | null | undefined, digits = 0) => (p == null ? "–" : `${(p * 100).toFixed(digits)}%`);
export const shortId = (id: string) => id.slice(0, 8);
export const title = (s: string) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

export function ago(iso?: string | null) {
  if (!iso) return "–";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}
