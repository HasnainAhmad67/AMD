import { useState } from "react";
import { api } from "../api/client";
import type { AiAnyResponse, AiAskResponse, AiMeta, AiOutputRecord, Assessment } from "../api/types";
import { ago, title } from "../lib/format";
import { toast } from "../lib/toast";
import { ErrorCard } from "./QueryState";
import Typed from "./Typed";

type Tab = "explain" | "analyze" | "evidence" | "ask" | "report";
const TABS: { id: Tab; label: string; button: string; blurb: string }[] = [
  { id: "explain", label: "Explanation", button: "Generate explanation", blurb: "A plain-language explanation of this assessment, grounded in the stored evidence." },
  { id: "analyze", label: "Model comparison", button: "Compare the models", blurb: "Compares the sensor, sound and bearing results: where they agree, send mixed signals or are missing." },
  { id: "evidence", label: "Evidence & uncertainty", button: "Review the evidence", blurb: "What is missing, stale or limited, and what these models cannot tell us." },
  { id: "ask", label: "Ask Telltale", button: "Ask", blurb: "Ask about this assessment or this machine. Answers use stored data only, and some questions are declined." },
  { id: "report", label: "Maintenance report", button: "Generate report", blurb: "A structured report over this machine's recent assessments." },
];
const QUESTIONS = ["What is the overall status?", "Which modalities were used?", "Are there quality flags or uncertainty?", "What did the sensor show?", "What did the sound model show?", "What about the bearing?", "Where does this data come from?", "How has it changed recently?"];

type Slot = { state: "loading" | "done" | "error"; meta?: AiAnyResponse; record?: AiOutputRecord | null; error?: unknown };
type AskItem = { q: string; loading: boolean; meta?: AiAskResponse; record?: AiOutputRecord | null; error?: unknown };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Out = Record<string, any>;

const Bullets = ({ head, items }: { head: string; items?: string[] }) =>
  items && items.length ? (<div><div className="lbl" style={{ marginBottom: 6 }}>{head}</div><ul className="ul">{items.map((x, i) => <li key={i}>{x}</li>)}</ul></div>) : null;

function Meta({ m }: { m: AiMeta }) {
  const v = m.validation_status;
  return (<div className="stackc" style={{ gap: 8 }}>
    <div className="gap8">
      <span className={`badge ${m.provider === "amd" ? "healthy" : "unknown"}`}>{m.provider === "amd" ? "AMD Token Factory" : "Template provider"}</span>
      {m.model && <span className="badge unknown mono">{m.model}</span>}
      <span className={`badge ${v === "validated" ? "healthy" : v === "fallback" ? "inspect" : "critical"}`}>{v}</span>
      <span className="badge unknown mono">{m.prompt_version}</span><span className="sub">{ago(m.created_at)}</span></div>
    {v === "fallback" && <div className="sub">The AI draft could not be verified against the stored evidence (or the provider failed), so Telltale shows the deterministic fallback instead.</div>}
    {v === "refused" && <div className="sub">There is not enough stored evidence to answer safely, so the request was declined.</div>}</div>);
}

function Loading({ text }: { text: string }) {
  return (<div className="stackc" role="status" aria-live="polite"><div className="eq" style={{ height: 30 }} aria-hidden>{Array.from({ length: 20 }, (_, i) => <i key={i} style={{ animationDelay: `${(i * 83) % 900}ms` }} />)}</div>
    <div className="sub">{text}</div><div className="skel" style={{ height: 14 }} /><div className="skel" style={{ height: 14, width: "80%" }} /><div className="skel" style={{ height: 14, width: "60%" }} /></div>);
}


function Explain({
  meta,
  o,
}: {
  meta: Extract<AiAnyResponse, { explanation_text: string }>;
  o: Out;
}) {
  return (
    <div className="stackc">
      <div className="rec">
        <div className="lbl" style={{ marginBottom: 8 }}>
          Assessment explanation
        </div>

        <div
          className="prose"
          style={{
            whiteSpace: "pre-wrap",
            overflowWrap: "anywhere",
            lineHeight: 1.8,
          }}
        >
          <Typed text={meta.explanation_text} />
        </div>
      </div>

      <Bullets
        head="Status reasons"
        items={o.status_reasons}
      />

      <Bullets
        head="Uncertainty and data quality"
        items={o.uncertainty_notes}
      />

      {Array.isArray(o.per_modality) &&
        o.per_modality.length > 0 && (
          <div className="stackc">
            <div className="lbl">Evidence by modality</div>

            <div className="g2">
              {o.per_modality.map((m: Out) => (
                <div className="rec" key={m.kind}>
                  <div
                    className="row"
                    style={{ marginBottom: 8 }}
                  >
                    <b className="cap">{m.kind}</b>

                    <span
                      className={`badge ${
                        m.availability === "available"
                          ? "healthy"
                          : "unknown"
                      }`}
                    >
                      {m.availability}
                    </span>
                  </div>

                  {Array.isArray(m.facts) &&
                  m.facts.length > 0 ? (
                    <ul className="ul">
                      {m.facts.map((f: string, i: number) => (
                        <li key={`${m.kind}-${i}`}>{f}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="sub">
                      No additional modality details are stored.
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

      {Array.isArray(o.limitations) &&
        o.limitations.length > 0 && (
          <details>
            <summary>
              Model limitations ({o.limitations.length})
            </summary>

            <ul className="ul">
              {o.limitations.map((x: string, i: number) => (
                <li key={i}>{x}</li>
              ))}
            </ul>
          </details>
        )}
    </div>
  );
}


const AGREE: Record<string, [string, string]> = { consistent_indications: ["healthy", "Indications consistent"], mixed_indications: ["inspect", "Indications mixed"], indeterminate: ["unknown", "Indeterminate"] };
function Analyze({ meta, o }: { meta: Extract<AiAnyResponse, { analysis_text: string }>; o: Out }) {
  const [t, l] = AGREE[o.agreement as string] ?? ["unknown", "No comparison available"];
  return (<div className="stackc">
    {o.agreement && <div className="gap8"><span className={`badge ${t}`}>{l}</span><span className="sub">{o.agreement_note}</span></div>}
    {Array.isArray(o.missing_modalities) && o.missing_modalities.length > 0 && <div className="gap8"><span className="lbl">Not available</span>{o.missing_modalities.map((k: string) => <span key={k} className="badge unknown cap">{k}</span>)}</div>}
    <p className="prose"><Typed text={meta.analysis_text} /></p>
    {Array.isArray(o.modality_summaries) && o.modality_summaries.length > 0 && <div className="g2">{o.modality_summaries.map((m: Out) => (<div className="rec" key={m.kind}>
      <div className="row" style={{ marginBottom: 0 }}><b className="cap">{m.kind}</b><span className={`badge ${m.availability === "available" ? "healthy" : "unknown"}`}>{m.availability}</span></div><div className="sub">{m.indication}</div></div>))}</div>}
    {Array.isArray(o.relations) && o.relations.length > 0 && <div><div className="lbl" style={{ marginBottom: 6 }}>Pairwise comparison</div><ul className="ul">{o.relations.map((r: Out, i: number) => <li key={i}><b className="cap">{(r.kinds ?? []).join(" + ")}</b>: {r.relation}</li>)}</ul></div>}
    <Bullets head="Conservative interpretation" items={o.conservative_interpretation} /><Bullets head="Uncertainty" items={o.uncertainty_indicators} /><Bullets head="What this does not claim" items={o.non_claims} /></div>);
}

function Evidence({ meta, o }: { meta: Extract<AiAnyResponse, { review_text: string }>; o: Out }) {
  return (<div className="stackc">
    {Array.isArray(o.missing_modalities) && o.missing_modalities.length > 0 && <div className="gap8"><span className="lbl">Missing</span>{o.missing_modalities.map((k: string) => <span key={k} className="badge inspect cap">{k}</span>)}</div>}
    {Array.isArray(o.quality_flags) && o.quality_flags.length > 0 && <div className="gap8"><span className="lbl">Quality flags</span>{o.quality_flags.map((f: Out) => <span key={f.code} className={`badge ${f.severity === "critical" ? "critical" : f.severity === "warning" ? "inspect" : "unknown"}`}>{title(String(f.code).toLowerCase())}</span>)}</div>}
    <p className="prose"><Typed text={meta.review_text} /></p>
    <Bullets head="Uncertainty indicators" items={o.uncertainty_indicators} />
    {o.provenance_summary && <div><div className="lbl" style={{ marginBottom: 6 }}>Where the data comes from</div><div className="sub">{o.provenance_summary}</div></div>}
    <Bullets head="What this does not claim" items={o.non_claims} />
    {Array.isArray(o.modality_reviews) && o.modality_reviews.length > 0 && <details><summary>Per-modality details</summary><div className="g2">{o.modality_reviews.map((m: Out) => (<div className="rec" key={m.kind}><b className="cap">{m.kind}</b>
      <div className="sub">{m.availability} · {m.reason_code} · {m.source_mode}</div><div className="sub">{[m.model_name, m.model_version, m.training_dataset].filter(Boolean).join(" · ") || "Model identity not recorded"}</div></div>))}</div></details>}</div>);
}

function Report({ meta, o }: { meta: Extract<AiAnyResponse, { report_text: string }>; o: Out }) {
  const text = `${meta.report_text}`;
  return (<div className="stackc">
    <div className="gap8"><span className="badge unknown">{meta.assessment_count} assessment{meta.assessment_count === 1 ? "" : "s"}</span>
      {meta.window_start && <span className="sub">{new Date(meta.window_start).toLocaleDateString()} – {meta.window_end ? new Date(meta.window_end).toLocaleDateString() : "now"}</span>}
      {meta.history_truncated && <span className="badge inspect">History truncated</span>}</div>
    {Array.isArray(o.status_distribution) && o.status_distribution.length > 0 && <div className="gap8"><span className="lbl">Status history</span>{o.status_distribution.map((s: Out) => <span key={s.status} className={`badge ${s.status === "critical" ? "critical" : s.status === "watch" ? "inspect" : s.status === "normal" ? "healthy" : "unknown"}`}>{s.status} × {s.count}</span>)}</div>}
    <p className="prose" style={{ whiteSpace: "pre-wrap" }}>{text}</p>
    <Bullets head="Observations" items={o.conservative_observations} /><Bullets head="Missing-data patterns" items={o.missing_data_patterns} />
    {Array.isArray(o.recurring_flags) && o.recurring_flags.length > 0 && <div className="gap8"><span className="lbl">Recurring flags</span>{o.recurring_flags.map((f: Out) => <span key={f.code} className="badge inspect">{title(String(f.code).toLowerCase())} × {f.count}</span>)}</div>}
    {Array.isArray(o.review_suggestions) && o.review_suggestions.length > 0 && <div><div className="lbl" style={{ marginBottom: 6 }}>Review suggestions (non-binding)</div><div className="stackc">{o.review_suggestions.map((s: Out) => <div className="rec" key={s.title}><b>{s.title}</b><div className="sub">{s.detail}</div><div className="lbl" style={{ textTransform: "none" }}>Basis: {s.basis}</div></div>)}</div></div>}
    <Bullets head="What this does not claim" items={o.non_claims} />
    <div className="gap8"><button className="ghost" onClick={() => { navigator.clipboard.writeText(text); toast("Report copied to clipboard"); }}>Copy report</button><button className="ghost" onClick={() => window.print()}>Print</button></div></div>);
}

export default function AiPanel({ a }: { a: Assessment }) {
  const [tab, setTab] = useState<Tab>("explain");
  const [slots, setSlots] = useState<Partial<Record<Tab, Slot>>>({});
  const [lookback, setLookback] = useState(30);
  const [scope, setScope] = useState<"assessment" | "machine">("assessment");
  const [q, setQ] = useState(""); const [asks, setAsks] = useState<AskItem[]>([]);
  const cur = TABS.find((t) => t.id === tab) as (typeof TABS)[number]; const slot = slots[tab];
  const put = (t: Tab, s: Slot) => setSlots((x) => ({ ...x, [t]: s }));
  const fetchRecord = (id: string) => api.aiOutput(id).catch(() => null);

  async function run(t: Exclude<Tab, "ask">) {
    put(t, { state: "loading" });
    try {
      const meta = t === "explain" ? await api.aiExplain(a.assessment_id) : t === "analyze" ? await api.aiAnalyze(a.assessment_id)
        : t === "evidence" ? await api.aiEvidenceReview(a.assessment_id) : await api.aiReport(a.machine_id, lookback);
      put(t, { state: "done", meta, record: await fetchRecord(meta.ai_output_id) });
    } catch (e) { put(t, { state: "error", error: e }); }
  }
  async function ask(question: string) {
    const text = question.trim(); if (!text) return; setQ("");
    setAsks((x) => [{ q: text, loading: true }, ...x]);
    const patch = (p: Partial<AskItem>) => setAsks((x) => x.map((i) => (i.q === text && i.loading ? { ...i, ...p } : i)));
    try {
      const meta = await api.aiAsk(text, scope === "assessment" ? { assessmentId: a.assessment_id } : { machineId: a.machine_id, lookbackDays: lookback });
      patch({ loading: false, meta, record: await fetchRecord(meta.ai_output_id) });
    } catch (e) { patch({ loading: false, error: e }); }
  }
  const out = (slot?.record?.output ?? {}) as Out;
  return (<div className="card reveal" style={{ gap: 16 }}>
    <div className="lbl">AI analysis</div>
    <div className="chips" role="tablist">{TABS.map((t) => <button key={t.id} role="tab" aria-selected={tab === t.id} className={`chip ${tab === t.id ? "on" : ""}`} onClick={() => setTab(t.id)}>{t.label}</button>)}</div>
    <div key={tab} className="fade stackc">
      <div className="sub">{cur.blurb}</div>
      {tab === "ask" ? (<>
        <div className="gap8"><select className="in" style={{ width: "auto" }} value={scope} onChange={(e) => setScope(e.target.value as "assessment" | "machine")} aria-label="Question scope"><option value="assessment">This assessment</option><option value="machine">This machine (recent history)</option></select>
          {scope === "machine" && <select className="in" style={{ width: "auto" }} value={lookback} onChange={(e) => setLookback(Number(e.target.value))} aria-label="Look-back window"><option value={7}>Last 7 days</option><option value={30}>Last 30 days</option><option value={90}>Last 90 days</option></select>}</div>
        <div className="chips">{QUESTIONS.map((s) => <button key={s} className="chip" style={{ height: 34 }} onClick={() => ask(s)}>{s}</button>)}</div>
        <div className="gap8" style={{ flexWrap: "nowrap" }}><input className="in" value={q} maxLength={500} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask(q)} aria-label="Ask Telltale" placeholder="Ask about status, sensors, sound, bearing, flags or history…" /><button className="btn" disabled={!q.trim()} onClick={() => ask(q)}>Ask</button></div>
        {asks.map((i, k) => (<div className="rec" key={k} style={{ gap: 10 }}><div><div className="lbl">You</div><div>{i.q}</div></div>
          {i.loading && <Loading text="Looking through the stored evidence…" />}
          {i.error != null && <ErrorCard error={i.error} title="Could not get an answer" />}
          {i.meta && (<><div className="lbl">Telltale</div><div style={{ fontWeight: 600 }}>{i.meta.answer_summary}</div>
            <p className="prose"><Typed text={i.meta.answer_text} /></p>
            {!i.meta.answered && <span className="badge inspect" style={{ justifySelf: "start" }}>Declined: {title(i.meta.refusal_reason ?? "unsupported_question")}</span>}
            <Bullets head="Relevant facts" items={(i.record?.output as Out | undefined)?.relevant_facts} /><Meta m={i.meta} /></>)}</div>))}
      </>) : !slot ? (<div className="gap8">
        {tab === "report" && <select className="in" style={{ width: "auto" }} value={lookback} onChange={(e) => setLookback(Number(e.target.value))} aria-label="Look-back window"><option value={7}>Last 7 days</option><option value={30}>Last 30 days</option><option value={90}>Last 90 days</option></select>}
        <button className="btn" onClick={() => run(tab)}>{cur.button}</button></div>
      ) : slot.state === "loading" ? <Loading text={tab === "report" ? "Building the report…" : "Asking the AI provider. This can take up to a minute…"} />
      : slot.state === "error" ? <ErrorCard error={slot.error} title="The AI request failed" onRetry={() => run(tab as Exclude<Tab, "ask">)} />
      : slot.meta && (<>
        {tab === "explain" && <Explain meta={slot.meta as Extract<AiAnyResponse, { explanation_text: string }>} o={out} />}
        {tab === "analyze" && <Analyze meta={slot.meta as Extract<AiAnyResponse, { analysis_text: string }>} o={out} />}
        {tab === "evidence" && <Evidence meta={slot.meta as Extract<AiAnyResponse, { review_text: string }>} o={out} />}
        {tab === "report" && <Report meta={slot.meta as Extract<AiAnyResponse, { report_text: string }>} o={out} />}
        <Meta m={slot.meta} />
        <div><button className="ghost" onClick={() => run(tab as Exclude<Tab, "ask">)}>Regenerate</button></div></>)}
    </div></div>);
}
