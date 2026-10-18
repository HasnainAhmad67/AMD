import type {
  AcousticAnalysis,
  AcousticModels,
  AiAskResponse,
  AiAnalyzeResponse,
  AiEvidenceReviewResponse,
  AiExplainResponse,
  AiOutputRecord,
  AiReportResponse,
  Alert,
  ApiHealth,
  Assessment,
  AssessmentList,
  EnrollmentStatus,
  FaultDiagnosis,
  FleetOverview,
  MachineDetail,
  MachineSummary,
  MaintenanceAction,
  ModelStatus,
  SampleClip,
  SensorAssessment,
  SensorField,
  SensorReading,
  SensorSchema,
  WhatIfResponse,
} from "./types";

// Always a relative URL: Vite (dev) and nginx (prod) proxy /api to the FastAPI server.
const BASE = "/api/v1";
// Demo bearer token for /assessments and /ai (set VITE_API_TOKEN in apps/web/.env). Never a real secret: it ships in the bundle.
const TOKEN = import.meta.env.VITE_API_TOKEN as string | undefined;
export const hasApiToken = Boolean(TOKEN);

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code?: string,
    public requestId?: string,
    public details?: unknown,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit, opts: { auth?: boolean; timeoutMs?: number } = {}): Promise<T> {
  const headers = new Headers(init?.headers);
  if (opts.auth && TOKEN) headers.set("Authorization", `Bearer ${TOKEN}`);
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...init, headers, signal: opts.timeoutMs ? AbortSignal.timeout(opts.timeoutMs) : init?.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "TimeoutError") {
      throw new ApiError(0, "The request took too long and was cancelled. The AI provider may be slow: try again.", "timeout");
    }
    throw new ApiError(0, "Cannot reach the backend. Check that it is running and that VITE_API_PROXY points to it.", "network");
  }
  if (!res.ok) {
    let message = res.statusText;
    let code: string | undefined;
    let requestId: string | undefined = res.headers.get("X-Request-ID") ?? undefined;
    let details: unknown;
    try {
      const body = await res.json();
      if (body?.error) {
        // New-style envelope used by /assessments and /ai: {error:{code,message,details}, request_id}
        message = body.error.message ?? message;
        code = body.error.code;
        details = body.error.details;
        requestId = body.request_id ?? requestId;
      } else if (typeof body?.detail === "string") {
        message = body.detail; // legacy FastAPI errors
      } else if (body?.detail) {
        message = JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message, code, requestId, details);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const AI_TIMEOUT = 120_000; // AMD calls can be slow (backend timeout + retries); nginx allows 120 s

export interface AssessInput {
  machineId: string;
  sensor?: { reading: SensorReading; sourceMode: "simulated" | "dataset_demo" };
  acoustic?: File;
  bearing?: File;
  observedAt?: string; // timezone-aware ISO 8601 (e.g. new Date().toISOString())
}

/** Picks the right assessment endpoint from what was provided: 1 modality -> its own route, 2+ -> /multimodal. */
function createAssessment(input: AssessInput): Promise<Assessment> {
  const parts = [input.sensor, input.acoustic, input.bearing].filter(Boolean).length;
  if (parts === 0) throw new ApiError(0, "Add at least one input: sensor readings, a sound recording or a bearing recording.", "no_input");
  const post = (path: string, body: RequestInit) => request<Assessment>(path, { method: "POST", ...body }, { auth: true });
  if (parts >= 2) {
    const form = new FormData();
    form.append("machine_id", input.machineId);
    if (input.sensor) form.append("sensor_json", JSON.stringify({ reading: input.sensor.reading, source_mode: input.sensor.sourceMode }));
    if (input.acoustic) form.append("acoustic_file", input.acoustic);
    if (input.bearing) form.append("bearing_file", input.bearing);
    if (input.observedAt) form.append("observed_at", input.observedAt);
    return post("/assessments/multimodal", { body: form });
  }
  if (input.sensor) {
    return post("/assessments", json({
      machine_id: input.machineId,
      reading: input.sensor.reading,
      source_mode: input.sensor.sourceMode,
      ...(input.observedAt ? { observed_at: input.observedAt } : {}),
    }));
  }
  const form = new FormData();
  form.append("file", (input.acoustic ?? input.bearing) as File);
  form.append("machine_id", input.machineId);
  if (input.observedAt) form.append("observed_at", input.observedAt);
  return post(input.acoustic ? "/assessments/acoustic" : "/assessments/bearing", { body: form });
}

const ai = <T>(path: string, body: unknown) => request<T>(`/ai${path}`, json(body), { auth: true, timeoutMs: AI_TIMEOUT });

export const api = {
  health: () => request<ApiHealth>("/health"),
  models: () => request<ModelStatus>("/models"),
  reloadModels: () => request<ModelStatus>("/models/reload", { method: "POST" }),

  overview: () => request<FleetOverview>("/machines/overview"),
  machines: () => request<MachineSummary[]>("/machines"),
  machine: (id: string) => request<MachineDetail>(`/machines/${id}`),
  maintenance: (id: string, action: MaintenanceAction, note?: string) =>
    request<MachineSummary>(`/machines/${id}/maintenance`, json({ action, note })),
  summary: (id: string) =>
    request<{ machine_id: string; summary: string; source: string }>(`/machines/${id}/summary`, { method: "POST" }),

  sensorSchema: () => request<SensorSchema>("/sensors/schema"),
  predict: (reading: SensorReading) => request<SensorAssessment>("/sensors/predict", json(reading)),
  whatIf: (reading: SensorReading, parameter: SensorField, steps = 40) =>
    request<WhatIfResponse>("/sensors/what-if", json({ reading, parameter, steps })),

  alerts: (status?: string) => request<Alert[]>(`/alerts${status ? `?status=${status}` : ""}`),
  ackAlert: (id: string) => request<Alert>(`/alerts/${id}/acknowledge`, { method: "POST" }),
  resolveAlert: (id: string, note?: string) => request<Alert>(`/alerts/${id}/resolve`, json({ note })),

  acousticModels: () => request<AcousticModels>("/acoustic/models"),
  samples: () => request<SampleClip[]>("/acoustic/samples"),
  sampleFile: async (name: string) => {
    const res = await fetch(`${BASE}/acoustic/samples/${encodeURIComponent(name)}`);
    if (!res.ok) throw new ApiError(res.status, "Sample clip not found");
    return new File([await res.blob()], name, { type: "audio/wav" });
  },
  analyzeSound: (file: File, opts: { machineType?: string; machineId?: string }) => {
    const form = new FormData();
    form.append("file", file);
    if (opts.machineType) form.append("machine_type", opts.machineType);
    if (opts.machineId) form.append("machine_id", opts.machineId);
    return request<AcousticAnalysis>("/acoustic/analyze", { method: "POST", body: form });
  },
  enrolled: (machineType?: string) =>
    request<EnrollmentStatus[]>(`/acoustic/enrolled${machineType ? `?machine_type=${machineType}` : ""}`),
  diagnoseBearing: (file: File, rpm?: number) => {
    const form = new FormData();
    form.append("file", file);
    if (rpm) form.append("rpm", String(rpm));
    return request<FaultDiagnosis>("/acoustic/diagnose", { method: "POST", body: form });
  },

  // ---- assessments (Authorization: Bearer <token>)
  createAssessment,
  assessment: (id: string) => request<Assessment>(`/assessments/${id}`, undefined, { auth: true }),
  assessments: (limit = 10, machineId?: string) =>
    request<AssessmentList>(`/assessments?limit=${limit}${machineId ? `&machine_id=${encodeURIComponent(machineId)}` : ""}`, undefined, { auth: true }),

  // ---- AI features (Authorization: Bearer <token>); Explain/Analyze/Evidence use the AMD provider, Ask/Report the template provider
  aiExplain: (assessmentId: string) => ai<AiExplainResponse>("/explain", { assessment_id: assessmentId }),
  aiAnalyze: (assessmentId: string) => ai<AiAnalyzeResponse>("/analyze", { assessment_id: assessmentId }),
  aiEvidenceReview: (assessmentId: string) => ai<AiEvidenceReviewResponse>("/evidence-review", { assessment_id: assessmentId }),
  aiReport: (machineId: string, lookbackDays = 30) => ai<AiReportResponse>("/report", { machine_id: machineId, lookback_days: lookbackDays }),
  aiAsk: (question: string, scope: { assessmentId: string } | { machineId: string; lookbackDays: number }) =>
    ai<AiAskResponse>("/ask", "assessmentId" in scope
      ? { question, assessment_id: scope.assessmentId }
      : { question, machine_id: scope.machineId, lookback_days: scope.lookbackDays }),
  aiOutput: (id: string) => request<AiOutputRecord>(`/ai/outputs/${id}`, undefined, { auth: true }),
};
