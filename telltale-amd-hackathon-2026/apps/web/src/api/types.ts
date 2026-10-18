// Mirrors apps/api/telltale_api/schemas/*.py — keep in sync when the API changes.

export type RiskLevel = "low" | "medium" | "high";
export type SoundStatus = "normal" | "watch" | "anomalous";
export type Health = "ok" | "watch" | "critical" | "unknown";
export type Priority = "high" | "medium" | "low";

export const SENSOR_FIELDS = [
  "air_temperature_k",
  "process_temperature_k",
  "rotational_speed_rpm",
  "torque_nm",
  "tool_wear_min",
] as const;
export type SensorField = (typeof SENSOR_FIELDS)[number];

export interface SensorReading {
  product_quality: "L" | "M" | "H";
  air_temperature_k: number;
  process_temperature_k: number;
  rotational_speed_rpm: number;
  torque_nm: number;
  tool_wear_min: number;
}

export interface Recommendation {
  title: string;
  reason: string;
  priority: Priority;
  steps: string[];
  source: string;
}

export interface FailureModeScore {
  code: string;
  name: string;
  probability: number;
  threshold: number;
  reliable: boolean;
  flagged: boolean;
}

export interface Contribution {
  feature: string;
  label: string;
  value: number;
  contribution: number;
}

export interface SensorAssessment {
  failure_probability: number;
  risk_level: RiskLevel;
  decision_threshold: number;
  failure_modes: FailureModeScore[];
  contributions: Contribution[];
  base_value: number;
  derived: Record<string, number>;
  warnings: string[];
  recommendations: Recommendation[];
}

export interface WhatIfPoint {
  value: number;
  failure_probability: number;
  risk_level: RiskLevel;
  top_mode: string | null;
  top_mode_probability: number | null;
}

export interface WhatIfResponse {
  parameter: SensorField;
  baseline_value: number;
  baseline_probability: number;
  points: WhatIfPoint[];
}

export interface FeatureRange {
  min: number;
  max: number;
  p01: number;
  p50: number;
  p99: number;
}

export interface SensorSchema {
  sensor_fields: SensorField[];
  feature_labels: Record<string, string>;
  feature_ranges: Record<string, FeatureRange>;
  risk_thresholds: { medium: number; high: number };
}

export interface AcousticSnapshot {
  machine_type: string;
  status: SoundStatus;
  score: number;
  severity: number;
  analyzed_at: string;
  filename: string | null;
}

export interface HistoryPoint {
  t: string;
  reading: SensorReading;
  failure_probability: number | null;
  risk_level: RiskLevel | null;
  top_mode: string | null;
}

export interface MachineSummary {
  id: string;
  name: string;
  line: string;
  product_quality: string;
  acoustic_profile: string | null;
  simulated_scenario: string;
  health: Health;
  latest_reading: SensorReading | null;
  failure_probability: number | null;
  risk_level: RiskLevel | null;
  top_mode: string | null;
  acoustic: AcousticSnapshot | null;
  open_alerts: number;
  sparkline: number[];
  tool_life: ToolLife | null;
}

export interface ToolLife {
  wear_min: number;
  replace_at_min: number;
  remaining_use_min: number;
  eta_minutes: number | null;
}

export interface MachineDetail extends MachineSummary {
  assessment: SensorAssessment | null;
  history: HistoryPoint[];
}

export interface FleetOverview {
  machines: number;
  by_health: Record<Health, number>;
  open_alerts: number;
  models_online: { sensor: boolean; acoustic: boolean; bearing: boolean };
  simulator: boolean;
}

export type MaintenanceAction = "replace_tool" | "service_cooling" | "reduce_load" | "inspect_drive";

export interface Alert {
  id: string;
  machine_id: string;
  machine_name: string;
  source: "sensor" | "acoustic";
  severity: "warning" | "critical";
  title: string;
  detail: string;
  status: "open" | "acknowledged" | "resolved";
  created_at: string;
  updated_at: string;
  note: string | null;
}

export interface AcousticAnalysis {
  machine_type: string;
  machine_id: string | null;
  filename: string | null;
  method: "ensemble" | "autoencoder";
  score: number;
  threshold: number;
  severity: number;
  p_value: number | null;
  unusual_percentile: number | null;
  calibration: string | null;
  status: SoundStatus;
  is_anomalous: boolean;
  duration_s: number;
  sample_rate: number;
  peak_time_s: number;
  frame_errors: { times: number[]; values: number[] };
  spectrogram: { times: number[]; band_hz: number[]; values: number[][] };
  band_deviation: { band_hz: number[]; z: number[] };
  waveform: { duration_s: number; min: number[]; max: number[] };
  recommendations: Recommendation[];
  model: Record<string, unknown>;
}

export interface FaultDiagnosis {
  filename: string | null;
  predicted_class: string;
  label: string;
  confidence: number;
  prediction_set: string[];
  prediction_set_labels: string[];
  coverage: number | null;
  uncertain: boolean;
  rpm_used: number | null;
  rpm_assumed: boolean;
  segment_agreement: number;
  segments: number;
  duration_s: number;
  probabilities: { class_name: string; label: string; probability: number }[];
  recommendations: Recommendation[];
  model: Record<string, unknown>;
}

export interface EnrollmentStatus {
  machine_type: string;
  name: string;
  clips: number;
  calibrated: boolean;
  clips_needed_for_calibration: number;
}

export interface SampleClip {
  name: string;
  machine_type: string | null;
  label: string | null;
  size_bytes: number;
}

export interface ApiHealth {
  status: string;
  version: string;
  models: { sensor: boolean; acoustic: boolean; bearing: boolean };
  acoustic_machine_types: string[];
  simulator: boolean;
  llm: boolean;
  sensor_bootstrap: { status: "idle" | "running" | "done" | "failed"; message: string };
}

export interface AcousticModels {
  anomaly: { machine_type: string; dataset: string; threshold: number; test_auc: number | null }[];
  bearing: { classes: string[]; class_labels: Record<string, string>; test_macro_f1: number } | null;
}

// /api/v1/models is intentionally loose: it is displayed, not computed on.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ModelStatus = Record<string, any>;

// ---------------------------------------------------------------- assessments (auth required)
// Mirrors apps/api/telltale_api/schemas/assessment.py
export type OverallStatus = "normal" | "watch" | "critical" | "unknown";
export type ModalityKind = "sensor" | "acoustic" | "bearing";
export type SourceMode = "dataset_demo" | "simulated" | "uploaded" | "live";

export interface ModelIdentity {
  name: string | null;
  version: string | null;
  trained_at: string | null;
  artifact_fingerprint: string | null;
  preprocessing_fingerprint: string | null;
  training_dataset: string | null;
}

export interface ModalityRecord {
  kind: ModalityKind;
  availability: "available" | "unavailable" | "unsupported" | "invalid" | "error";
  reason_code: string;
  source_mode: SourceMode;
  observed_at: string | null;
  inference_at: string | null;
  model_identity: ModelIdentity | null;
  recording_id: string | null;
  reference_id: string | null;
  limitations: string[];
  output: SensorAssessment | AcousticAnalysis | FaultDiagnosis | null;
  error_detail: string | null;
  recording_sha256: string | null;
  result_id: string;
  illustrative: boolean;
}

export interface EvidenceItem {
  kind: string;
  value: number | string | boolean;
  unit: string | null;
  source: string;
  source_mode: SourceMode;
  observed_at: string | null;
  evidence_id: string;
  illustrative: boolean;
}

export interface QualityFlag {
  code: string;
  severity: "info" | "warning" | "critical";
  explanation: string;
}

export interface Assessment {
  assessment_id: string;
  machine_id: string;
  source_mode: SourceMode;
  schema_version: string;
  rule_version: string;
  created_at: string;
  expected_modalities: ModalityKind[];
  modalities: ModalityRecord[];
  evidence: EvidenceItem[];
  quality_flags: QualityFlag[];
  overall_status: OverallStatus;
  status_reasons: string[];
  evidence_completeness: { expected: number; present: number; ratio: number };
  approved_playbook_item_ids: string[];
}

export interface AssessmentList {
  items: Assessment[];
  limit: number;
  offset: number;
}

// ---------------------------------------------------------------- AI (auth required)
// Mirrors apps/api/telltale_api/schemas/ai.py
export type ValidationStatus = "validated" | "fallback" | "refused";
export type AiFeature = "explain" | "analyze" | "evidence_review" | "report" | "ask";

export interface AiMeta {
  ai_output_id: string;
  kind: AiFeature;
  validation_status: ValidationStatus;
  provider: string; // "amd" | "template"
  model: string | null;
  prompt_version: string;
  referenced_assessment_ids: string[];
  created_at: string;
}
export interface AiExplainResponse extends AiMeta { assessment_id: string; explanation_text: string }
export interface AiAnalyzeResponse extends AiMeta { assessment_id: string; analysis_text: string }
export interface AiEvidenceReviewResponse extends AiMeta { assessment_id: string; review_text: string }
export interface AiReportResponse extends AiMeta {
  machine_id: string;
  window_start: string | null;
  window_end: string | null;
  assessment_count: number;
  history_truncated: boolean;
  report_text: string;
}
export interface AiAskResponse extends AiMeta {
  scope: "assessment" | "machine";
  question_echo: string;
  answered: boolean;
  refusal_reason: string | null;
  answer_summary: string;
  answer_text: string;
}
export type AiAnyResponse = AiExplainResponse | AiAnalyzeResponse | AiEvidenceReviewResponse | AiReportResponse | AiAskResponse;

// GET /ai/outputs/{id}: the full stored record. `output` is the structured, grounding-checked payload
// (its shape depends on the feature: see schemas/ai.py), so it is read loosely and displayed only.
export interface AiOutputRecord {
  feature: AiFeature;
  ai_output_id: string;
  kind: AiFeature;
  assessment_ids: string[];
  provider: string;
  model: string | null;
  prompt_version: string;
  validation_status: ValidationStatus;
  created_at: string;
  response_text: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  output: Record<string, any>;
}
