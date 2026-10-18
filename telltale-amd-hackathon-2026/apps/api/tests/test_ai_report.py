"""Central AI Analysis Agent - slice D tests: POST /api/v1/ai/report.

Covers the approved slice-D contract: strict request boundary (machine_id + bounded
lookback, extra="forbid"), owner resolution before evidence access, read-only use of
immutable snapshots, newest-first owner-scoped history retrieval capped at 50 with
explicit truncation disclosure, a window anchored ONLY on stored created_at values
(never wall-clock time), deterministic refusal for an empty window, the structured
facts-only report (status distribution, modality availability, recurring flags,
missing-data patterns, non-binding review suggestions, non-claims), strict
evidence-grounding validation (invented ids/statuses/flags/numbers/quoted tokens and
forbidden claim wording are rejected), fallback/refusal behaviour, append-only
persistence (same ai_outputs table, kind="report", no new migration), GET parity via
the shared owner-scoped output route, restart persistence, immutability, prompt-digest
determinism, and unchanged /ai/explain, /ai/evidence-review, /ai/analyze and legacy
assessment routes.

No downloads, no GPUs, no network, no provider: snapshots are built through the real
assess/v1 service; the conftest fixture models are used only by the legacy-route
regression checks.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from telltale_api.api.routes import ai as ai_routes
from telltale_api.core.config import Settings
from telltale_api.main import create_app
from telltale_api.schemas.acoustic import (
    AcousticAnalysis,
    BandDeviation,
    FaultDiagnosis,
    FaultProbability,
    Spectrogram,
    Series,
    Waveform,
)
from telltale_api.schemas.ai import (
    AiReportOutput,
    AiReportRequest,
    AiStatusCount,
)
from telltale_api.schemas.assessment import EvidenceDraft, ModalityDraft, ModelIdentity
from telltale_api.schemas.sensor import Contribution, FailureModeScore, SensorAssessment
from telltale_api.services.ai_report_template import (
    TemplateReportProvider,
    modality_availability,
    recurring_flags,
    status_distribution,
)
from telltale_api.services.ai_report_validate import validate_report_output
from telltale_api.services.assess import create_assessment

URL_REPORT = "/api/v1/ai/report"
URL_EXPLAIN = "/api/v1/ai/explain"
URL_REVIEW = "/api/v1/ai/evidence-review"
URL_ANALYZE = "/api/v1/ai/analyze"
URL_BASE = "/api/v1/assessments"

TOKENS = json.dumps({"alice": "tok-alice-9f3d2c7e", "bob": "tok-bob-77aa11bb"})
AUTH_ALICE = {"Authorization": "Bearer tok-alice-9f3d2c7e"}
AUTH_BOB = {"Authorization": "Bearer tok-bob-77aa11bb"}

MACHINE = "cnc-a1"
NOW = datetime(2026, 10, 16, 2, 0, 0, tzinfo=timezone.utc)
READING = {
    "product_quality": "M",
    "air_temperature_k": 302.4,
    "process_temperature_k": 310.6,
    "rotational_speed_rpm": 1352.0,
    "torque_nm": 54.2,
    "tool_wear_min": 212.0,
}

REPORT_RESPONSE_KEYS = {
    "feature", "ai_output_id", "machine_id", "kind", "window_start", "window_end",
    "assessment_count", "history_truncated", "report_text", "validation_status",
    "provider", "model", "prompt_version", "referenced_assessment_ids", "created_at",
}

SENSOR_IDENTITY = ModelIdentity(
    name="sensor-lightgbm", version="9.9.9",
    trained_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    artifact_fingerprint="a" * 64, preprocessing_fingerprint=None,
    training_dataset="ai4i2020_synthetic",
)
ACOUSTIC_IDENTITY = ModelIdentity(
    name="acoustic-fan-ensemble", version="9.9.9",
    trained_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    artifact_fingerprint="b" * 64, preprocessing_fingerprint="c" * 64,
    training_dataset="mimii_due",
)
BEARING_IDENTITY = ModelIdentity(
    name="bearing-lightgbm", version="9.9.9",
    trained_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    artifact_fingerprint="d" * 64, preprocessing_fingerprint="e" * 64,
    training_dataset="mendeley_n9y9c7xrz3",
)


def parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def sensor_draft(*, risk: str = "high", probability: float = 0.8237,
                 observed: datetime | None = NOW) -> ModalityDraft:
    output = SensorAssessment(
        failure_probability=probability,
        risk_level=risk,  # type: ignore[arg-type]
        decision_threshold=0.4569,
        failure_modes=[
            FailureModeScore(
                code="TWF", name="Tool wear", probability=0.21, threshold=0.25,
                flagged=(risk == "high"),
            )
        ],
        contributions=[
            Contribution(feature="tool_wear_min", label="Tool wear", value=212.0, contribution=0.51)
        ],
        base_value=-2.0,
        derived={"tool_wear_slope": 1.0},
        warnings=[],
    )
    return ModalityDraft(
        kind="sensor", availability="available", reason_code="ok", source_mode="simulated",
        observed_at=observed, inference_at=NOW, model_identity=SENSOR_IDENTITY,
        limitations=["synthetic dataset caveat"], output=output,
    )


def acoustic_draft(*, status: str = "normal", observed: datetime | None = NOW) -> ModalityDraft:
    output = AcousticAnalysis(
        machine_type="fan", method="ensemble", score=-0.4181, threshold=1.3392,
        severity=0.3119, p_value=0.6047, unusual_percentile=60.47, calibration="global",
        status=status,  # type: ignore[arg-type]
        is_anomalous=(status == "anomalous"), duration_s=10.0, sample_rate=16000,
        peak_time_s=5.0, frame_errors=Series(times=[1.0], values=[-0.4]),
        spectrogram=Spectrogram(times=[1.0], band_hz=[125.0], values=[[-20.0]]),
        band_deviation=BandDeviation(band_hz=[125.0], z=[0.1]),
        waveform=Waveform(duration_s=10.0, min=[-0.1], max=[0.1]),
    )
    return ModalityDraft(
        kind="acoustic", availability="available", reason_code="ok", source_mode="uploaded",
        observed_at=observed, inference_at=NOW + timedelta(seconds=1),
        model_identity=ACOUSTIC_IDENTITY, recording_id=str(uuid4()),
        recording_sha256="f" * 64, reference_id="sha256:" + "0" * 64,
        limitations=["trained on normal sounds only; flags difference, not a specific fault"],
        output=output,
    )


def bearing_draft(*, uncertain: bool = False, plain: bool = False,
                  observed: datetime | None = NOW) -> ModalityDraft:
    if plain:
        classes, labels, probs = ["normal"], ["Healthy bearing"], [
            FaultProbability(class_name="normal", label="Healthy bearing", probability=0.95)
        ]
        predicted, label, confidence = "normal", "Healthy bearing", 0.95
    else:
        classes, labels = ["inner_race", "normal"], ["Inner race defect", "Healthy bearing"]
        probs = [
            FaultProbability(class_name="inner_race", label="Inner race defect", probability=0.8123),
            FaultProbability(class_name="normal", label="Healthy bearing", probability=0.1877),
        ]
        predicted, label, confidence = "inner_race", "Inner race defect", 0.8123
    output = FaultDiagnosis(
        predicted_class=predicted, label=label, confidence=confidence,
        prediction_set=classes, prediction_set_labels=labels, coverage=0.9,
        uncertain=uncertain, rpm_used=1200.0, rpm_assumed=True, segment_agreement=0.88,
        segments=39, duration_s=20.0, probabilities=probs,
    )
    return ModalityDraft(
        kind="bearing", availability="available", reason_code="ok", source_mode="uploaded",
        observed_at=observed, inference_at=NOW + timedelta(seconds=2),
        model_identity=BEARING_IDENTITY, recording_id=str(uuid4()),
        recording_sha256="a" * 64, reference_id="sha256:" + "1" * 64,
        limitations=["lab rig with artificial defects; real machines differ"], output=output,
    )


BUILDERS = {
    "sensor": lambda **kw: sensor_draft(**kw),
    "acoustic": lambda **kw: acoustic_draft(**kw),
    "bearing": lambda **kw: bearing_draft(**kw),
}

EVIDENCE_VALUE = {"sensor": 0.8237, "acoustic": -0.4181, "bearing": 0.8123}
EVIDENCE_KIND = {"sensor": "failure_probability", "acoustic": "anomaly_score",
                 "bearing": "class_confidence"}


def make_snapshot(bits=("sensor",), *, machine_id=MACHINE, created_at=None,
                  observed: datetime | None = NOW, source_mode: str = "uploaded",
                  drop=None, **per_kind):
    """Deterministic snapshot through the real assess/v1 service."""
    modalities = []
    for kind in bits:
        if drop == kind:
            modalities.append(
                ModalityDraft(
                    kind=kind, availability="unavailable", reason_code="not_submitted",
                    source_mode="simulated" if kind == "sensor" else "uploaded",
                    observed_at=observed, inference_at=NOW, model_identity=None,
                    limitations=[], output=None,
                )
            )
        else:
            modalities.append(BUILDERS[kind](observed=observed, **per_kind.get(kind, {})))
    evidence = [
        EvidenceDraft(
            kind=EVIDENCE_KIND[kind], value=EVIDENCE_VALUE[kind],
            unit="probability" if kind != "acoustic" else "score", source=kind,
            source_mode="simulated" if kind == "sensor" else "uploaded", observed_at=observed,
        )
        for kind in bits if drop != kind
    ]
    return create_assessment(
        machine_id=machine_id, source_mode=source_mode, expected_modalities=list(bits),
        modalities=modalities, evidence=evidence, extra_flags=[], created_at=created_at,
    )


@pytest.fixture()
def client_factory(models_dir, tmp_path):
    """Build a TestClient over the session fixture models (caller enters the context)."""

    def _make(**over):
        over.setdefault("api_tokens", TOKENS)
        settings = Settings(
            models_dir=over.pop("models_dir", models_dir),
            data_dir=over.pop("data_dir", tmp_path),
            simulator_enabled=True,
            simulator_interval_s=3600,  # one startup tick, then quiet
            llm_base_url=None,
            **over,
        )
        return TestClient(create_app(settings))

    return _make


def seed(client, bits=("sensor",), *, owner="alice", **kw):
    snapshot = make_snapshot(bits, **kw)
    client.app.state.telltale.assessments.insert(snapshot, owner=owner)
    return snapshot


def report(client, machine_id=MACHINE, auth=AUTH_ALICE, **over):
    body = {"machine_id": machine_id}
    body.update(over)
    return client.post(URL_REPORT, json=body, headers=auth)


def assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error", "request_id"}, body
    assert body["error"]["code"] == code, body
    assert re.fullmatch(r"[0-9a-f]{32}", body["request_id"]), body["request_id"]
    return body


def ai_rows(tmp_path) -> int:
    with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
        return conn.execute("SELECT COUNT(*) FROM ai_outputs").fetchone()[0]


def ai_kinds(tmp_path) -> list[str]:
    with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
        return [row[0] for row in conn.execute("SELECT kind FROM ai_outputs")]


# ---------------------------------------------------------------- empty + single + multi


def test_empty_history_window_produces_persisted_refusal(client_factory, tmp_path):
    with client_factory() as client:
        resp = report(client, machine_id="no-such-machine")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert set(body) == REPORT_RESPONSE_KEYS
        assert body["validation_status"] == "refused"
        assert body["assessment_count"] == 0
        assert body["referenced_assessment_ids"] == []
        assert body["window_start"] is None and body["window_end"] is None
        assert "no maintenance report is offered" in body["report_text"]
        for phrase in ("not a fault diagnosis", "not machine-health completeness",
                       "not coverage", "not a fused failure probability",
                       "not a guaranteed maintenance plan"):
            assert phrase in body["report_text"], phrase
        record = client.get(
            f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        assert record["kind"] == "report"
        assert record["validation_status"] == "refused"
        assert record["output"]["assessment_count"] == 0
        assert record["assessment_ids"] == []
        assert ai_rows(tmp_path) == 1


def test_single_assessment_report(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor",))
        resp = report(client, lookback_days=30)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert set(body) == REPORT_RESPONSE_KEYS
        assert body["validation_status"] == "validated"
        assert body["feature"] == "report"
        assert body["kind"] == "report"
        assert body["provider"] == "template"
        assert body["model"] is None
        assert body["prompt_version"] == "report/v1"
        assert body["assessment_count"] == 1
        assert body["history_truncated"] is False
        assert body["referenced_assessment_ids"] == [str(snapshot.assessment_id)]
        assert parse_dt(body["window_end"]) == snapshot.created_at
        assert parse_dt(body["window_start"]) == snapshot.created_at - timedelta(days=30)
        assert resp.headers["Location"] == f"/api/v1/ai/outputs/{body['ai_output_id']}"
        record = client.get(
            f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        assert record["response_text"] == body["report_text"]
        assert record["output"]["report_text"] == body["report_text"]
        assert record["assessment_ids"] == [str(snapshot.assessment_id)]
        assert re.fullmatch(r"[0-9a-f]{64}", record["prompt_sha256"])


def test_multiple_assessments_with_mixed_modalities(client_factory, tmp_path):
    with client_factory() as client:
        base = NOW
        seeds = [
            seed(client, ("sensor",), created_at=base, **{"sensor": {"risk": "high"}}),
            seed(client, ("acoustic",), created_at=base - timedelta(minutes=1),
                 **{"acoustic": {"status": "watch"}}),
            seed(client, ("sensor", "bearing"), created_at=base - timedelta(minutes=2)),
            seed(client, ("sensor", "acoustic"), created_at=base - timedelta(minutes=3),
                 drop="acoustic"),
        ]
        resp = report(client)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "validated"
        assert body["assessment_count"] == 4
        assert sorted(body["referenced_assessment_ids"]) == sorted(
            str(s.assessment_id) for s in seeds
        )
        record = client.get(
            f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        output = record["output"]
        rows = {row["kind"]: row for row in output["modality_availability"]}
        assert set(rows) == {"sensor", "acoustic", "bearing"}
        # acoustic: 1 available (acoustic-only) + 1 unavailable(not_submitted, dropped)
        assert {c["state"]: c["count"] for c in rows["acoustic"]["counts"]} == {
            "available": 1, "unavailable": 1
        }
        assert "not_submitted" in rows["acoustic"]["reason_codes"]
        assert output["recurring_flags"]
        assert output["review_suggestions"]
        assert all(s["scope"] == "review_suggestion" for s in output["review_suggestions"])


def test_status_distribution_counts_match_snapshots(client_factory, tmp_path):
    with client_factory() as client:
        base = NOW
        seed(client, ("sensor",), created_at=base, **{"sensor": {"risk": "high"}})  # critical
        seed(client, ("sensor",), created_at=base - timedelta(hours=1),
             **{"sensor": {"risk": "low", "probability": 0.01}})  # normal
        seed(client, ("acoustic",), created_at=base - timedelta(hours=2),
             **{"acoustic": {"status": "watch"}})  # watch
        seed(client, ("sensor", "acoustic"), created_at=base - timedelta(hours=3),
             drop="acoustic", **{"sensor": {"risk": "low", "probability": 0.01}})  # unknown
        resp = report(client)
        assert resp.status_code == 201, resp.text
        record = client.get(
            f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        got = {row["status"]: row["count"] for row in record["output"]["status_distribution"]}
        assert got == {"normal": 1, "watch": 1, "critical": 1, "unknown": 1}


def test_recurring_flags_and_missing_data_patterns_match_stored_facts(client_factory, tmp_path):
    with client_factory() as client:
        base = NOW
        s1 = seed(client, ("sensor", "acoustic"), created_at=base, observed=None, drop="acoustic")
        s2 = seed(client, ("sensor",), created_at=base - timedelta(hours=1), observed=None)
        resp = report(client)
        assert resp.status_code == 201, resp.text
        record = client.get(
            f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        output = record["output"]
        flags = {f["code"]: f["count"] for f in output["recurring_flags"]}
        assert flags.get("TIMESTAMP_MISSING") == 2
        assert flags.get("SOURCE_ILLUSTRATIVE") == 2
        assert any("TIMESTAMP_MISSING" in p for p in output["missing_data_patterns"])
        assert any(
            'acoustic evidence is "unavailable" (stored reason "not_submitted")' in p
            for p in output["missing_data_patterns"]
        )
        # The stored flag/severity/count triple must equal the recomputation.
        assert output["recurring_flags"] == [
            f.model_dump() for f in recurring_flags([s2, s1])
        ]


# ---------------------------------------------------------------- window + truncation


def test_window_filtering_uses_stored_created_at(client_factory, tmp_path):
    with client_factory() as client:
        base = NOW
        newest = seed(client, ("sensor",), created_at=base)
        inside = seed(client, ("acoustic",), created_at=base - timedelta(days=10),
                      **{"acoustic": {"status": "watch"}})
        seed(client, ("sensor",), created_at=base - timedelta(days=40))  # outside 30-day window
        resp = report(client, lookback_days=30)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["assessment_count"] == 2
        assert sorted(body["referenced_assessment_ids"]) == sorted(
            [str(newest.assessment_id), str(inside.assessment_id)]
        )
        assert parse_dt(body["window_end"]) == newest.created_at
        assert parse_dt(body["window_start"]) == newest.created_at - timedelta(days=30)


def test_window_is_not_anchored_on_wall_clock(client_factory, tmp_path):
    with client_factory() as client:
        old = datetime(2020, 1, 2, tzinfo=timezone.utc)
        seed(client, ("sensor",), created_at=old)
        seed(client, ("acoustic",), created_at=old - timedelta(days=10),
             **{"acoustic": {"status": "watch"}})
        resp = report(client, lookback_days=30)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["assessment_count"] == 2, "wall-clock time must never be the anchor"
        assert parse_dt(body["window_end"]) == old


def test_history_truncation_at_50_with_disclosure(client_factory, tmp_path):
    with client_factory() as client:
        base = NOW
        seeded = []
        for i in range(51):
            seeded.append(
                seed(client, ("sensor",), created_at=base - timedelta(minutes=i),
                     **{"sensor": {"risk": "high"}})
            )
        resp = report(client, lookback_days=90)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "validated"
        assert body["assessment_count"] == 50
        assert body["history_truncated"] is True
        assert len(body["referenced_assessment_ids"]) == 50
        # newest-first: the 51st (oldest) assessment must be dropped
        assert str(seeded[50].assessment_id) not in body["referenced_assessment_ids"]
        assert str(seeded[0].assessment_id) in body["referenced_assessment_ids"]
        assert "History truncated: true" in body["report_text"]
        record = client.get(
            f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        output = record["output"]
        assert output["history_truncated"] is True
        assert any("truncated" in p for p in output["missing_data_patterns"])
        assert any("truncated" in s["title"].lower() or "truncated" in s["basis"].lower()
                   for s in output["review_suggestions"])
        assert any("truncated" in item.lower() for item in output["limitations"])


# ---------------------------------------------------------------- owner isolation + no leak


def test_owner_isolation_and_no_history_leak(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"), owner="alice")
        alice = report(client, auth=AUTH_ALICE)
        assert alice.status_code == 201, alice.text
        assert alice.json()["assessment_count"] == 1
        oid = alice.json()["ai_output_id"]

        # bob sees no history for alice's machine: same refusal as a nonexistent machine
        bob = report(client, auth=AUTH_BOB)
        unknown = report(client, machine_id="not-a-real-machine", auth=AUTH_BOB)
        assert bob.status_code == 201 and unknown.status_code == 201
        assert bob.json()["assessment_count"] == 0
        assert bob.json()["validation_status"] == "refused"
        assert unknown.json()["assessment_count"] == 0
        assert bob.json()["referenced_assessment_ids"] == unknown.json()["referenced_assessment_ids"] == []

        # bob cannot read alice's stored report
        assert client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_BOB).status_code == 404
        assert client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE).status_code == 200
        # alice's assessment stays owner-scoped too
        assert client.get(f"{URL_BASE}/{snapshot.assessment_id}",
                          headers=AUTH_BOB).status_code == 404


# ---------------------------------------------------------------- grounding validation (unit)


def _ordered(snapshots):
    return sorted(snapshots, key=lambda a: a.created_at, reverse=True)


def test_grounding_rejects_invented_facts(client_factory):
    with client_factory() as client:
        base = NOW
        seeded = [
            seed(client, ("sensor",), created_at=base, **{"sensor": {"risk": "high"}}),
            seed(client, ("sensor", "bearing"), created_at=base - timedelta(hours=1)),
            seed(client, ("acoustic",), created_at=base - timedelta(hours=2),
                 **{"acoustic": {"status": "anomalous"}}),
        ]
        snapshots = _ordered(seeded)
        window_end = snapshots[0].created_at
        window_start = window_end - timedelta(days=30)
        output = TemplateReportProvider().report(
            snapshots, window_start=window_start, window_end=window_end,
            history_truncated=False, lookback_days=30,
        )

        def check(candidate):
            return validate_report_output(
                candidate, snapshots, machine_id=MACHINE, window_start=window_start,
                window_end=window_end, history_truncated=False, lookback_days=30,
            )

        ok, reason = check(output)
        assert ok, reason

        assert not check(output.model_copy(
            update={"referenced_assessment_ids": [str(uuid4())]}))[0]
        assert not check(output.model_copy(
            update={"referenced_evidence_ids": [str(uuid4())]}))[0]
        assert not check(output.model_copy(update={"assessment_count": 99}))[0]
        assert not check(output.model_copy(update={"machine_id": "other-machine"}))[0]
        assert not check(output.model_copy(update={"feature": "explain"}))[0]
        assert not check(output.model_copy(
            update={"window_end": window_end + timedelta(days=1)}))[0]
        assert not check(output.model_copy(
            update={"history_truncated": True}))[0]
        assert not check(output.model_copy(
            update={"status_distribution": [AiStatusCount(status="normal", count=99)]}))[0]
        assert not check(output.model_copy(update={"recurring_flags": []}))[0]
        assert not check(output.model_copy(
            update={"report_text": output.report_text + " confidence 0.99999 in the model"}))[0]
        assert not check(output.model_copy(
            update={"report_text": output.report_text + ' The verdict is "confirmed".'}))[0]
        assert not check(output.model_copy(
            update={"report_text": output.report_text + " maintenance is required"}))[0]
        assert not check(output.model_copy(
            update={"report_text": output.report_text.replace(
                "not a guaranteed maintenance plan", "a guaranteed maintenance plan")}))[0]


def test_grounding_matches_recomputed_summaries(client_factory):
    with client_factory() as client:
        base = NOW
        seeded = [
            seed(client, ("sensor",), created_at=base, **{"sensor": {"risk": "high"}}),
            seed(client, ("acoustic",), created_at=base - timedelta(hours=1),
                 **{"acoustic": {"status": "watch"}}),
        ]
        snapshots = _ordered(seeded)
        window_end = snapshots[0].created_at
        window_start = window_end - timedelta(days=30)
        output = TemplateReportProvider().report(
            snapshots, window_start=window_start, window_end=window_end,
            history_truncated=False, lookback_days=30,
        )
        assert output.status_distribution == status_distribution(snapshots)
        assert output.modality_availability == modality_availability(snapshots)
        assert output.recurring_flags == recurring_flags(snapshots)


# ---------------------------------------------------------------- strict requests


def test_invalid_requests(client_factory, tmp_path):
    with client_factory() as client:
        seed(client, ("sensor",))
        assert_error(client.post(URL_REPORT, json={}, headers=AUTH_ALICE),
                     422, "request_validation")
        assert_error(report(client, lookback_days=0), 422, "request_validation")
        assert_error(report(client, lookback_days=91), 422, "request_validation")
        assert_error(report(client, lookback_days="soon"), 422, "request_validation")
        assert_error(report(client, machine_id=MACHINE, extra_field=1), 422, "request_validation")
        malformed = client.post(
            URL_REPORT, data=b"{not json",
            headers={**AUTH_ALICE, "Content-Type": "application/json"},
        )
        assert_error(malformed, 422, "request_validation")
        oversized = client.post(URL_REPORT, data=b"x" * 70000, headers=AUTH_ALICE)
        assert_error(oversized, 422, "request_validation")
        assert_error(report(client, auth=None), 401, "unauthorized")
        assert ai_rows(tmp_path) == 0
        with pytest.raises(Exception):
            AiReportRequest.model_validate({"machine_id": MACHINE, "lookback_days": 0})
        with pytest.raises(Exception):
            AiReportRequest.model_validate({"machine_id": MACHINE, "forged": "x"})


def test_schema_strictness(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor",))
        output = TemplateReportProvider().report(
            [snapshot], window_start=snapshot.created_at - timedelta(days=30),
            window_end=snapshot.created_at, history_truncated=False, lookback_days=30,
        )
        payload = json.loads(output.model_dump_json())
        payload["forged"] = "overall_status"
        with pytest.raises(Exception):
            AiReportOutput.model_validate(payload)
        payload2 = json.loads(output.model_dump_json())
        payload2["feature"] = "analyze"
        with pytest.raises(Exception):
            AiReportOutput.model_validate(payload2)
        payload3 = json.loads(output.model_dump_json())
        payload3["report_text"] = "x"
        with pytest.raises(Exception):
            AiReportOutput.model_validate(payload3)


# ---------------------------------------------------------------- failure + persistence


def test_store_failures_produce_nothing(client_factory, tmp_path, monkeypatch):
    with client_factory() as client:
        seed(client, ("sensor",))
        ai_store = client.app.state.telltale.ai_outputs
        assert ai_store is not None

        def boom(*args, **kwargs):
            raise RuntimeError("insert failed at C:\\secret\\telltale.db")

        monkeypatch.setattr(ai_store, "insert", boom)
        body = assert_error(report(client), 500, "internal_error")
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert "secret" not in body["error"]["message"]
        assert ai_rows(tmp_path) == 0
        monkeypatch.undo()
        client.app.state.telltale.assessments = None
        assert_error(report(client), 503, "storage_unavailable")
        assert ai_rows(tmp_path) == 0


def test_ai_storage_unavailable(client_factory, tmp_path):
    with client_factory() as client:
        seed(client, ("sensor",))
        client.app.state.telltale.ai_outputs = None
        assert_error(report(client), 503, "storage_unavailable")
        assert ai_rows(tmp_path) == 0


def test_provider_failure_produces_fallback_no_partial_write(
    client_factory, tmp_path, monkeypatch
):
    with client_factory() as client:
        seed(client, ("sensor",))
        original = TemplateReportProvider.report

        def boom(self, snapshots, **kwargs):
            raise RuntimeError("report exploded")

        monkeypatch.setattr(TemplateReportProvider, "report", boom)
        resp = report(client)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "fallback"
        assert "no interpretation is offered" in body["report_text"]
        assert ai_rows(tmp_path) == 1
        monkeypatch.setattr(TemplateReportProvider, "report", original)


def test_route_falls_back_when_grounding_fails(client_factory, tmp_path, monkeypatch):
    with client_factory() as client:
        seed(client, ("sensor",))
        calls = {"n": 0}
        original = ai_routes.validate_report_output

        def flaky(output, snapshots, **kwargs):
            calls["n"] += 1
            return (True, "") if calls["n"] > 1 else (False, "forced grounding failure")

        monkeypatch.setattr(ai_routes, "validate_report_output", flaky)
        resp = report(client)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "fallback"
        assert ai_rows(tmp_path) == 1
        monkeypatch.setattr(ai_routes, "validate_report_output", original)


def test_snapshots_remain_byte_identical(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        before = client.get(f"{URL_BASE}/{snapshot.assessment_id}", headers=AUTH_ALICE)
        assert before.status_code == 200
        assert report(client).status_code == 201
        after = client.get(f"{URL_BASE}/{snapshot.assessment_id}", headers=AUTH_ALICE)
        assert after.status_code == 200
        assert before.content == after.content
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM ai_outputs").fetchone()[0] == 1


def test_restart_persistence(client_factory, tmp_path):
    with client_factory() as client:
        seed(client, ("sensor", "acoustic"))
        resp = report(client)
        assert resp.status_code == 201
        oid = resp.json()["ai_output_id"]
        first = client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE).json()
    with client_factory(data_dir=tmp_path) as client2:
        got = client2.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE)
        assert got.status_code == 200
        assert got.json() == first
        assert client2.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_BOB).status_code == 404


def test_prompt_digest_deterministic(client_factory, tmp_path):
    with client_factory() as client:
        seed(client, ("sensor", "acoustic"))
        r1 = report(client)
        r2 = report(client)
        assert r1.status_code == 201 and r2.status_code == 201
        rec1 = client.get(f"/api/v1/ai/outputs/{r1.json()['ai_output_id']}", headers=AUTH_ALICE).json()
        rec2 = client.get(f"/api/v1/ai/outputs/{r2.json()['ai_output_id']}", headers=AUTH_ALICE).json()
        assert rec1["prompt_sha256"] == rec2["prompt_sha256"]
        assert rec1["ai_output_id"] != rec2["ai_output_id"]
        assert rec1["output"]["report_text"] == rec2["output"]["report_text"]
        assert ai_rows(tmp_path) == 2


# ---------------------------------------------------------------- secrets + migration + regression


def test_no_secrets_or_paths_in_stored_report(client_factory, tmp_path):
    with client_factory() as client:
        seed(client, ("sensor", "acoustic", "bearing"))
        resp = report(client)
        assert resp.status_code == 201
        record = client.get(
            f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        blob = json.dumps(record)
        for forbidden in ("C:\\", "telltale.db", "models/sensor", "models/acoustic",
                          "models/bearing", "tok-alice", "tok-bob", "schema_migrations",
                          "SELECT ", "INSERT "):
            assert forbidden not in blob, forbidden
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            raw = conn.execute("SELECT json FROM ai_outputs").fetchone()[0]
        for forbidden in ("C:\\", "telltale.db", "models/", "tok-alice", "tok-bob"):
            assert forbidden not in raw, forbidden


def test_no_new_migration(client_factory, tmp_path):
    with client_factory() as client:
        seed(client, ("sensor",))
        report(client)
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            versions = dict(conn.execute("SELECT version, name FROM schema_migrations").fetchall())
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            columns = {r[1] for r in conn.execute("PRAGMA table_info(ai_outputs)")}
        assert versions == {1: "assessments_v1", 2: "assessments_owner_v2", 3: "ai_outputs_v3"}
        assert {"assessments", "ai_outputs"} <= tables
        assert {"assessment_ids", "kind", "json"} <= columns
        assert ai_kinds(tmp_path) == ["report"]


def test_other_ai_routes_unchanged(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        r_explain = client.post(URL_EXPLAIN, json={"assessment_id": str(snapshot.assessment_id)},
                                headers=AUTH_ALICE)
        assert r_explain.status_code == 201 and r_explain.json()["feature"] == "explain"
        r_review = client.post(URL_REVIEW, json={"assessment_id": str(snapshot.assessment_id)},
                               headers=AUTH_ALICE)
        assert r_review.status_code == 201 and r_review.json()["feature"] == "evidence_review"
        r_analyze = client.post(URL_ANALYZE, json={"assessment_id": str(snapshot.assessment_id)},
                                headers=AUTH_ALICE)
        assert r_analyze.status_code == 201 and r_analyze.json()["feature"] == "analyze"
        assert ai_rows(tmp_path) == 3


def test_legacy_assessment_routes_unchanged(client_factory):
    with client_factory() as client:
        resp = client.post(
            URL_BASE,
            json={"reading": READING, "machine_id": MACHINE, "source_mode": "simulated"},
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["sensor"]
