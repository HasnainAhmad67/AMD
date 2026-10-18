"""Central AI Analysis Agent - slice A tests: POST /api/v1/ai/explain.

Covers the approved slice-A contract: strict request boundary (extra/forged fields
rejected), owner resolution BEFORE any evidence read, read-only use of immutable
assessment snapshots (no ML calls, no audio, no fleet telemetry), deterministic
template generation, evidence-grounding validation (invented ids/statuses/numbers
are rejected, never persisted as validated), refusal/fallback behavior, append-only
persistence with owner scoping, restart persistence, immutability of the explained
snapshots, no secrets/paths in stored artifacts, and unchanged legacy routes.

Snapshots are built directly from the assess/v1 service (deterministic, no ML); the
only fixture models used are the conftest-trained tiny models for the legacy-route
regression check. No downloads, no GPUs, no network, no provider.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from telltale_api.api.routes import ai as ai_routes
from telltale_api.core.config import Settings
from telltale_api.main import create_app
from telltale_api.schemas.acoustic import AcousticAnalysis, BandDeviation, Spectrogram, Series, Waveform
from telltale_api.schemas.assessment import EvidenceDraft, ModalityDraft, ModelIdentity
from telltale_api.schemas.ai import AiExplainOutput, AiExplainRequest
from telltale_api.schemas.sensor import Contribution, FailureModeScore, SensorAssessment
from telltale_api.schemas.acoustic import FaultDiagnosis, FaultProbability
from telltale_api.services.ai_template import TemplateExplainProvider
from telltale_api.services.ai_validate import validate_explain_output
from telltale_api.services.assess import create_assessment

URL_EXPLAIN = "/api/v1/ai/explain"
URL_BASE = "/api/v1/assessments"

TOKENS = json.dumps({"alice": "tok-alice-9f3d2c7e", "bob": "tok-bob-77aa11bb"})
AUTH_ALICE = {"Authorization": "Bearer tok-alice-9f3d2c7e"}
AUTH_BOB = {"Authorization": "Bearer tok-bob-77aa11bb"}

MACHINE = "cnc-a1"
NOW = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
READING = {
    "product_quality": "M",
    "air_temperature_k": 302.4,
    "process_temperature_k": 310.6,
    "rotational_speed_rpm": 1352.0,
    "torque_nm": 54.2,
    "tool_wear_min": 212.0,
}

SENSOR_IDENTITY = ModelIdentity(
    name="sensor-lightgbm",
    version="9.9.9",
    trained_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    artifact_fingerprint="a" * 64,
    preprocessing_fingerprint=None,
    training_dataset="ai4i2020_synthetic",
)
ACOUSTIC_IDENTITY = ModelIdentity(
    name="acoustic-fan-ensemble",
    version="9.9.9",
    trained_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    artifact_fingerprint="b" * 64,
    preprocessing_fingerprint="c" * 64,
    training_dataset="mimii_due",
)
BEARING_IDENTITY = ModelIdentity(
    name="bearing-lightgbm",
    version="9.9.9",
    trained_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    artifact_fingerprint="d" * 64,
    preprocessing_fingerprint="e" * 64,
    training_dataset="mendeley_n9y9c7xrz3",
)


def sensor_draft(
    *, risk: str = "high", probability: float = 0.8237, observed: datetime | None = None
) -> ModalityDraft:
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
            Contribution(
                feature="tool_wear_min", label="Tool wear", value=212.0, contribution=0.51
            )
        ],
        base_value=-2.0,
        derived={"tool_wear_slope": 1.0},
        warnings=[],
    )
    return ModalityDraft(
        kind="sensor",
        availability="available",
        reason_code="ok",
        source_mode="simulated",
        observed_at=observed,
        inference_at=NOW,
        model_identity=SENSOR_IDENTITY,
        limitations=["synthetic dataset caveat", "SHAP explains the model, not physics"],
        output=output,
    )


def acoustic_draft(*, status: str = "normal", observed: datetime | None = None) -> ModalityDraft:
    output = AcousticAnalysis(
        machine_type="fan",
        method="ensemble",
        score=-0.4181,
        threshold=1.3392,
        severity=0.3119,
        p_value=0.6047,
        unusual_percentile=60.47,
        calibration="global",
        status=status,  # type: ignore[arg-type]
        is_anomalous=(status == "anomalous"),
        duration_s=10.0,
        sample_rate=16000,
        peak_time_s=5.0,
        frame_errors=Series(times=[1.0], values=[-0.4]),
        spectrogram=Spectrogram(times=[1.0], band_hz=[125.0], values=[[-20.0]]),
        band_deviation=BandDeviation(band_hz=[125.0], z=[0.1]),
        waveform=Waveform(duration_s=10.0, min=[-0.1], max=[0.1]),
    )
    return ModalityDraft(
        kind="acoustic",
        availability="available",
        reason_code="ok",
        source_mode="uploaded",
        observed_at=observed,
        inference_at=NOW + timedelta(seconds=1),
        model_identity=ACOUSTIC_IDENTITY,
        recording_id=str(uuid4()),
        recording_sha256="f" * 64,
        reference_id="sha256:" + "0" * 64,
        limitations=["trained on normal sounds only; flags difference, not a specific fault"],
        output=output,
    )


def bearing_draft(*, uncertain: bool = False, observed: datetime | None = None) -> ModalityDraft:
    output = FaultDiagnosis(
        predicted_class="inner_race",
        label="Inner race defect",
        confidence=0.8123,
        prediction_set=["inner_race", "normal"],
        prediction_set_labels=["Inner race defect", "Healthy bearing"],
        coverage=0.9,
        uncertain=uncertain,
        rpm_used=1200.0,
        rpm_assumed=True,
        segment_agreement=0.88,
        segments=39,
        duration_s=20.0,
        probabilities=[
            FaultProbability(class_name="inner_race", label="Inner race defect", probability=0.8123),
            FaultProbability(class_name="normal", label="Healthy bearing", probability=0.1877),
        ],
    )
    return ModalityDraft(
        kind="bearing",
        availability="available",
        reason_code="ok",
        source_mode="uploaded",
        observed_at=observed,
        inference_at=NOW + timedelta(seconds=2),
        model_identity=BEARING_IDENTITY,
        recording_id=str(uuid4()),
        recording_sha256="a" * 64,
        reference_id="sha256:" + "1" * 64,
        limitations=["lab rig with artificial defects; real machines differ"],
        output=output,
    )


BUILDERS = {"sensor": sensor_draft, "acoustic": acoustic_draft, "bearing": bearing_draft}


def make_snapshot(bits=("sensor",), *, observed: datetime | None = NOW, source_mode: str = "uploaded", **kw):
    """Deterministic assessment snapshot through the real assess/v1 service."""
    modalities = [BUILDERS[k](**kw.get(k, {})) for k in bits]
    for m in modalities:
        m.observed_at = observed
    evidence: list[EvidenceDraft] = []
    if "sensor" in bits:
        evidence.append(EvidenceDraft(kind="failure_probability", value=0.8237, unit="probability",
                                      source="sensor", source_mode="simulated", observed_at=observed))
    if "acoustic" in bits:
        evidence.append(EvidenceDraft(kind="anomaly_score", value=-0.4181, unit="score",
                                      source="acoustic", source_mode="uploaded", observed_at=observed))
    if "bearing" in bits:
        evidence.append(EvidenceDraft(kind="class_confidence", value=0.8123, unit="probability",
                                      source="bearing", source_mode="uploaded", observed_at=observed))
    return create_assessment(
        machine_id=MACHINE,
        source_mode=source_mode,
        expected_modalities=list(bits),
        modalities=modalities,
        evidence=evidence,
        extra_flags=[],
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


def seed(client, bits=("sensor",), **kw) -> dict:
    """Insert a snapshot under alice and return its id."""
    snapshot = make_snapshot(bits, **kw)
    client.app.state.telltale.assessments.insert(snapshot, owner="alice")
    return snapshot


def explain(client, assessment_id, auth=AUTH_ALICE, **over):
    body = {"assessment_id": str(assessment_id)}
    body.update(over)
    return client.post(URL_EXPLAIN, json=body, headers=auth)


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


# ---------------------------------------------------------------- happy paths


def test_valid_explanation_201_get_parity(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        aid = body["ai_output_id"]
        assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", aid)
        assert body["assessment_id"] == str(snapshot.assessment_id)
        assert body["kind"] == "explain"
        assert body["validation_status"] == "validated"
        assert body["provider"] == "template"
        assert body["prompt_version"] == "explain/v1"
        assert body["referenced_assessment_ids"] == [str(snapshot.assessment_id)]
        assert resp.headers["Location"] == f"/api/v1/ai/outputs/{aid}"
        text = body["explanation_text"]
        assert "model indications, not confirmed faults" in text
        assert "missing" in text.lower()
        assert "not machine-health completeness" in text
        # GET parity: the stored record matches what was returned
        got = client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE)
        assert got.status_code == 200, got.text
        record = got.json()
        assert record["validation_status"] == "validated"
        assert record["response_text"] == text
        assert record["output"]["explanation_text"] == text
        assert record["assessment_ids"] == [str(snapshot.assessment_id)]
        assert re.fullmatch(r"[0-9a-f]{64}", record["prompt_sha256"])
        assert ai_rows(tmp_path) == 1


@pytest.mark.parametrize(
    "bits",
    [("sensor",), ("acoustic",), ("bearing",), ("sensor", "acoustic"), ("sensor", "bearing"),
     ("acoustic", "bearing"), ("sensor", "acoustic", "bearing")],
)
def test_valid_explanation_for_every_snapshot_shape(client_factory, bits):
    with client_factory() as client:
        snapshot = seed(client, bits)
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "validated"
        record = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE).json()
        output = record["output"]
        assert set(output["expected_modalities"]) == set(bits)
        assert sorted(output["present_modalities"]) == sorted(bits)
        assert output["overall_status"] == snapshot.overall_status
        assert output["referenced_evidence_ids"], "available modalities must cite their stored evidence"


def test_multimodal_content_quotes_stored_facts(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        aid = resp.json()["ai_output_id"]
        record = client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE).json()
        output = record["output"]
        text = output["explanation_text"]
        # every stored number the agent quotes is the STORED value (no rounding/invention)
        assert "0.8237" in text  # sensor failure_probability (stored)
        assert "0.4569" in text  # decision threshold (stored)
        assert "-0.4181" in text  # acoustic anomaly score (stored)
        assert "1.3392" in text  # acoustic threshold (stored)
        assert "0.8123" in text  # bearing confidence (stored)
        assert '"high"' in text  # stored sensor risk level, quoted
        assert '"normal"' in text  # stored acoustic status, quoted
        assert '"inner_race"' in text  # stored predicted class, quoted
        assert '"simulated"' in text and '"uploaded"' in text  # per-modality provenance
        assert "synthetic dataset caveat" in text  # limitation preserved
        assert "lab rig with artificial defects" in text  # limitation preserved
        assert "sensor-lightgbm" in text and "mimii_due" in text and "mendeley_n9y9c7xrz3" in text
        notes = output["uncertainty_notes"]
        assert any("illustrative demo/simulated" in n for n in notes)
        assert any("hash only" in n or "conformal" in n for n in notes)
        assert output["referenced_evidence_ids"]
        for evidence_id in output["referenced_evidence_ids"]:
            assert evidence_id in {str(e.evidence_id) for e in snapshot.evidence}


# ---------------------------------------------------------------- uncertainty handling


def test_missing_timestamp_uncertainty_and_flags(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "bearing"), observed=None, bearing={"uncertain": True})
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201
        aid = resp.json()["ai_output_id"]
        record = client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE).json()
        output = record["output"]
        assert "TIMESTAMP_MISSING" in output["quality_flag_codes"]
        assert "BEARING_UNCERTAIN" in output["quality_flag_codes"]
        assert "missing measurement timestamp" in output["explanation_text"]
        notes = " ".join(output["uncertainty_notes"]).lower()
        assert "timestamp is missing" in notes
        assert "conservative uncertainty flags" in notes
        assert "review priority" in notes


def test_conservative_status_meaning_is_stated(client_factory):
    with client_factory() as client:
        watch = seed(client, ("sensor",), sensor={"risk": "medium", "probability": 0.5})
        resp = explain(client, watch.assessment_id)
        assert resp.status_code == 201
        record = client.get(f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE).json()
        output = record["output"]
        assert output["overall_status"] == "watch"
        assert '"watch"' in output["explanation_text"]
        assert "conservative review priorities, not confirmed faults" in output["explanation_text"]


def test_insufficient_evidence_refusal(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = make_snapshot(("sensor",))
        dropped = snapshot.model_copy(deep=True)
        dropped.modalities[0] = snapshot.modalities[0].model_copy(
            update={"availability": "unavailable", "reason_code": "not_submitted", "output": None}
        )
        client.app.state.telltale.assessments.insert(dropped, owner="alice")
        resp = explain(client, dropped.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "refused"
        assert "no further interpretation is offered" in body["explanation_text"]
        record = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE).json()
        assert record["validation_status"] == "refused"
        assert ai_rows(tmp_path) == 1


# ---------------------------------------------------------------- strict requests + auth


def test_invalid_request_schemas(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client)
        aid = str(snapshot.assessment_id)
        assert_error(explain(client, aid, style="short"), 422, "request_validation")
        assert_error(explain(client, aid, owner="bob"), 422, "request_validation")
        assert_error(explain(client, aid, source_mode="live"), 422, "request_validation")
        resp = client.post(URL_EXPLAIN, json={"assessment": aid}, headers=AUTH_ALICE)
        assert_error(resp, 422, "request_validation")  # wrong field name
        bad_body = client.post(
            URL_EXPLAIN,
            data=b"{not json",
            headers={**AUTH_ALICE, "Content-Type": "application/json"},
        )
        assert_error(bad_body, 422, "request_validation")
        body = explain(client, "not-a-uuid")
        assert_error(body, 422, "request_validation")
        assert ai_rows(tmp_path) == 0
        # schema strictness is also unit-verifiable
        with pytest.raises(Exception):
            AiExplainRequest.model_validate({"assessment_id": aid, "style": "x"})


def test_auth_required_before_processing(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client)
        assert_error(explain(client, snapshot.assessment_id, auth=None), 401, "unauthorized")
        bad_token = {"Authorization": "Bearer nope"}
        assert_error(
            explain(client, snapshot.assessment_id, auth=bad_token), 401, "unauthorized"
        )
        assert_error(
            client.get(f"/api/v1/ai/outputs/{uuid4()}", headers=None), 401, "unauthorized"
        )
        assert ai_rows(tmp_path) == 0


def test_owner_isolation_and_no_existence_leak(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        # unknown id and foreign id share one response shape
        unknown = uuid4()
        a = explain(client, snapshot.assessment_id, auth=AUTH_BOB)
        b = explain(client, unknown, auth=AUTH_BOB)
        assert_error(a, 404, "not_found")
        assert_error(b, 404, "not_found")
        assert a.json()["error"]["message"] == b.json()["error"]["message"]
        assert ai_rows(tmp_path) == 0
        # actor A creates; actor B cannot read the output either
        resp = explain(client, snapshot.assessment_id)
        aid = resp.json()["ai_output_id"]
        assert_error(client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_BOB), 404, "not_found")
        assert client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE).status_code == 200


def test_get_output_unknown_and_invalid_ids(client_factory):
    with client_factory() as client:
        get_uuid = client.get(f"/api/v1/ai/outputs/{uuid4()}", headers=AUTH_ALICE)
        assert_error(get_uuid, 404, "not_found")
        bad_id = client.get("/api/v1/ai/outputs/not-a-uuid", headers=AUTH_ALICE)
        assert_error(bad_id, 422, "request_validation")


# ---------------------------------------------------------------- grounding validation (unit)


def test_grounding_rejects_invented_facts(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        output = TemplateExplainProvider().explain(snapshot)
        ok, _ = validate_explain_output(output, snapshot)
        assert ok, "the template output must ground against its own snapshot"

        other = "critical" if output.overall_status != "critical" else "normal"
        bad_status = output.model_copy(update={"overall_status": other})
        assert not validate_explain_output(bad_status, snapshot)[0], "invented overall status must fail"

        bad_ref = output.model_copy(update={"referenced_assessment_ids": [str(uuid4())]})
        assert not validate_explain_output(bad_ref, snapshot)[0], "invented assessment id must fail"

        bad_evidence = output.model_copy(update={"referenced_evidence_ids": [str(uuid4())]})
        assert not validate_explain_output(bad_evidence, snapshot)[0], "invented evidence id must fail"

        bad_number = output.model_copy(
            update={"explanation_text": output.explanation_text + " confidence 0.99999 in the model"}
        )
        assert not validate_explain_output(bad_number, snapshot)[0], "invented numeric value must fail"

        bad_quote = output.model_copy(
            update={"explanation_text": output.explanation_text + ' The verdict is "confirmed".'}
        )
        assert not validate_explain_output(bad_quote, snapshot)[0], "invented quoted status must fail"

        bad_reason = output.model_copy(update={"status_reasons": ["sensor:invented_reason"]})
        assert not validate_explain_output(bad_reason, snapshot)[0], "invented status reason must fail"

        bad_expected = output.model_copy(update={"expected_modalities": ["sensor"]})
        assert not validate_explain_output(bad_expected, snapshot)[0], "wrong requested set must fail"

        bad_machine = output.model_copy(update={"machine_id": "other"})
        assert not validate_explain_output(bad_machine, snapshot)[0], "wrong machine must fail"


def test_route_falls_back_when_grounding_fails(client_factory, tmp_path, monkeypatch):
    with client_factory() as client:
        snapshot = seed(client, ("sensor",))
        calls = {"n": 0}
        original = ai_routes.validate_explain_output

        def flaky(output, assessment):
            calls["n"] += 1
            return (True, "") if calls["n"] > 1 else (False, "forced grounding failure")

        monkeypatch.setattr(ai_routes, "validate_explain_output", flaky)
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "fallback"
        assert "no additional interpretation is offered" in body["explanation_text"]
        record = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE).json()
        assert record["validation_status"] == "fallback"
        assert ai_rows(tmp_path) == 1
        monkeypatch.setattr(ai_routes, "validate_explain_output", original)


def test_route_refuses_when_provider_fails(client_factory, tmp_path, monkeypatch):
    with client_factory() as client:
        snapshot = seed(client, ("sensor",))

        def boom(self, assessment):
            raise RuntimeError("template exploded")

        monkeypatch.setattr(TemplateExplainProvider, "explain", boom)
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        # generation failure -> deterministic FALLBACK output (contract: fallback/refusal)
        assert body["validation_status"] == "fallback"
        assert "not confirmed faults" in body["explanation_text"]
        assert "no additional interpretation is offered" in body["explanation_text"]
        assert ai_rows(tmp_path) == 1


# ---------------------------------------------------------------- persistence + immutability


def test_snapshot_immutability_after_ai_output(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        before = client.get(f"{URL_BASE}/{snapshot.assessment_id}", headers=AUTH_ALICE)
        assert before.status_code == 200
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201
        after = client.get(f"{URL_BASE}/{snapshot.assessment_id}", headers=AUTH_ALICE)
        assert after.status_code == 200
        assert before.content == after.content  # byte-identical stored document
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0] == 1


def test_restart_persistence(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201
        aid = resp.json()["ai_output_id"]
        first = client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE).json()
    with client_factory(data_dir=tmp_path) as client2:
        got = client2.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE)
        assert got.status_code == 200
        assert got.json() == first  # byte-level content survives restart
        assert client2.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_BOB).status_code == 404


def test_prompt_digest_is_deterministic(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor",))
        r1 = explain(client, snapshot.assessment_id)
        r2 = explain(client, snapshot.assessment_id)
        assert r1.status_code == 201 and r2.status_code == 201
        rec1 = client.get(f"/api/v1/ai/outputs/{r1.json()['ai_output_id']}", headers=AUTH_ALICE).json()
        rec2 = client.get(f"/api/v1/ai/outputs/{r2.json()['ai_output_id']}", headers=AUTH_ALICE).json()
        assert rec1["prompt_sha256"] == rec2["prompt_sha256"]
        assert rec1["ai_output_id"] != rec2["ai_output_id"]
        assert rec1["created_at"] != rec2["created_at"]
        assert ai_rows(tmp_path) == 2


# ---------------------------------------------------------------- failure behavior


def test_store_failures_produce_nothing(client_factory, tmp_path, monkeypatch):
    with client_factory() as client:
        snapshot = seed(client)
        ai_store = client.app.state.telltale.ai_outputs
        assert ai_store is not None

        def boom(*args, **kwargs):
            raise RuntimeError("insert failed at C:\\secret\\telltale.db")

        monkeypatch.setattr(ai_store, "insert", boom)
        body = assert_error(explain(client, snapshot.assessment_id), 500, "internal_error")
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert "secret" not in body["error"]["message"]
        assert ai_rows(tmp_path) == 0
        monkeypatch.undo()
        # assessment storage unavailable -> 503, nothing saved
        client.app.state.telltale.assessments = None
        assert_error(explain(client, snapshot.assessment_id), 503, "storage_unavailable")
        assert ai_rows(tmp_path) == 0


def test_ai_storage_unavailable(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client)
        client.app.state.telltale.ai_outputs = None
        assert_error(explain(client, snapshot.assessment_id), 503, "storage_unavailable")
        assert ai_rows(tmp_path) == 0


# ---------------------------------------------------------------- secrets/paths hygiene


def test_no_secrets_or_paths_in_stored_artifacts(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = explain(client, snapshot.assessment_id)
        assert resp.status_code == 201
        aid = resp.json()["ai_output_id"]
        record = client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE).json()
        blob = json.dumps(record)
        for forbidden in ("C:\\", "telltale.db", "models/sensor", "models/acoustic", "models/bearing",
                          "tok-alice", "tok-bob", "schema_migrations", "SELECT ", "INSERT "):
            assert forbidden not in blob, forbidden
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            raw = conn.execute("SELECT json FROM ai_outputs").fetchone()[0]
        for forbidden in ("C:\\", "telltale.db", "models/", "tok-alice", "tok-bob"):
            assert forbidden not in raw, forbidden


# ---------------------------------------------------------------- migration + legacy routes


def test_ai_outputs_table_created_by_versioned_migration(client_factory, tmp_path):
    with client_factory() as client:
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            versions = dict(conn.execute("SELECT version, name FROM schema_migrations").fetchall())
            assert versions[3] == "ai_outputs_v3"  # new table only, versions 1/2 untouched
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert {"assessments", "ai_outputs"} <= tables


def test_legacy_assessment_routes_unchanged(client_factory):
    with client_factory() as client:
        # legacy sensor route (JSON body)
        resp = client.post(
            URL_BASE,
            json={"reading": READING, "machine_id": MACHINE, "source_mode": "simulated"},
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["sensor"]
        # legacy acoustic route (multipart, real fixture fan model)
        import io

        import numpy as np
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, (0.05 * np.ones(32000)).astype(np.float32), 16000, format="WAV")
        resp = client.post(
            URL_BASE + "/acoustic",
            data={"machine_id": MACHINE},
            files=[("file", ("clip.wav", buf.getvalue(), "audio/wav"))],
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["acoustic"]
        # the new AI slice then explains one of the legacy snapshots
        legacy_id = client.get(URL_BASE, headers=AUTH_ALICE).json()["items"][0]["assessment_id"]
        resp = explain(client, legacy_id)
        assert resp.status_code == 201, resp.text
        assert resp.json()["validation_status"] in {"validated", "refused"}


def test_explanation_output_is_schema_strict(client_factory):
    with client_factory() as client:
        snapshot = seed(client)
        output = TemplateExplainProvider().explain(snapshot)
        payload = json.loads(output.model_dump_json())
        payload["forged"] = "overall_status"
        with pytest.raises(Exception):
            AiExplainOutput.model_validate(payload)
        # a truncated/oversized text is rejected by the schema, not silently stored
        payload2 = json.loads(output.model_dump_json())
        payload2["explanation_text"] = "x"
        with pytest.raises(Exception):
            AiExplainOutput.model_validate(payload2)  # below the minimum length


# SENTINEL-APPEND-POINT
