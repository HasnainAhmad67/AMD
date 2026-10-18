"""Central AI Analysis Agent - slice B tests: POST /api/v1/ai/evidence-review.

Covers the approved slice-B contract: strict request boundary, owner resolution
before evidence access, read-only use of immutable snapshots, deterministic
template review (structured fields exactly stored facts), uncertainty explanations
for every stored flag path, independent-style grounding rejections (invented
ids/statuses/numbers), refusal/fallback behavior, append-only persistence with the
existing ai_outputs store (kind="evidence_review", no new migration), GET parity via
the shared owner-scoped output route, restart persistence, immutability, and unchanged
legacy assessment + /ai/explain routes.

No downloads, no GPUs, no network, no provider: snapshots are built through the real
assess/v1 service and the conftest fixture models are used only by the legacy-route
regression checks.
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
from telltale_api.schemas.acoustic import (
    AcousticAnalysis,
    BandDeviation,
    FaultDiagnosis,
    FaultProbability,
    Spectrogram,
    Series,
    Waveform,
)
from telltale_api.schemas.assessment import EvidenceDraft, ModalityDraft, ModelIdentity
from telltale_api.schemas.ai import (
    AiEvidenceReviewOutput,
    AiEvidenceReviewRequest,
    AiFlagEntry,
)
from telltale_api.schemas.sensor import Contribution, FailureModeScore, SensorAssessment
from telltale_api.services.ai_review_template import TemplateEvidenceReviewProvider
from telltale_api.services.ai_review_validate import validate_review_output
from telltale_api.services.assess import create_assessment

URL_REVIEW = "/api/v1/ai/evidence-review"
URL_EXPLAIN = "/api/v1/ai/explain"
URL_BASE = "/api/v1/assessments"

TOKENS = json.dumps({"alice": "tok-alice-9f3d2c7e", "bob": "tok-bob-77aa11bb"})
AUTH_ALICE = {"Authorization": "Bearer tok-alice-9f3d2c7e"}
AUTH_BOB = {"Authorization": "Bearer tok-bob-77aa11bb"}

MACHINE = "cnc-a1"
NOW = datetime(2026, 10, 16, 0, 30, 0, tzinfo=timezone.utc)
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


def sensor_draft(*, risk: str = "high", probability: float = 0.8237, observed: datetime | None = NOW,
                 identity: ModelIdentity | None = None) -> ModalityDraft:
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
        kind="sensor",
        availability="available",
        reason_code="ok",
        source_mode="simulated",
        observed_at=observed,
        inference_at=NOW,
        model_identity=identity or SENSOR_IDENTITY,
        limitations=["synthetic dataset caveat", "SHAP explains the model, not physics"],
        output=output,
    )


def acoustic_draft(*, status: str = "normal", observed: datetime | None = NOW,
                   limitations: list[str] | None = None) -> ModalityDraft:
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
        limitations=limitations or ["trained on normal sounds only; flags difference, not a specific fault"],
        output=output,
    )


def bearing_draft(*, uncertain: bool = False, observed: datetime | None = NOW) -> ModalityDraft:
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


BUILDERS = {
    "sensor": lambda **kw: sensor_draft(**kw),
    "acoustic": lambda **kw: acoustic_draft(**kw),
    "bearing": lambda **kw: bearing_draft(**kw),
}


def make_snapshot(bits=("sensor",), *, observed: datetime | None = NOW, source_mode: str = "uploaded",
                  drop=None, **per_kind):
    """Deterministic assessment snapshot through the real assess/v1 service.

    ``drop`` marks a requested kind unavailable (availability 'unavailable'), which is
    the only way a snapshot can carry a MISSING modality under the immutable schema.
    ``per_kind`` forwards builder kwargs (e.g. ``sensor={"risk": "medium"}``).
    """
    modalities = []
    for k in bits:
        if drop == k:
            modalities.append(
                ModalityDraft(
                    kind=k, availability="unavailable", reason_code="not_submitted",
                    source_mode="simulated" if k == "sensor" else "uploaded",
                    observed_at=observed, inference_at=NOW, model_identity=None,
                    limitations=[], output=None,
                )
            )
        else:
            modalities.append(BUILDERS[k](observed=observed, **per_kind.get(k, {})))
    evidence: list[EvidenceDraft] = []
    for k in bits:
        if drop != k:
            evidence.append(
                EvidenceDraft(
                    kind={"sensor": "failure_probability", "acoustic": "anomaly_score",
                          "bearing": "class_confidence"}[k],
                    value={"sensor": 0.8237, "acoustic": -0.4181, "bearing": 0.8123}[k],
                    unit="probability" if k != "acoustic" else "score",
                    source=k,
                    source_mode="simulated" if k == "sensor" else "uploaded",
                    observed_at=observed,
                )
            )
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


def seed(client, bits=("sensor",), **kw):
    snapshot = make_snapshot(bits, **kw)
    client.app.state.telltale.assessments.insert(snapshot, owner="alice")
    return snapshot


def review(client, assessment_id, auth=AUTH_ALICE, **over):
    body = {"assessment_id": str(assessment_id)}
    body.update(over)
    return client.post(URL_REVIEW, json=body, headers=auth)


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


def test_valid_review_201_get_parity(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        aid = body["ai_output_id"]
        assert body["assessment_id"] == str(snapshot.assessment_id)
        assert body["kind"] == "evidence_review"
        assert body["feature"] == "evidence_review"
        assert body["validation_status"] == "validated"
        assert body["provider"] == "template"
        assert body["prompt_version"] == "evidence-review/v1"
        assert body["referenced_assessment_ids"] == [str(snapshot.assessment_id)]
        assert resp.headers["Location"] == f"/api/v1/ai/outputs/{aid}"
        got = client.get(f"/api/v1/ai/outputs/{aid}", headers=AUTH_ALICE)
        assert got.status_code == 200, got.text
        record = got.json()
        assert record["feature"] == "evidence_review"
        assert record["kind"] == "evidence_review"
        assert record["response_text"] == body["review_text"]
        assert record["output"]["review_text"] == body["review_text"]
        assert record["assessment_ids"] == [str(snapshot.assessment_id)]
        assert re.fullmatch(r"[0-9a-f]{64}", record["prompt_sha256"])
        assert ai_rows(tmp_path) == 1


@pytest.mark.parametrize(
    "bits",
    [("sensor",), ("acoustic",), ("bearing",), ("sensor", "acoustic"), ("sensor", "bearing"),
     ("acoustic", "bearing"), ("sensor", "acoustic", "bearing")],
)
def test_valid_review_for_each_snapshot_shape(client_factory, bits):
    with client_factory() as client:
        snapshot = seed(client, bits)
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "validated"
        record = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE).json()
        output = record["output"]
        assert set(output["expected_modalities"]) == set(bits)
        assert sorted(output["present_modalities"]) == sorted(bits)
        assert output["missing_modalities"] == []
        assert output["overall_status"] == snapshot.overall_status
        assert output["referenced_evidence_ids"]


def test_structured_fields_exactly_match_stored_snapshot(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201
        record = client.get(
            f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        output = record["output"]
        by_kind = {m.kind: m for m in snapshot.modalities}
        assert output["machine_id"] == MACHINE
        assert output["overall_status"] == snapshot.overall_status
        assert output["status_reasons"] == snapshot.status_reasons
        assert output["expected_modalities"] == snapshot.expected_modalities
        rows = {r["kind"]: r for r in output["modality_reviews"]}
        assert set(rows) == set(by_kind)
        for kind, stored in by_kind.items():
            row = rows[kind]
            assert row["availability"] == stored.availability
            assert row["reason_code"] == stored.reason_code
            assert row["source_mode"] == stored.source_mode
            assert row["observed_at"] == (
                stored.observed_at.isoformat().replace("+00:00", "Z") if stored.observed_at else None
            )
            assert row["inference_at"] == (
                stored.inference_at.isoformat().replace("+00:00", "Z") if stored.inference_at else None
            )
            assert row["model_name"] == (stored.model_identity.name if stored.model_identity else None)
            assert row["model_version"] == (stored.model_identity.version if stored.model_identity else None)
            assert row["training_dataset"] == (
                stored.model_identity.training_dataset if stored.model_identity else None
            )
            assert row["limitations"] == list(stored.limitations)
            stored_kinds = sorted(e.kind for e in snapshot.evidence if e.source == kind)
            assert sorted(row["evidence_kinds"]) == stored_kinds
        stored_flag_codes = {f.code for f in snapshot.quality_flags}
        assert {f["code"] for f in output["quality_flags"]} == stored_flag_codes
        severities = {f["code"]: f["severity"] for f in output["quality_flags"]}
        stored_severities = {f.code: f.severity for f in snapshot.quality_flags}
        assert severities == stored_severities
        assert output["referenced_evidence_ids"] == [str(e.evidence_id) for e in snapshot.evidence]
        text = output["review_text"]
        assert "not a fault diagnosis" in text
        assert "not machine-health completeness" in text
        assert "not coverage" in text
        assert output["non_claims"]


def test_missing_modalities_derived_from_expected_minus_present(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"), drop="acoustic")
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201
        record = client.get(
            f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        output = record["output"]
        assert output["expected_modalities"] == ["sensor", "acoustic"]
        assert output["present_modalities"] == ["sensor"]
        assert output["missing_modalities"] == ["acoustic"]
        assert "missing modalities: acoustic" in output["completeness_note"]
        assert "EVIDENCE_INCOMPLETE" in {f["code"] for f in output["quality_flags"]}
        row = {r["kind"]: r for r in output["modality_reviews"]}["acoustic"]
        assert row["availability"] == "unavailable" and row["reason_code"] == "not_submitted"
        assert row["evidence_kinds"] == []
        assert "no model output exists for this modality" in output["review_text"]


# ---------------------------------------------------------------- uncertainty explanation paths


UNCERTAINTY_CASES = [
    ("status-watch", ("sensor",), {"sensor": {"risk": "medium", "probability": 0.5}},
     "conservative review priority, not a confirmed fault"),
    ("status-unknown-bearing", ("bearing",), {"bearing": {"uncertain": True}},
     "ambiguous, empty or inconsistent"),
    ("status-anomalous-acoustic", ("acoustic",), {"acoustic": {"status": "anomalous"}},
     "conservative anomaly indication, not a confirmed fault"),
    ("flag-bearing-uncertain", ("sensor", "bearing"), {"bearing": {"uncertain": True}},
     "BEARING_UNCERTAIN"),
    ("flag-identity-incomplete", ("sensor",), {},
     "MODEL_IDENTITY_INCOMPLETE"),
    ("flag-limitations-present", ("sensor", "acoustic", "bearing"), {},
     "MODEL_LIMITATIONS_PRESENT"),
    ("flag-timestamp-missing", ("sensor",), {"observed": None},
     "TIMESTAMP_MISSING"),
    ("flag-source-illustrative", ("sensor",), {},
     "SOURCE_ILLUSTRATIVE"),
    ("flag-mixed-provenance", ("sensor", "acoustic"), {},
     "MIXED_SOURCE_PROVENANCE"),
    ("flag-evidence-incomplete", ("sensor", "acoustic"), {"drop": "acoustic"},
     "EVIDENCE_INCOMPLETE"),
]


@pytest.mark.parametrize("name,bits,kw,needle", UNCERTAINTY_CASES)
def test_uncertainty_paths(client_factory, name, bits, kw, needle):
    with client_factory() as client:
        snapshot = seed(client, bits, **kw)
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        record = client.get(
            f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        output = record["output"]
        assert needle in output["review_text"] or needle in " ".join(output["uncertainty_indicators"]), name
        assert "not a fault diagnosis" in output["review_text"]
        assert "not machine-health completeness" in output["review_text"]
        assert "not coverage" in output["review_text"]
        # uncertainty is never phrased as a fault claim (only negations may mention faults)
        lowered = output["review_text"].lower()
        for banned in ("is a confirmed fault", "confirmed fault diagnosis", "is faulty",
                       "definitely", "guaranteed", "certainly broken", "machine is broken"):
            assert banned not in lowered, (name, banned)


def test_missing_timestamp_and_provenance_summary(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"), observed=None)
        resp = review(client, snapshot.assessment_id)
        record = client.get(
            f"/api/v1/ai/outputs/{resp.json()['ai_output_id']}", headers=AUTH_ALICE
        ).json()
        output = record["output"]
        assert "TIMESTAMP_MISSING" in {f["code"] for f in output["quality_flags"]}
        assert "missing measurement timestamp" in output["review_text"]
        for row in output["modality_reviews"]:
            assert row["observed_at"] is None
        assert 'Assessment-level source mode is "uploaded"' in output["provenance_summary"]
        assert 'sensor "simulated"' in output["provenance_summary"]
        assert "Mixed provenance flag: stored" in output["provenance_summary"]


# ---------------------------------------------------------------- auth + strict requests


def test_owner_isolation_and_no_existence_leak(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        unknown = uuid4()
        a = review(client, snapshot.assessment_id, auth=AUTH_BOB)
        b = review(client, unknown, auth=AUTH_BOB)
        assert_error(a, 404, "not_found")
        assert_error(b, 404, "not_found")
        assert a.json()["error"]["message"] == b.json()["error"]["message"]
        assert ai_rows(tmp_path) == 0
        resp = review(client, snapshot.assessment_id)
        oid = resp.json()["ai_output_id"]
        assert client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_BOB).status_code == 404
        assert client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE).status_code == 200


def test_invalid_requests(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client)
        aid = str(snapshot.assessment_id)
        assert_error(review(client, aid, style="short"), 422, "request_validation")
        assert_error(review(client, aid, machine_id=MACHINE), 422, "request_validation")
        assert_error(review(client, "not-a-uuid"), 422, "request_validation")
        wrong = client.post(URL_REVIEW, json={"assessment": aid}, headers=AUTH_ALICE)
        assert_error(wrong, 422, "request_validation")
        malformed = client.post(
            URL_REVIEW, data=b"{not json",
            headers={**AUTH_ALICE, "Content-Type": "application/json"},
        )
        assert_error(malformed, 422, "request_validation")
        oversized = client.post(URL_REVIEW, data=b"x" * 70000, headers=AUTH_ALICE)
        assert_error(oversized, 422, "request_validation")
        assert_error(review(client, aid, auth=None), 401, "unauthorized")
        assert ai_rows(tmp_path) == 0
        with pytest.raises(Exception):
            AiEvidenceReviewRequest.model_validate({"assessment_id": aid, "style": "x"})


def test_missing_assessment_404(client_factory, tmp_path):
    with client_factory() as client:
        assert_error(review(client, uuid4()), 404, "not_found")
        assert ai_rows(tmp_path) == 0


# ---------------------------------------------------------------- grounding validation (unit)


def test_grounding_rejects_invented_facts(client_factory):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        output = TemplateEvidenceReviewProvider().review(snapshot)
        ok, _ = validate_review_output(output, snapshot)
        assert ok, "the template review must ground against its own snapshot"

        bad_ref = output.model_copy(update={"referenced_assessment_ids": [str(uuid4())]})
        assert not validate_review_output(bad_ref, snapshot)[0], "invented assessment id must fail"

        bad_evidence = output.model_copy(update={"referenced_evidence_ids": [str(uuid4())]})
        assert not validate_review_output(bad_evidence, snapshot)[0], "invented evidence id must fail"

        bad_status = output.model_copy(
            update={"overall_status": "normal" if output.overall_status != "normal" else "critical"}
        )
        assert not validate_review_output(bad_status, snapshot)[0], "invented overall status must fail"

        bad_number = output.model_copy(
            update={"review_text": output.review_text + " confidence 0.99999 in the model"}
        )
        assert not validate_review_output(bad_number, snapshot)[0], "invented numeric value must fail"

        bad_quote = output.model_copy(
            update={"review_text": output.review_text + ' The verdict is "confirmed".'}
        )
        assert not validate_review_output(bad_quote, snapshot)[0], "invented quoted status must fail"

        bad_missing = output.model_copy(update={"missing_modalities": ["bearing"]})
        assert not validate_review_output(bad_missing, snapshot)[0], "invented missing modality must fail"

        bad_present = output.model_copy(update={"present_modalities": []})
        assert not validate_review_output(bad_present, snapshot)[0], "wrong present set must fail"

        bad_flags = output.model_copy(
            update={"quality_flags": [AiFlagEntry(code="TIMESTAMP_MISSING", severity="info")]}
        )
        assert not validate_review_output(bad_flags, snapshot)[0], "wrong stored flags must fail"

        bad_phrase = output.model_copy(
            update={"review_text": output.review_text.replace("not a fault diagnosis", "a fault diagnosis")}
        )
        assert not validate_review_output(bad_phrase, snapshot)[0], "missing non-claim phrase must fail"

        bad_feature = output.model_copy(update={"feature": "explain"})
        assert not validate_review_output(bad_feature, snapshot)[0], "wrong feature must fail"


def test_route_falls_back_when_grounding_fails(client_factory, tmp_path, monkeypatch):
    with client_factory() as client:
        snapshot = seed(client, ("sensor",))
        calls = {"n": 0}
        original = ai_routes.validate_review_output

        def flaky(output, assessment):
            calls["n"] += 1
            return (True, "") if calls["n"] > 1 else (False, "forced grounding failure")

        monkeypatch.setattr(ai_routes, "validate_review_output", flaky)
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "fallback"
        assert "no interpretation is offered" in body["review_text"]
        record = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE).json()
        assert record["validation_status"] == "fallback"
        assert ai_rows(tmp_path) == 1
        monkeypatch.setattr(ai_routes, "validate_review_output", original)


def test_route_refuses_when_provider_fails(client_factory, tmp_path, monkeypatch):
    with client_factory() as client:
        snapshot = seed(client, ("sensor",))

        def boom(self, assessment):
            raise RuntimeError("review exploded")

        monkeypatch.setattr(TemplateEvidenceReviewProvider, "review", boom)
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "fallback"
        assert "not coverage" in body["review_text"]
        assert ai_rows(tmp_path) == 1


def test_refusal_path(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = make_snapshot(("sensor",))
        dropped = snapshot.model_copy(deep=True)
        dropped.modalities[0] = snapshot.modalities[0].model_copy(
            update={"availability": "unavailable", "reason_code": "not_submitted", "output": None}
        )
        client.app.state.telltale.assessments.insert(dropped, owner="alice")
        resp = review(client, dropped.assessment_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["validation_status"] == "refused"
        assert "no further interpretation is offered" in body["review_text"]
        record = client.get(f"/api/v1/ai/outputs/{body['ai_output_id']}", headers=AUTH_ALICE).json()
        assert record["validation_status"] == "refused"
        assert record["output"]["missing_modalities"] == ["sensor"]
        assert ai_rows(tmp_path) == 1


# ---------------------------------------------------------------- persistence + immutability


def test_snapshot_immutability_after_review(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        before = client.get(f"{URL_BASE}/{snapshot.assessment_id}", headers=AUTH_ALICE)
        assert before.status_code == 200
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201
        after = client.get(f"{URL_BASE}/{snapshot.assessment_id}", headers=AUTH_ALICE)
        assert after.status_code == 200
        assert before.content == after.content
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0] == 1


def test_restart_persistence(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201
        oid = resp.json()["ai_output_id"]
        first = client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE).json()
    with client_factory(data_dir=tmp_path) as client2:
        got = client2.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE)
        assert got.status_code == 200
        assert got.json() == first
        assert client2.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_BOB).status_code == 404


def test_prompt_digest_is_deterministic(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        r1 = review(client, snapshot.assessment_id)
        r2 = review(client, snapshot.assessment_id)
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
        body = assert_error(review(client, snapshot.assessment_id), 500, "internal_error")
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert "secret" not in body["error"]["message"]
        assert ai_rows(tmp_path) == 0
        monkeypatch.undo()
        client.app.state.telltale.assessments = None
        assert_error(review(client, snapshot.assessment_id), 503, "storage_unavailable")
        assert ai_rows(tmp_path) == 0


def test_ai_storage_unavailable(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client)
        client.app.state.telltale.ai_outputs = None
        assert_error(review(client, snapshot.assessment_id), 503, "storage_unavailable")
        assert ai_rows(tmp_path) == 0


# ---------------------------------------------------------------- secrets/paths + migrations


def test_no_secrets_or_paths_in_stored_review(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic", "bearing"))
        resp = review(client, snapshot.assessment_id)
        assert resp.status_code == 201
        oid = resp.json()["ai_output_id"]
        record = client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE).json()
        blob = json.dumps(record)
        for forbidden in ("C:\\", "telltale.db", "models/sensor", "models/acoustic", "models/bearing",
                          "tok-alice", "tok-bob", "schema_migrations", "SELECT ", "INSERT "):
            assert forbidden not in blob, forbidden
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            raw = conn.execute("SELECT json FROM ai_outputs").fetchone()[0]
        for forbidden in ("C:\\", "telltale.db", "models/", "tok-alice", "tok-bob"):
            assert forbidden not in raw, forbidden


def test_no_new_migration_and_table_reuse(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client)
        review(client, snapshot.assessment_id)
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            versions = dict(conn.execute("SELECT version, name FROM schema_migrations").fetchall())
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert versions == {1: "assessments_v1", 2: "assessments_owner_v2", 3: "ai_outputs_v3"}
        assert {"assessments", "ai_outputs"} <= tables
        kinds = [r[0] for r in sqlite3.connect(tmp_path / "state" / "telltale.db").execute(
            "SELECT kind FROM ai_outputs")]
        assert kinds == ["evidence_review"]


# ---------------------------------------------------------------- regression: explain + legacy


def test_explain_route_unchanged(client_factory, tmp_path):
    with client_factory() as client:
        snapshot = seed(client, ("sensor", "acoustic"))
        resp = client.post(URL_EXPLAIN, json={"assessment_id": str(snapshot.assessment_id)},
                           headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["feature"] == "explain"
        assert body["kind"] == "explain"
        assert body["validation_status"] == "validated"
        assert "model indications, not confirmed faults" in body["explanation_text"]
        oid = body["ai_output_id"]
        record = client.get(f"/api/v1/ai/outputs/{oid}", headers=AUTH_ALICE).json()
        assert record["feature"] == "explain"
        assert record["kind"] == "explain"
        assert ai_rows(tmp_path) == 1


def test_legacy_assessment_routes_unchanged(client_factory):
    with client_factory() as client:
        resp = client.post(
            URL_BASE,
            json={"reading": READING, "machine_id": MACHINE, "source_mode": "simulated"},
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["sensor"]
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


def test_schema_strictness(client_factory):
    with client_factory() as client:
        snapshot = seed(client)
        output = TemplateEvidenceReviewProvider().review(snapshot)
        payload = json.loads(output.model_dump_json())
        payload["forged"] = "overall_status"
        with pytest.raises(Exception):
            AiEvidenceReviewOutput.model_validate(payload)
        payload2 = json.loads(output.model_dump_json())
        payload2["feature"] = "explain"
        with pytest.raises(Exception):
            AiEvidenceReviewOutput.model_validate(payload2)
        payload3 = json.loads(output.model_dump_json())
        payload3["review_text"] = "x"
        with pytest.raises(Exception):
            AiEvidenceReviewOutput.model_validate(payload3)
