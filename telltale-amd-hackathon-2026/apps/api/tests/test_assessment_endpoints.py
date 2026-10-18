"""Endpoint slice-1 tests: trusted sensor-only assessments with demo auth.

Covers the exact approved contract: strict request boundaries (forged fields are
rejected, never ignored), server-side inference through the registry, owner-scoped
storage (migration v2), the route-local error envelope with server-generated
request ids, cached model identity and migration safety (fresh DB, v1 -> v2
upgrade, rerun, rollback).

Everything runs on the tiny synthetic fixture models from conftest: no downloads,
no training, no GPUs. All temporary databases live under D: (TEMP/TMP are pointed
there by the test run).
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from telltale_api.core.config import Settings
from telltale_api.main import create_app
from telltale_api.schemas.assessment import SCHEMA_VERSION, EvidenceDraft, ModalityDraft
from telltale_api.schemas.sensor import Contribution, FailureModeScore, SensorAssessment
from telltale_api.services import assessment_store as store_mod
from telltale_api.services.assess import RULE_VERSION, create_assessment
from telltale_api.services.assessment_store import AssessmentAlreadyExists, AssessmentStore

URL = "/api/v1/assessments"

TOKENS = json.dumps({"alice": "tok-alice-9f3d2c7e", "bob": "tok-bob-77aa11bb"})
AUTH_ALICE = {"Authorization": "Bearer tok-alice-9f3d2c7e"}
AUTH_BOB = {"Authorization": "Bearer tok-bob-77aa11bb"}

READING = {
    "product_quality": "M",
    "air_temperature_k": 302.4,
    "process_temperature_k": 310.6,
    "rotational_speed_rpm": 1352.0,
    "torque_nm": 54.2,
    "tool_wear_min": 212.0,
}

FAKE_RESULT = {
    "failure_probability": 0.42424242424242427,
    "risk_level": "medium",
    "decision_threshold": 0.3131313131313131,
    "failure_modes": [
        {
            "code": "TWF",
            "name": "Tool wear",
            "probability": 0.2,
            "threshold": 0.5,
            "reliable": True,  # schema default, so it round-trips through the output
            "flagged": False,
        }
    ],
    "contributions": [
        {"feature": "tool_wear_min", "label": "Tool wear", "value": 212.0, "contribution": 0.51}
    ],
    "base_value": -2.0,
    "derived": {"tool_wear_slope": 1.0},
    "warnings": [],
}


class FakeSensor:
    """Deterministic stand-in for SensorPredictor: records calls, replays a fixed result."""

    def __init__(self, result: dict | None = None, error: Exception | None = None):
        self.result = dict(result or FAKE_RESULT)
        self.error = error
        self.calls: list[list[dict]] = []
        self.metadata = {"caveats": ["fixture demo model"], "version": "9.9.9-fake"}

    def assess(self, readings: list[dict]) -> list[dict]:
        self.calls.append(readings)
        if self.error is not None:
            raise self.error
        return [dict(self.result)]


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


def post(client, body, auth=AUTH_ALICE):
    return client.post(URL, json=body, headers=dict(auth) if auth else {})


def payload(**over):
    body = {"reading": dict(READING), "machine_id": "cnc-a1", "source_mode": "simulated"}
    body.update(over)
    return body


def assert_error(resp, status: int, code: str) -> dict:
    """Assert the exact new-route error envelope shape and return the body."""
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error", "request_id"}, body
    assert set(body["error"]) <= {"code", "message", "details"}, body["error"]
    assert body["error"]["code"] == code, body
    assert body["error"].get("message"), "error message must be non-empty"
    assert re.fullmatch(r"[0-9a-f]{32}", body["request_id"]), body["request_id"]
    assert resp.headers.get("X-Request-ID") == body["request_id"]
    return body


def create_ok(client, body=None, auth=AUTH_ALICE) -> dict:
    resp = post(client, body or payload(), auth=auth)
    assert resp.status_code == 201, resp.text
    return resp.json()


def sensor_output_local(risk: str = "low", p: float = 0.002423118420741755) -> SensorAssessment:
    return SensorAssessment(
        failure_probability=p,
        risk_level=risk,  # type: ignore[arg-type]
        decision_threshold=0.45691827248299344,
        failure_modes=[
            FailureModeScore(code="TWF", name="Tool wear", probability=0.1, threshold=0.5, flagged=False)
        ],
        contributions=[
            Contribution(feature="tool_wear_min", label="Tool wear", value=212.0, contribution=0.51)
        ],
        base_value=-2.0,
        derived={"tool_wear_slope": 1.0},
        warnings=[],
    )


def legacy_snapshot(machine_id: str = "cnc-a1"):
    """A stored assessment as the foundation slice would have written it."""
    now = datetime.now(timezone.utc)
    return create_assessment(
        machine_id=machine_id,
        source_mode="simulated",
        expected_modalities=["sensor"],
        modalities=[
            ModalityDraft(
                kind="sensor",
                availability="available",
                reason_code="ok",
                source_mode="simulated",
                observed_at=None,
                inference_at=now,
                model_identity=None,
                limitations=[],
                output=sensor_output_local(),
            )
        ],
        evidence=[
            EvidenceDraft(
                kind="failure_probability",
                value=0.1,
                unit="probability",
                source="sensor",
                source_mode="simulated",
                observed_at=None,
            )
        ],
    )


# ---------------------------------------------------------------- strict request boundaries


def test_create_rejects_forged_outer_fields(client_factory):
    forged_keys = {
        "assessment_id",
        "schema_version",
        "rule_version",
        "overall_status",
        "owner",
        "expected_modalities",
        "quality_flags",
        "status_reasons",
        "evidence_completeness",
    }
    forged = payload(
        assessment_id=str(uuid4()),
        schema_version="9.9",
        rule_version="assess/v9",
        overall_status="normal",
        owner="alice",
        expected_modalities=["sensor", "acoustic"],
        quality_flags=[],
        status_reasons=[],
        evidence_completeness={"expected": 3, "present": 3, "ratio": 1.0},
    )
    with client_factory() as c:
        body = assert_error(post(c, forged), 422, "request_validation")
        forbidden = [d for d in body["error"]["details"] if d["type"] == "extra_forbidden"]
        assert {d["loc"][-1] for d in forbidden} == forged_keys
        assert c.app.state.telltale.assessments.list() == []  # nothing stored


def test_create_rejects_forged_nested_reading_fields(client_factory):
    bad = payload()
    bad["reading"] = dict(READING, score=0.987, risk_level="critical", assessment_id=str(uuid4()))
    with client_factory() as c:
        body = assert_error(post(c, bad), 422, "request_validation")
        forbidden = [d for d in body["error"]["details"] if d["type"] == "extra_forbidden"]
        assert {d["loc"][-1] for d in forbidden} == {"score", "risk_level", "assessment_id"}
        assert all(d["loc"][0] == "reading" for d in forbidden)
        assert c.app.state.telltale.assessments.list() == []


def test_create_rejects_unsupported_source_modes(client_factory):
    with client_factory() as c:
        for mode in ("live", "uploaded"):
            body = assert_error(post(c, payload(source_mode=mode)), 422, "request_validation")
            assert "source_mode" in json.dumps(body["error"]["details"])
        for mode in ("dataset_demo", "simulated"):
            assert post(c, payload(source_mode=mode)).status_code == 201


def test_create_rejects_malformed_bodies(client_factory):
    with client_factory() as c:
        resp = c.post(URL, data="air=1", headers={"content-type": "text/plain", **AUTH_ALICE})
        assert_error(resp, 422, "invalid_content_type")
        resp = c.post(URL, data="{oops", headers={"content-type": "application/json", **AUTH_ALICE})
        assert_error(resp, 422, "invalid_json")
        resp = c.post(URL, json=[1, 2], headers=AUTH_ALICE)
        assert_error(resp, 422, "request_validation")
        assert c.app.state.telltale.assessments.list() == []


# ---------------------------------------------------------------- trusted server flow


def test_create_calls_predictor_and_preserves_output(client_factory):
    with client_factory() as c:
        state = c.app.state.telltale
        fake = FakeSensor()
        state.registry.sensor = fake
        readings_before = {
            mid: dict(st.reading) if st.reading else None for mid, st in state.fleet.machines.items()
        }
        machines_before = set(state.fleet.machines)

        resp = post(c, payload())
        assert resp.status_code == 201, resp.text
        # the predictor received exactly the strict reading (defaults applied)
        assert fake.calls == [[dict(READING)]]

        body = resp.json()
        # server-owned stamps and the deliberate sensor-only profile
        assert body["schema_version"] == SCHEMA_VERSION
        assert body["rule_version"] == RULE_VERSION
        assert body["machine_id"] == "cnc-a1"
        assert body["source_mode"] == "simulated"
        assert body["expected_modalities"] == ["sensor"]  # completeness = 1/1 SENSOR only
        assert body["evidence_completeness"] == {"expected": 1, "present": 1, "ratio": 1.0}
        assert body["approved_playbook_item_ids"] == []
        # assessment/result/evidence ids are server-generated full UUIDs
        assessment_id = body["assessment_id"]
        assert isinstance(assessment_id, str)
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", assessment_id
        )
        modality = body["modalities"][0]
        assert re.fullmatch(r"[0-9a-f-]{36}", modality["result_id"])
        assert re.fullmatch(r"[0-9a-f-]{36}", body["evidence"][0]["evidence_id"])
        # original model output preserved byte-for-byte through JSON
        out = modality["output"]
        assert out["failure_probability"] == FAKE_RESULT["failure_probability"]
        assert out["decision_threshold"] == FAKE_RESULT["decision_threshold"]
        assert out["failure_modes"] == FAKE_RESULT["failure_modes"]
        assert out["contributions"] == FAKE_RESULT["contributions"]
        assert modality["illustrative"] is True
        assert modality["limitations"] == ["fixture demo model"]
        assert body["evidence"][0]["value"] == FAKE_RESULT["failure_probability"]
        # assess/v1 status mapping + quality flags
        assert body["overall_status"] == "watch"  # medium -> watch
        assert body["status_reasons"] == []
        codes = {f["code"] for f in body["quality_flags"]}
        assert {"SOURCE_ILLUSTRATIVE", "MODEL_LIMITATIONS_PRESENT", "MODEL_IDENTITY_INCOMPLETE"} <= codes
        # headers
        assert resp.headers["Location"] == f"{URL}/{assessment_id}"
        assert re.fullmatch(r"[0-9a-f]{32}", resp.headers["X-Request-ID"])
        # creating an assessment never mutates the fleet
        assert set(state.fleet.machines) == machines_before
        readings_after = {
            mid: dict(st.reading) if st.reading else None for mid, st in state.fleet.machines.items()
        }
        assert readings_after == readings_before
        # exactly one row for the authenticated owner
        assert [str(a.assessment_id) for a in state.assessments.list(owner="alice")] == [assessment_id]


def test_create_real_fixture_model_matches_direct_inference(client_factory):
    with client_factory() as c:
        state = c.app.state.telltale
        predictor = state.registry.require_sensor()
        body = create_ok(c)
        [direct] = predictor.assess([dict(READING)])
        out = body["modalities"][0]["output"]
        assert out["failure_probability"] == direct["failure_probability"]
        assert out["risk_level"] == direct["risk_level"]
        assert body["evidence"][0]["value"] == direct["failure_probability"]
        expected_status = {"low": "normal", "medium": "watch", "high": "critical"}[direct["risk_level"]]
        assert body["overall_status"] == expected_status
        identity = body["modalities"][0]["model_identity"]
        assert identity["artifact_fingerprint"] == state.registry.sensor_identity["artifact_fingerprint"]
        assert re.fullmatch(r"[0-9a-f]{64}", identity["artifact_fingerprint"])
        assert identity["version"] == predictor.metadata.get("version")
        assert identity["training_dataset"] == (
            predictor.metadata.get("dataset") or predictor.metadata.get("data_source")
        )
        # preprocessing identity is unknown, stays None, and is flagged - never invented
        assert identity["preprocessing_fingerprint"] is None
        codes = {f["code"] for f in body["quality_flags"]}
        assert "MODEL_IDENTITY_INCOMPLETE" in codes
        assert body["expected_modalities"] == ["sensor"]
        assert body["evidence_completeness"] == {"expected": 1, "present": 1, "ratio": 1.0}


# ---------------------------------------------------------------- timestamps


def test_create_normalizes_aware_observed_at(client_factory):
    with client_factory() as c:
        resp = post(c, payload(observed_at="2026-10-14T16:00:00+05:00"))
        assert resp.status_code == 201, resp.text
        body = resp.json()
        observed = datetime.fromisoformat(body["modalities"][0]["observed_at"])
        assert observed == datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc)
        assert datetime.fromisoformat(body["evidence"][0]["observed_at"]) == observed
        # observed_at (measurement) stays distinct from server inference/created times
        inference_at = datetime.fromisoformat(body["modalities"][0]["inference_at"])
        created_at = datetime.fromisoformat(body["created_at"])
        assert inference_at.tzinfo is not None and created_at.tzinfo is not None
        assert observed < inference_at <= created_at
        codes = {f["code"] for f in body["quality_flags"]}
        assert "TIMESTAMP_MISSING" not in codes  # a known timestamp is not "missing"


def test_create_rejects_naive_and_malformed_timestamps(client_factory):
    with client_factory() as c:
        naive = assert_error(post(c, payload(observed_at="2026-10-15T12:00:00")), 422, "request_validation")
        assert "timezone-aware" in json.dumps(naive["error"]["details"])
        for stamp in ("not-a-timestamp", "2026-13-45T99:00:00Z"):
            body = assert_error(post(c, payload(observed_at=stamp)), 422, "request_validation")
            assert "observed_at" in json.dumps(body["error"]["details"])
        assert c.app.state.telltale.assessments.list() == []


# ---------------------------------------------------------------- machine association


def test_create_unknown_machine_422_without_inventory(client_factory):
    with client_factory() as c:
        unknown = post(c, payload(machine_id="zzz-99"))
        body = assert_error(unknown, 422, "unknown_machine")
        assert body["error"]["details"] == {"machine_id": "zzz-99"}
        # the error never becomes a machine-ID inventory
        assert "cnc-a2" not in unknown.text and "drl-c1" not in unknown.text
        assert create_ok(c, payload(machine_id="cnc-a2"))["machine_id"] == "cnc-a2"


# ---------------------------------------------------------------- failure paths


def test_missing_model_returns_503_and_stores_nothing(client_factory, tmp_path):
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir(exist_ok=True)
    with client_factory(
        models_dir=empty_models,
        fallback_models_dir=tmp_path / "no-fallback",
        sensor_autotrain=False,
    ) as c:
        body = assert_error(post(c, payload()), 503, "dependency_unavailable")
        assert "telltale train-sensor" in body["error"]["message"]  # actionable hint survives
        assert c.app.state.telltale.assessments.list() == []


def test_inference_failure_returns_500_and_stores_nothing(client_factory):
    with client_factory() as c:
        state = c.app.state.telltale
        state.registry.sensor = FakeSensor(error=RuntimeError("boom at C:\\secret\\model.txt"))
        failed = post(c, payload())
        body = assert_error(failed, 500, "internal_error")
        for leak in ("boom", "secret", "model.txt", "Traceback"):
            assert leak not in failed.text
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert state.assessments.list() == []  # no fabricated success, no partial row


def test_storage_failure_returns_500_without_partial_row(client_factory):
    with client_factory() as c:
        state = c.app.state.telltale

        def broken_insert(*args, **kwargs):
            raise OSError("disk full: D:\\secret\\telltale.db")

        state.assessments.insert = broken_insert
        failed = post(c, payload())
        body = assert_error(failed, 500, "internal_error")
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert "disk full" not in failed.text and "secret" not in failed.text
        assert state.assessments.list() == []


def test_conflict_maps_to_409_envelope(client_factory):
    with client_factory() as c:
        state = c.app.state.telltale

        def duplicate(*args, **kwargs):
            raise AssessmentAlreadyExists(uuid4())

        state.assessments.insert = duplicate
        assert_error(post(c, payload()), 409, "assessment_conflict")


def test_storage_unavailable_returns_503(client_factory, monkeypatch):
    def raise_store(path):
        raise RuntimeError("cannot open the assessment database")

    monkeypatch.setattr("telltale_api.main.AssessmentStore", raise_store)
    with client_factory() as c:
        state = c.app.state.telltale
        assert state.assessments is None
        assert_error(post(c, payload()), 503, "storage_unavailable")
        assert_error(c.get(URL, headers=AUTH_ALICE), 503, "storage_unavailable")
        assert_error(c.get(f"{URL}/{uuid4()}", headers=AUTH_ALICE), 503, "storage_unavailable")


# ---------------------------------------------------------------- demo authentication


def test_auth_missing_invalid_and_actor_header_ignored(client_factory):
    with client_factory() as c:
        assert_error(post(c, payload(), auth=None), 401, "unauthorized")
        wrong = {"Authorization": "Bearer wrong-token-value"}
        assert_error(post(c, payload(), auth=wrong), 401, "unauthorized")
        actor_only = {"X-Actor-ID": "alice"}  # never trusted as authentication
        assert_error(post(c, payload(), auth=actor_only), 401, "unauthorized")
        assert_error(c.get(URL), 401, "unauthorized")
        assert_error(c.get(f"{URL}/{uuid4()}"), 401, "unauthorized")
        assert c.app.state.telltale.assessments.list() == []  # denied requests store nothing


def test_auth_unconfigured_denies_all(client_factory):
    with client_factory(api_tokens="") as c:
        assert_error(post(c, payload()), 401, "auth_disabled")
        assert_error(c.get(URL), 401, "auth_disabled")
        assert_error(post(c, payload(), auth=AUTH_ALICE), 401, "auth_disabled")


def test_auth_malformed_configuration_fails_closed(client_factory):
    bad_configs = [
        "{oops",  # malformed JSON
        '{"alice": "sekrit-token-one",',  # malformed JSON containing a token - must not echo it
        '{"a": "t1", "a": "t2"}',  # duplicate actor definitions
        '{"a": "t1", "b": "t1"}',  # duplicate token values
        '{"": "t1"}',  # empty actor name
        '{"a": ""}',  # empty token
        '{"a": {"nested": "x"}}',  # token value is not a string
        '["not", "an", "object"]',  # top level is not an object
    ]
    with client_factory() as c:
        settings = c.app.state.telltale.settings
        for raw in bad_configs:
            settings.api_tokens = raw
            denied = post(c, payload())
            body = assert_error(denied, 401, "auth_misconfigured")
            assert "TELLTALE_API_TOKENS" in body["error"]["message"]
            for secret in ("sekrit-token-one", "t1", "nested"):
                assert secret not in denied.text
        # a valid configuration still works afterwards (per-string cache stays correct)
        settings.api_tokens = TOKENS
        assert post(c, payload()).status_code == 201


# ---------------------------------------------------------------- ownership


def test_actor_isolation_get_and_list(client_factory):
    with client_factory() as c:
        a1 = create_ok(c, payload(observed_at="2026-10-14T10:00:00+00:00"))
        a2 = create_ok(c, payload(machine_id="cnc-a2"))
        b1 = create_ok(c, auth=AUTH_BOB)
        # cross-owner GET is 404, identical to an unknown id (no existence leak)
        assert_error(c.get(f"{URL}/{a1['assessment_id']}", headers=AUTH_BOB), 404, "not_found")
        assert_error(c.get(f"{URL}/{uuid4()}", headers=AUTH_ALICE), 404, "not_found")
        got = c.get(f"{URL}/{a1['assessment_id']}", headers=AUTH_ALICE)
        assert got.status_code == 200
        assert got.json() == a1
        # lists are always owner-filtered
        alice_ids = {i["assessment_id"] for i in c.get(URL, headers=AUTH_ALICE).json()["items"]}
        bob_ids = {i["assessment_id"] for i in c.get(URL, headers=AUTH_BOB).json()["items"]}
        assert alice_ids == {a1["assessment_id"], a2["assessment_id"]}
        assert bob_ids == {b1["assessment_id"]}


def test_list_machine_filter_and_pagination(client_factory):
    with client_factory() as c:
        ids_cnc_a1 = [create_ok(c)["assessment_id"], create_ok(c)["assessment_id"]]
        id_cnc_a2 = create_ok(c, payload(machine_id="cnc-a2"))["assessment_id"]
        all_ids = set(ids_cnc_a1) | {id_cnc_a2}

        filtered = c.get(f"{URL}?machine_id=cnc-a1", headers=AUTH_ALICE).json()
        assert {i["assessment_id"] for i in filtered["items"]} == set(ids_cnc_a1)
        # machine filter never escapes the owner scope
        assert c.get(f"{URL}?machine_id=cnc-a1", headers=AUTH_BOB).json()["items"] == []

        page1 = c.get(f"{URL}?limit=2", headers=AUTH_ALICE).json()
        page2 = c.get(f"{URL}?limit=2&offset=2", headers=AUTH_ALICE).json()
        p1_ids = [i["assessment_id"] for i in page1["items"]]
        p2_ids = [i["assessment_id"] for i in page2["items"]]
        assert len(p1_ids) == 2 and len(p2_ids) == 1
        assert not set(p1_ids) & set(p2_ids)
        assert set(p1_ids) | set(p2_ids) == all_ids
        # stable newest-first ordering across pages
        stamps = [datetime.fromisoformat(i["created_at"]) for i in page1["items"] + page2["items"]]
        assert stamps == sorted(stamps, reverse=True)
        # bounded parameters fail locally with the envelope (not FastAPI's default 422)
        for query in ("limit=0", "limit=201", "limit=abc", "offset=-1", "offset=x"):
            body = assert_error(c.get(f"{URL}?{query}", headers=AUTH_ALICE), 422, "request_validation")
            assert body["error"]["details"]["param"] in {"limit", "offset"}
        default = c.get(URL, headers=AUTH_ALICE).json()
        assert default["limit"] == 50 and default["offset"] == 0 and len(default["items"]) == 3


def test_ownerless_legacy_row_stays_inaccessible(client_factory):
    with client_factory() as c:
        state = c.app.state.telltale
        legacy = legacy_snapshot()
        state.assessments.insert(legacy)  # foundation-style insert: owner defaults to ''
        # authenticated views never see the ownerless row
        items = c.get(URL, headers=AUTH_ALICE).json()["items"]
        assert [i for i in items if i["assessment_id"] == str(legacy.assessment_id)] == []
        assert_error(c.get(f"{URL}/{legacy.assessment_id}", headers=AUTH_ALICE), 404, "not_found")
        # the row itself is preserved and never auto-assigned to an owner
        assert state.assessments.get(legacy.assessment_id) is not None  # unscoped internal view
        conn = sqlite3.connect(state.settings.state_db or state.settings.data_dir / "state" / "telltale.db")
        owners = conn.execute("SELECT owner FROM assessments").fetchall()
        conn.close()
        assert owners == [("",)]


# ---------------------------------------------------------------- error envelope + legacy boundary


def test_error_envelope_request_ids_are_server_generated(client_factory):
    with client_factory() as c:
        first = assert_error(post(c, payload(), auth=None), 401, "unauthorized")
        # caller-supplied request ids are ignored; the server mints its own
        forged = {"X-Request-ID": "forged-by-client"}
        denied = c.post(URL, json=payload(), headers=forged)
        second = assert_error(denied, 401, "unauthorized")
        assert second["request_id"] != "forged-by-client"
        assert first["request_id"] != second["request_id"]
        # the same holds for successful responses: a fresh server id per request
        ok = c.post(URL, json=payload(), headers={**AUTH_ALICE, **forged})
        assert ok.status_code == 201, ok.text
        assert re.fullmatch(r"[0-9a-f]{32}", ok.headers["X-Request-ID"])
        assert ok.headers["X-Request-ID"] != "forged-by-client"


def test_legacy_routes_outside_auth_boundary_and_errors_unchanged(client_factory):
    with client_factory() as c:  # api_tokens IS configured here
        # legacy inference stays open (no token required - boundary is new-routes-only)
        resp = c.post("/api/v1/sensors/predict", json=dict(READING))
        assert resp.status_code == 200, resp.text
        assert "failure_probability" in resp.json()
        # legacy validation errors keep FastAPI's {"detail": ...} shape (no app-wide handler)
        bad = c.post("/api/v1/sensors/predict", json=dict(READING, torque_nm=999))
        assert bad.status_code == 422
        body = bad.json()
        assert "detail" in body and "error" not in body and "request_id" not in body


# ---------------------------------------------------------------- immutability + restart


def test_snapshot_immutable_across_reads_and_restart(client_factory):
    with client_factory() as c:
        first = create_ok(c, payload(observed_at="2026-10-14T10:00:00+00:00"))
        again = c.get(f"{URL}/{first['assessment_id']}", headers=AUTH_ALICE)
        assert again.status_code == 200
        assert again.json() == first
        second = create_ok(c)
        assert second["assessment_id"] != first["assessment_id"]
        # the first snapshot is byte-identical after other writes
        refetch = c.get(f"{URL}/{first['assessment_id']}", headers=AUTH_ALICE)
        assert refetch.json() == first
        assert refetch.content == again.content
    # a fresh app instance over the same data directory serves the identical snapshot
    with client_factory() as c2:
        relaunch = c2.get(f"{URL}/{first['assessment_id']}", headers=AUTH_ALICE)
        assert relaunch.status_code == 200
        assert relaunch.json() == first


# ---------------------------------------------------------------- cached model identity


def test_cached_identity_stable_across_requests_and_reload(client_factory, tmp_path, monkeypatch):
    with client_factory() as c:
        state = c.app.state.telltale
        identity = state.registry.sensor_identity
        assert identity is not None
        assert re.fullmatch(r"[0-9a-f]{64}", identity["artifact_fingerprint"])
        assert identity["preprocessing_fingerprint"] is None
        assert identity["version"] == state.registry.sensor.metadata.get("version")
        # POST reuses the cached identity (no per-request artifact hashing)
        body = create_ok(c)
        assert state.registry.sensor_identity is identity
        assert (
            body["modalities"][0]["model_identity"]["artifact_fingerprint"]
            == identity["artifact_fingerprint"]
        )
        # reload recomputes the same deterministic fingerprint
        state.registry.load()
        assert state.registry.sensor_identity["artifact_fingerprint"] == identity["artifact_fingerprint"]
        assert state.registry.sensor is not None
        # a failed load drops the identity together with the predictor (always aligned)
        monkeypatch.setattr(state.registry, "sensor_dir", lambda: tmp_path / "no-sensor-here")
        state.registry.load()
        assert state.registry.sensor is None
        assert state.registry.sensor_identity is None
        assert_error(post(c, payload()), 503, "dependency_unavailable")


# ---------------------------------------------------------------- migrations (temporary DBs only)


def _versions(db):
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    finally:
        conn.close()


def test_migration_v2_fresh_database(tmp_path):
    db = tmp_path / "fresh.db"
    store = AssessmentStore(db)
    store.close()
    assert _versions(db) == [1, 2]
    conn = sqlite3.connect(db)
    columns = [r[1] for r in conn.execute("PRAGMA table_info(assessments)")]
    conn.close()
    assert "owner" in columns


def test_migration_v1_to_v2_upgrade(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
        "applied_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (1, 'assessments_v1', ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    store_mod._migration_v1(conn)  # an old (v1-only) database
    legacy = legacy_snapshot()
    conn.execute(
        "INSERT INTO assessments (assessment_id, machine_id, source_mode, schema_version, "
        "rule_version, created_at, json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(legacy.assessment_id),
            legacy.machine_id,
            legacy.source_mode,
            legacy.schema_version,
            legacy.rule_version,
            legacy.created_at.isoformat(),
            legacy.model_dump_json(),
        ),
    )
    conn.close()

    store = AssessmentStore(db)  # upgrade path: v2 is applied on open
    try:
        assert _versions(db) == [1, 2]
        raw = sqlite3.connect(db)
        owners = raw.execute("SELECT owner FROM assessments").fetchall()
        raw.close()
        assert owners == [("",)]  # legacy row keeps owner='' and is never reassigned
        # legacy rows stay reachable only through the unscoped internal view
        assert store.get(legacy.assessment_id) is not None
        assert store.get(legacy.assessment_id, owner="alice") is None
        assert store.list(owner="alice") == []
        assert [a.assessment_id for a in store.list()] == [legacy.assessment_id]
        # new inserts carry their owner immediately
        fresh = legacy_snapshot(machine_id="cnc-a2")
        store.insert(fresh, owner="alice")
        assert store.get(fresh.assessment_id, owner="alice") is not None
        assert store.get(fresh.assessment_id, owner="bob") is None
        assert {a.assessment_id for a in store.list(owner="alice")} == {fresh.assessment_id}
    finally:
        store.close()


def test_migration_rerun_is_recorded_noop(tmp_path):
    db = tmp_path / "rerun.db"
    AssessmentStore(db).close()
    AssessmentStore(db).close()  # second open: pending set is empty, no error
    assert _versions(db) == [1, 2]  # recorder rows are not duplicated


def test_migration_failure_rolls_back(tmp_path, monkeypatch):
    db = tmp_path / "rollback.db"

    def boom(conn):
        conn.execute("CREATE TABLE should_not_exist (x TEXT)")
        raise RuntimeError("migration failed on purpose")

    monkeypatch.setattr(store_mod, "MIGRATIONS", store_mod.MIGRATIONS + ((3, "boom", boom),))
    with pytest.raises(RuntimeError):
        AssessmentStore(db)
    conn = sqlite3.connect(db)
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    columns = [r[1] for r in conn.execute("PRAGMA table_info(assessments)")]
    conn.close()
    assert versions == [1, 2]  # the failed version is never recorded
    assert "should_not_exist" not in tables  # its DDL was rolled back
    assert "owner" in columns  # committed migrations stay intact
