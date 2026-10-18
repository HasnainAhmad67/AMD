"""Endpoint slice-4 tests: trusted MULTIMODAL assessment creation (one request, >=2 modalities).

Covers the approved contract: strict multipart boundaries (unknown/duplicate/forged
parts rejected, one part never serving two modalities, identical bytes in DISTINCT
parts allowed), auth before body parsing, the complete machine-compatibility
matrix validated before any inference, threadpool inference for every requested
modality, server-owned snapshot fields, max-rank overall status, provenance rules
(assessment "uploaded" whenever any upload exists - always on success here because
two modalities need at least one upload; the legacy sensor-declared mode is kept
per-modality), timestamp semantics, owner scoping, immutability, restart
persistence, upload safety (bounded reads; corrupt/empty/oversized/too-short/
non-finite rejected; the acoustic MATLAB rule vs the bearing .mat rule), hash-only
retention, failure atomicity (no row on any failure) and resource cleanup.

Mock-based tests inject deterministic fakes; the API/direct comparison test runs
the REAL fixture-trained models (sensor + fan ensemble + bearing classifier) from
conftest. No downloads, no GPUs, no network: everything runs on the validated venv
with all temporary state on D: (TEMP/TMP pointed there by the test run).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio
import soundfile as sf
from fastapi.testclient import TestClient

from telltale_api.api.routes import assessments as assessments_routes
from telltale_api.core.config import Settings
from telltale_api.main import create_app
from telltale_api.services.registry import fingerprint_dir

URL_M = "/api/v1/assessments/multimodal"
URL_BASE = "/api/v1/assessments"

TOKENS = json.dumps({"alice": "tok-alice-9f3d2c7e", "bob": "tok-bob-77aa11bb"})
AUTH_ALICE = {"Authorization": "Bearer tok-alice-9f3d2c7e"}
AUTH_BOB = {"Authorization": "Bearer tok-bob-77aa11bb"}

MACHINE = "cnc-a1"  # declared acoustic_profile: fan (fixture fan ensemble is loaded)

READING = {
    "product_quality": "M",
    "air_temperature_k": 302.4,
    "process_temperature_k": 310.6,
    "rotational_speed_rpm": 1352.0,
    "torque_nm": 54.2,
    "tool_wear_min": 212.0,
}

SENSOR_JSON = json.dumps({"reading": READING, "source_mode": "simulated"})
SENSOR_JSON_DEMO = json.dumps({"reading": READING, "source_mode": "dataset_demo"})

# ---------------------------------------------------------------- deterministic fakes

FAKE_RESULT = {
    "failure_probability": 0.42424242424242427,
    "risk_level": "medium",  # -> watch under assess/v1
    "decision_threshold": 0.3131313131313131,
    "failure_modes": [
        {
            "code": "TWF",
            "name": "Tool wear",
            "probability": 0.2,
            "threshold": 0.5,
            "reliable": True,
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

SENSOR_LOW = {**FAKE_RESULT, "failure_probability": 0.01, "risk_level": "low"}
SENSOR_HIGH = {**FAKE_RESULT, "failure_probability": 0.99, "risk_level": "high"}

FAKE_ACOUSTIC_RESULT = {
    "machine_type": "fan",
    "method": "ensemble",
    "score": 1.5,
    "threshold": 1.34,
    "severity": 1.1194,
    "p_value": 0.03,
    "unusual_percentile": 97.0,
    "calibration": "global",
    "status": "anomalous",  # -> critical under assess/v1
    "is_anomalous": True,
    "duration_s": 2.0,
    "sample_rate": 16000,
    "peak_time_s": 1.0,
    "frame_errors": {"times": [0.5, 1.0], "values": [1.4, 1.5]},
    "spectrogram": {"times": [0.5], "band_hz": [125.0], "values": [[-20.0]]},
    "band_deviation": {"band_hz": [125.0], "z": [0.1]},
    "waveform": {"duration_s": 2.0, "min": [-0.1], "max": [0.1]},
}

ACOUSTIC_NORMAL = {**FAKE_ACOUSTIC_RESULT, "status": "normal", "is_anomalous": False, "score": 0.2}

FAKE_IDENTITY = {
    "name": "acoustic-fan-ensemble",
    "version": "9.9.9-fake",
    "trained_at": None,
    "artifact_fingerprint": "a" * 64,
    "preprocessing_fingerprint": "b" * 64,
    "training_dataset": "mimii_due",
    "reference_fingerprint": "c" * 64,
}

FAKE_BEARING_RESULT = {
    "predicted_class": "normal",
    "label": "Healthy bearing",
    "confidence": 0.87,
    "prediction_set": ["normal"],
    "prediction_set_labels": ["Healthy bearing"],
    "coverage": 0.9,
    "uncertain": False,
    "rpm_used": 1200.0,
    "rpm_assumed": False,
    "segment_agreement": 0.95,
    "segments": 3,
    "duration_s": 2.0,
    "probabilities": [
        {"class_name": "normal", "label": "Healthy bearing", "probability": 0.87},
        {"class_name": "inner_race", "label": "Inner race defect", "probability": 0.13},
    ],
}

FAULT_RESULT = {
    **FAKE_BEARING_RESULT,
    "predicted_class": "inner_race",
    "label": "Inner race defect",
    "confidence": 0.61,
    "prediction_set": ["inner_race"],
    "prediction_set_labels": ["Inner race defect"],
}

UNCERTAIN_RESULT = {
    **FAKE_BEARING_RESULT,
    "uncertain": True,
    "prediction_set": ["normal", "inner_race"],
    "prediction_set_labels": ["Healthy bearing", "Inner race defect"],
}

AMBIGUOUS_RESULT = {
    **FAKE_BEARING_RESULT,
    "prediction_set": ["normal", "inner_race"],
    "prediction_set_labels": ["Healthy bearing", "Inner race defect"],
}

FAKE_BEARING_IDENTITY = {
    "name": "bearing-lightgbm",
    "version": "9.9.9-fake",
    "trained_at": None,
    "artifact_fingerprint": "a" * 64,
    "preprocessing_fingerprint": "b" * 64,
    "training_dataset": "mendeley_n9y9c7xrz3",
    "reference_fingerprint": "c" * 64,
}


class FakeSensor:
    """Deterministic stand-in for SensorPredictor (records one assess() call)."""

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


class FakeAcoustic:
    """Deterministic stand-in for AcousticDetector (same surface the route uses)."""

    def __init__(self, result: dict | None = None, error: Exception | None = None, enrolled: dict | None = None):
        self.cfg = SimpleNamespace(sample_rate=16000, channel=0)
        self.metadata = {
            "caveats": ["fixture demo sound model"],
            "max_audio_seconds": 30.0,
            "machine_type": "fan",
            "scorer": "ensemble",
            "version": "9.9.9-fake",
        }
        self.enrolled = dict(enrolled or {})
        self.error = error
        self.result = copy.deepcopy(result or FAKE_ACOUSTIC_RESULT)
        self.calls: list = []

    def analyze(self, source, *args, **kwargs):
        self.calls.append(source)
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.result)


class FakeBearing:
    """Deterministic stand-in for FaultClassifier (same surface the route uses)."""

    def __init__(self, result: dict | None = None, error: Exception | None = None):
        self.cfg = SimpleNamespace(sample_rate=44100, channel=0, segment_seconds=1.0)
        self.metadata = {
            "caveats": ["fixture demo bearing classifier"],
            "max_audio_seconds": 60.0,
            "kind": "bearing_fault",
            "version": "9.9.9-fake",
            "classes": ["normal", "inner_race"],
        }
        self.error = error
        self.result = copy.deepcopy(result or FAKE_BEARING_RESULT)
        self.calls: list = []

    def diagnose(self, source, rpm=None):
        self.calls.append((source, rpm))
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.result)


# ---------------------------------------------------------------- fixtures + helpers


@pytest.fixture()
def client_factory(models_dir, tmp_path):
    """Build a TestClient over the session fixture models (caller enters the context)."""

    def _make(**over):
        over.setdefault("api_tokens", TOKENS)
        settings = Settings(
            models_dir=over.pop("models_dir", models_dir),
            data_dir=over.pop("data_dir", tmp_path),
            simulator_enabled=True,
            simulator_interval_s=3600,  # one startup tick (live telemetry exists), then quiet
            llm_base_url=None,
            **over,
        )
        return TestClient(create_app(settings))

    return _make


def wav_bytes(seconds: float = 2.0, sr: int = 16000, amplitude: float = 0.05, with_nan: bool = False) -> bytes:
    n = max(1, int(round(seconds * sr)))
    y = np.full(n, amplitude, dtype=np.float32)
    subtype = "PCM_16"
    if with_nan:
        y = y.astype(np.float64)
        y[0] = np.nan
        subtype = "FLOAT"
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype=subtype, format="WAV")
    return buf.getvalue()


def mat_bytes(seconds: float = 2.0, sr: int = 44100, freq: float = 900.0) -> bytes:
    """A valid MATLAB .mat recording - the bearing dataset's own container format."""
    t = np.arange(int(sr * seconds)) / sr
    buf = io.BytesIO()
    sio.savemat(buf, {"x": (0.4 * np.sin(2 * np.pi * freq * t))[:, None]})
    return buf.getvalue()


WAV_A = wav_bytes(seconds=2.0, amplitude=0.05)  # acoustic upload
WAV_B = wav_bytes(seconds=2.0, amplitude=0.02)  # bearing upload (distinct bytes)


def install_fakes(
    state,
    sensor_stub: FakeSensor | None = None,
    acoustic_stub: FakeAcoustic | None = None,
    bearing_stub: FakeBearing | None = None,
    acoustic_identity: dict | None = None,
    bearing_identity: dict | None = None,
) -> tuple[FakeSensor, FakeAcoustic, FakeBearing]:
    """Install fakes only where given; anything left installed keeps its current value."""
    if sensor_stub is not None:
        state.registry.sensor = sensor_stub
    if acoustic_stub is not None:
        state.registry.acoustic = {"fan": acoustic_stub}
        state.registry.acoustic_identity = {"fan": dict(acoustic_identity or FAKE_IDENTITY)}
    if bearing_stub is not None:
        state.registry.bearing = bearing_stub
        state.registry.bearing_identity = dict(bearing_identity or FAKE_BEARING_IDENTITY)
    return sensor_stub, acoustic_stub, bearing_stub


def install_all(state, **kw) -> tuple[FakeSensor, FakeAcoustic, FakeBearing]:
    """Deterministic fakes for every modality (default: normal-ish results)."""
    return install_fakes(state, FakeSensor(**kw.pop("sensor", {})), FakeAcoustic(**kw.pop("acoustic", {})), FakeBearing(**kw.pop("bearing", {})))


def bits_parts(bits: tuple[str, ...], *, acoustic: bytes | None = None, bearing: bytes | None = None, observed_at: str | None = None):
    """Build (fields, files) for exactly the requested modality bits."""
    fields: list[tuple[str, str]] = [("machine_id", MACHINE)]
    if "sensor" in bits:
        fields.append(("sensor_json", SENSOR_JSON))
    if observed_at is not None:
        fields.append(("observed_at", observed_at))
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    if "acoustic" in bits:
        files.append(("acoustic_file", ("clip.wav", acoustic if acoustic is not None else WAV_A, "audio/wav")))
    if "bearing" in bits:
        files.append(("bearing_file", ("clip.wav", bearing if bearing is not None else WAV_B, "audio/wav")))
    return fields, files


def post_mm(client, *, fields=None, files=None, auth=AUTH_ALICE, extra_headers=None):
    """Default: a valid sensor_json + acoustic_file request (>= 2 modalities).

    NOTE: httpx silently drops ``data`` when it is a list of tuples AND ``files``
    is present, so ordered field lists are normalized to a dict here (insertion
    order preserved). Part-order/duplicate cases use the hand-crafted
    ``raw_multipart`` bodies instead, never this helper.
    """
    if fields is None and files is None:
        fields, files = bits_parts(("sensor", "acoustic"))
    headers = dict(auth) if auth else {}
    headers.update(extra_headers or {})
    return client.post(URL_M, data=dict(fields or []), files=files, headers=headers)


def create_mm(client, bits=("sensor", "acoustic"), **kw) -> dict:
    """Valid >=2-modality creation; kwargs go to bits_parts (acoustic/bearing/observed_at)."""
    fields, files = bits_parts(bits, **kw)
    resp = post_mm(client, fields=fields, files=files)
    assert resp.status_code == 201, resp.text
    return resp.json()


def raw_multipart(fields, files, boundary: str = "----telltale-raw-mm1"):
    """Hand-crafted multipart body: exact control over duplicate/extra/typed parts."""
    out = bytearray()
    for name, value in fields:
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()
    for name, filename, payload, ctype in files:
        out += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        out += payload + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def raw_files(files) -> list[tuple[str, str, bytes, str]]:
    """Convert httpx-style file tuples to raw_multipart's 4-tuple shape."""
    return [(name, entry[0], entry[1], entry[2]) for name, entry in files]


def post_raw(client, fields, files, auth=AUTH_ALICE):
    body, ctype = raw_multipart(fields, raw_files(files))
    return client.post(URL_M, content=body, headers={**(auth or {}), "Content-Type": ctype})


def assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error", "request_id"}, body
    assert set(body["error"]) <= {"code", "message", "details"}, body["error"]
    assert body["error"]["code"] == code, body
    assert body["error"].get("message"), "error message must be non-empty"
    assert re.fullmatch(r"[0-9a-f]{32}", body["request_id"]), body["request_id"]
    assert resp.headers.get("X-Request-ID") == body["request_id"]
    return body


def modal(doc: dict, kind: str) -> dict:
    return next(m for m in doc["modalities"] if m["kind"] == kind)


def flag_codes(doc: dict) -> set[str]:
    return {f["code"] for f in doc["quality_flags"]}


def rows(client) -> list:
    return client.get(URL_BASE, headers=AUTH_ALICE).json()["items"]


# ---------------------------------------------------------------- happy paths (2 and 3 modalities)


def test_valid_combinations_201_location_and_get_parity(client_factory):
    """Every supported >=2-modality combination: 201 + Location, authenticated GET parity."""
    with client_factory() as client:
        install_all(client.app.state.telltale)
        for bits in (("sensor", "acoustic"), ("sensor", "bearing"), ("acoustic", "bearing"), ("sensor", "acoustic", "bearing")):
            doc = create_mm(client, bits)
            aid = doc["assessment_id"]
            assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", aid)
            # expected_modalities is EXACTLY the server-validated requested set (canonical order)
            canonical = [k for k in ("sensor", "acoustic", "bearing") if k in bits]
            assert doc["expected_modalities"] == canonical
            assert [m["kind"] for m in doc["modalities"]] == canonical
            assert all(m["availability"] == "available" and m["reason_code"] == "ok" for m in doc["modalities"])
            # requested-multimodal completeness is 1.0 on success (scope: the requested set only)
            n = len(canonical)
            assert doc["evidence_completeness"] == {"expected": n, "present": n, "ratio": 1.0}
            assert "EVIDENCE_INCOMPLETE" not in flag_codes(doc)
            # assessment-level provenance: any upload present -> "uploaded"
            assert doc["source_mode"] == "uploaded"
            # no partial document: evidence carries every produced fact, in modality order
            kinds = [e["kind"] for e in doc["evidence"]]
            expected_kinds = []
            if "sensor" in bits:
                expected_kinds.append("failure_probability")
            if "acoustic" in bits:
                expected_kinds += [
                    "anomaly_score",
                    "conformal_p_value",
                    "audio_duration_original_s",
                    "audio_duration_processed_s",
                ]
            if "bearing" in bits:
                expected_kinds += [
                    "class_confidence",
                    "audio_duration_original_s",
                    "audio_duration_processed_s",
                ]
            assert kinds == expected_kinds
            # authenticated GET parity
            got = client.get(f"{URL_BASE}/{aid}", headers=AUTH_ALICE)
            assert got.status_code == 200, got.text
            assert got.json() == doc
            # bob cannot see or fetch it
            assert client.get(f"{URL_BASE}/{aid}", headers=AUTH_BOB).status_code == 404


def test_location_header_and_request_echo(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        resp = post_mm(client)
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        assert resp.headers["Location"] == f"/api/v1/assessments/{doc['assessment_id']}"
        rid = resp.headers["X-Request-ID"]
        assert re.fullmatch(r"[0-9a-f]{32}", rid)


def test_upload_evidence_identity_and_linkage(client_factory):
    """Each upload: own server UUID + sha256 of its own bytes; 1:1 linkage to its part."""
    with client_factory() as client:
        install_all(client.app.state.telltale)
        doc = create_mm(client, ("sensor", "acoustic", "bearing"))
        a, b = modal(doc, "acoustic"), modal(doc, "bearing")
        assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", a["recording_id"])
        assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", b["recording_id"])
        assert a["recording_id"] != b["recording_id"]
        assert a["recording_sha256"] == hashlib.sha256(WAV_A).hexdigest()
        assert b["recording_sha256"] == hashlib.sha256(WAV_B).hexdigest()
        # sanitized filename is display metadata only
        assert a["output"]["filename"] == "clip.wav" and b["output"]["filename"] == "clip.wav"
        # per-modality model/reference identity
        assert a["model_identity"]["artifact_fingerprint"] == FAKE_IDENTITY["artifact_fingerprint"]
        assert a["reference_id"] == "sha256:" + FAKE_IDENTITY["reference_fingerprint"]
        assert b["model_identity"]["artifact_fingerprint"] == FAKE_BEARING_IDENTITY["artifact_fingerprint"]
        assert b["reference_id"] == "sha256:" + FAKE_BEARING_IDENTITY["reference_fingerprint"]
        # sensor keeps its declared demo provenance; uploads are not illustrative
        assert modal(doc, "sensor")["source_mode"] == "simulated"
        assert modal(doc, "sensor")["illustrative"] is True
        assert a["source_mode"] == "uploaded" and a["illustrative"] is False
        assert b["source_mode"] == "uploaded" and b["illustrative"] is False


def test_max_rank_combinations(client_factory):
    """Overall status = conservative max-rank over available modalities (never a fusion)."""
    cases = [
        # sensor high + acoustic normal -> critical
        (SENSOR_HIGH, ACOUSTIC_NORMAL, None, ("sensor", "acoustic"), "critical", set()),
        # sensor low + acoustic anomalous -> critical
        (SENSOR_LOW, None, None, ("sensor", "acoustic"), "critical", set()),
        # sensor medium (watch) + acoustic normal -> watch
        (FAKE_RESULT, ACOUSTIC_NORMAL, None, ("sensor", "acoustic"), "watch", set()),
        # all normal -> normal
        (SENSOR_LOW, ACOUSTIC_NORMAL, FAKE_BEARING_RESULT, ("sensor", "acoustic", "bearing"), "normal", set()),
        # bearing fault set -> watch
        (SENSOR_LOW, ACOUSTIC_NORMAL, FAULT_RESULT, ("sensor", "acoustic", "bearing"), "watch", set()),
        # uncertain classifier -> unknown + flag
        (
            SENSOR_LOW,
            ACOUSTIC_NORMAL,
            UNCERTAIN_RESULT,
            ("sensor", "acoustic", "bearing"),
            "unknown",
            {"BEARING_UNCERTAIN"},
        ),
        # ambiguous normal-with-faults -> unknown + same conservative flag
        (
            SENSOR_LOW,
            ACOUSTIC_NORMAL,
            AMBIGUOUS_RESULT,
            ("sensor", "acoustic", "bearing"),
            "unknown",
            {"BEARING_UNCERTAIN"},
        ),
        # no sensor: bearing watch + acoustic normal -> watch
        (None, ACOUSTIC_NORMAL, FAULT_RESULT, ("acoustic", "bearing"), "watch", set()),
    ]
    with client_factory() as client:
        for sensor_res, acoustic_res, bearing_res, bits, expected, flags in cases:
            state = client.app.state.telltale
            install_fakes(
                state,
                FakeSensor(result=sensor_res) if sensor_res is not None else None,
                FakeAcoustic(result=acoustic_res),
                FakeBearing(result=bearing_res),
            )
            doc = create_mm(client, bits)
            assert doc["overall_status"] == expected, (bits, doc["overall_status"])
            assert flags <= flag_codes(doc), (bits, flag_codes(doc))
            assert doc["evidence_completeness"]["ratio"] == 1.0


# ---------------------------------------------------------------- provenance rules


def test_provenance_flags_rules(client_factory):
    """MIXED_SOURCE_PROVENANCE + SOURCE_ILLUSTRATIVE follow the existing rules."""
    with client_factory() as client:
        install_all(client.app.state.telltale)
        # sensor (simulated) + uploads: mixed + illustrative, assessment stays "uploaded"
        doc = create_mm(client, ("sensor", "acoustic", "bearing"))
        assert doc["source_mode"] == "uploaded"
        assert "MIXED_SOURCE_PROVENANCE" in flag_codes(doc)
        assert "SOURCE_ILLUSTRATIVE" in flag_codes(doc)
        # uploads only: no mixing, no illustrative
        doc2 = create_mm(client, ("acoustic", "bearing"))
        assert doc2["source_mode"] == "uploaded"
        assert "MIXED_SOURCE_PROVENANCE" not in flag_codes(doc2)
        assert "SOURCE_ILLUSTRATIVE" not in flag_codes(doc2)
        # other declared legacy demo mode: still illustrative + mixed
        fields, files = bits_parts(("sensor", "bearing"))
        fields[1] = ("sensor_json", SENSOR_JSON_DEMO)
        resp = post_mm(client, fields=fields, files=files)
        assert resp.status_code == 201, resp.text
        doc3 = resp.json()
        assert doc3["source_mode"] == "uploaded"
        assert modal(doc3, "sensor")["source_mode"] == "dataset_demo"
        assert "SOURCE_ILLUSTRATIVE" in flag_codes(doc3)
        assert "MIXED_SOURCE_PROVENANCE" in flag_codes(doc3)


# ---------------------------------------------------------------- timestamp semantics


def test_observed_at_applies_to_every_modality_and_evidence(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        doc = create_mm(client, ("sensor", "acoustic", "bearing"), observed_at="2026-10-14T16:00:00+05:00")
        want = "2026-10-14T11:00:00Z"  # normalized to UTC (pydantic JSON renders UTC as Z)
        for m in doc["modalities"]:
            assert m["observed_at"] == want
        for e in doc["evidence"]:
            assert e["observed_at"] == want
        assert "TIMESTAMP_MISSING" not in flag_codes(doc)
        assert doc["created_at"] >= max(m["inference_at"] for m in doc["modalities"])
        assert max(m["inference_at"] for m in doc["modalities"]) > want


def test_missing_observed_at_flag(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        doc = create_mm(client, ("sensor", "acoustic"))
        assert all(m["observed_at"] is None for m in doc["modalities"])
        assert all(e["observed_at"] is None for e in doc["evidence"])
        assert "TIMESTAMP_MISSING" in flag_codes(doc)


def test_naive_observed_at_rejected(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        resp = post_mm(client)
        assert resp.status_code == 201  # sanity: default request is fine
        fields, files = bits_parts(("sensor", "acoustic"), observed_at="2026-10-14T16:00:00")
        resp = post_mm(client, fields=fields, files=files)
        body = assert_error(resp, 422, "request_validation")
        assert body["error"]["details"]["param"] == "observed_at"
        assert len(rows(client)) == 1


# ---------------------------------------------------------------- ownership, immutability, restart


def test_owner_isolation(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        doc = create_mm(client, ("sensor", "acoustic", "bearing"))
        aid = doc["assessment_id"]
        assert client.get(f"{URL_BASE}/{aid}", headers=AUTH_BOB).status_code == 404
        assert client.get(URL_BASE, headers=AUTH_BOB).json()["items"] == []
        assert [d["assessment_id"] for d in rows(client)] == [aid]
        # creating under a second owner keeps the first owner's list unchanged
        fields, files = bits_parts(("acoustic", "bearing"))
        resp = post_mm(client, fields=fields, files=files, auth=AUTH_BOB)
        assert resp.status_code == 201, resp.text
        bob_id = resp.json()["assessment_id"]
        assert [d["assessment_id"] for d in rows(client)] == [aid]
        assert [d["assessment_id"] for d in client.get(URL_BASE, headers=AUTH_BOB).json()["items"]] == [bob_id]


def test_documents_immutable_and_restart_persistent(client_factory, tmp_path):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        first = create_mm(client, ("sensor", "acoustic", "bearing"))
        # a later create, a rejected create and a failed create never touch stored rows
        create_mm(client, ("acoustic", "bearing"))
        post_mm(client, fields=bits_parts(("sensor",))[0], files=[])  # too few modalities
        stub = install_all(client.app.state.telltale, sensor={"error": RuntimeError("boom")})[0]
        assert post_mm(client, fields=bits_parts(("sensor", "bearing"))[0], files=bits_parts(("sensor", "bearing"))[1]).status_code == 500
        stub.error = None
        got = client.get(f"{URL_BASE}/{first['assessment_id']}", headers=AUTH_ALICE)
        assert got.status_code == 200 and got.json() == first
    # restart with the same data dir: rows survive byte-identically
    with client_factory(data_dir=tmp_path) as client2:
        got = client2.get(f"{URL_BASE}/{first['assessment_id']}", headers=AUTH_ALICE)
        assert got.status_code == 200 and got.json() == first
        assert len(client2.get(URL_BASE, headers=AUTH_ALICE).json()["items"]) == 2


def test_auth_before_parsing_multipart(client_factory):
    with client_factory() as client:
        # unauthorized callers never get their body parsed, even a garbage one
        body = b"this-is-not-even-multipart"
        resp = client.post(URL_M, content=body, headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert_error(resp, 401, "unauthorized")
        resp = post_mm(client, auth=None)
        assert_error(resp, 401, "unauthorized")
        resp = post_mm(client, auth={"Authorization": "Bearer wrong-token"})
        assert_error(resp, 401, "unauthorized")
        assert rows(client) == []


# ---------------------------------------------------------------- strict multipart boundary


def test_unknown_and_forged_form_fields_rejected(client_factory):
    forged = [
        "score", "status", "overall_status", "owner", "expected_modalities", "source_mode",
        "machine_type", "assessment_id", "model_identity", "recording_sha256", "reference_id", "rpm",
    ]
    with client_factory() as client:
        install_all(client.app.state.telltale)
        for name in forged:
            fields, files = bits_parts(("sensor", "acoustic"))
            fields.append((name, "forged"))
            body = assert_error(post_mm(client, fields=fields, files=files), 422, "request_validation")
            assert name in json.dumps(body["error"]["details"]), name
        # legacy "file" part name is unknown on this route
        resp = post_mm(client, fields=bits_parts(("sensor",))[0], files=[("file", ("clip.wav", WAV_A, "audio/wav"))])
        body = assert_error(resp, 422, "request_validation")
        assert "file" in json.dumps(body["error"]["details"])
        assert rows(client) == []


def test_duplicate_fields_and_duplicate_file_parts_rejected(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        # duplicate text part
        body = assert_error(
            post_raw(
                client,
                [("machine_id", MACHINE), ("machine_id", MACHINE), ("sensor_json", SENSOR_JSON)],
                [("acoustic_file", ("clip.wav", WAV_A, "audio/wav"))],
            ),
            422,
            "request_validation",
        )
        assert "machine_id" in json.dumps(body["error"]["details"])
        # duplicate sensor_json part
        body = assert_error(
            post_raw(
                client,
                [("machine_id", MACHINE), ("sensor_json", SENSOR_JSON), ("sensor_json", SENSOR_JSON)],
                [("acoustic_file", ("clip.wav", WAV_A, "audio/wav"))],
            ),
            422,
            "request_validation",
        )
        assert "sensor_json" in json.dumps(body["error"]["details"])
        # duplicate FILE part (duplicate modality upload)
        body = assert_error(
            post_raw(
                client,
                [("machine_id", MACHINE), ("sensor_json", SENSOR_JSON)],
                [
                    ("acoustic_file", ("a.wav", WAV_A, "audio/wav")),
                    ("acoustic_file", ("b.wav", WAV_B, "audio/wav")),
                ],
            ),
            422,
            "request_validation",
        )
        assert "acoustic_file" in json.dumps(body["error"]["details"])
        assert rows(client) == []


def test_part_type_discipline(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        # sensor_json must be a text part, not a file
        body = assert_error(
            post_raw(
                client,
                [("machine_id", MACHINE)],
                [
                    ("sensor_json", ("sensor.json", b"{}", "application/json")),
                    ("acoustic_file", ("clip.wav", WAV_A, "audio/wav")),
                ],
            ),
            422,
            "request_validation",
        )
        assert body["error"]["details"]["param"] == "sensor_json"
        # audio parts must be files, not text
        body = assert_error(
            post_raw(
                client,
                [("machine_id", MACHINE), ("sensor_json", SENSOR_JSON), ("acoustic_file", "not-a-file")],
                [("bearing_file", ("clip.wav", WAV_B, "audio/wav"))],
            ),
            422,
            "request_validation",
        )
        assert body["error"]["details"]["param"] == "acoustic_file"
        # missing machine_id
        body = assert_error(post_mm(client, fields=[("sensor_json", SENSOR_JSON)], files=[]), 422, "request_validation")
        assert body["error"]["details"]["param"] == "machine_id"
        assert rows(client) == []


def test_too_many_parts_rejected(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        # three file parts: parser max_files=2 -> malformed/oversized, never silently dropped
        fields, _ = bits_parts(("sensor", "acoustic", "bearing"))
        assert_error(
            post_raw(
                client,
                fields,
                [
                    ("acoustic_file", ("a.wav", WAV_A, "audio/wav")),
                    ("bearing_file", ("b.wav", WAV_B, "audio/wav")),
                    ("extra_file", ("c.wav", WAV_B, "audio/wav")),
                ],
            ),
            422,
            "request_validation",
        )
        # five text parts: parser max_fields=4
        assert_error(
            post_raw(
                client,
                [
                    ("machine_id", MACHINE),
                    ("sensor_json", SENSOR_JSON),
                    ("observed_at", "2026-10-14T11:00:00Z"),
                    ("junk1", "x"),
                    ("junk2", "y"),
                ],
                [("acoustic_file", ("clip.wav", WAV_A, "audio/wav"))],
            ),
            422,
            "request_validation",
        )
        assert rows(client) == []


# ---------------------------------------------------------------- same-part rule + identical bytes


def test_one_part_never_serves_two_modalities(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        # sensor + acoustic_file: bearing is never inferred from the acoustic upload
        doc = create_mm(client, ("sensor", "acoustic"))
        assert doc["expected_modalities"] == ["sensor", "acoustic"]
        assert [m["kind"] for m in doc["modalities"]] == ["sensor", "acoustic"]
        assert modal(doc, "sensor")["recording_id"] is None
        # sensor + bearing_file: acoustic is never inferred from the bearing upload
        doc = create_mm(client, ("sensor", "bearing"))
        assert doc["expected_modalities"] == ["sensor", "bearing"]
        # a single upload alone is one modality -> rejected, not double-counted
        assert_error(
            post_mm(client, fields=bits_parts(("acoustic",))[0], files=bits_parts(("acoustic",))[1]),
            422,
            "too_few_modalities",
        )
        assert_error(
            post_mm(client, fields=bits_parts(("bearing",))[0], files=bits_parts(("bearing",))[1]),
            422,
            "too_few_modalities",
        )


def test_identical_bytes_in_distinct_parts_allowed(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        fields, files = bits_parts(("sensor", "acoustic", "bearing"), bearing=WAV_A)  # same raw bytes
        resp = post_mm(client, fields=fields, files=files)
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        a, b = modal(doc, "acoustic"), modal(doc, "bearing")
        assert a["recording_sha256"] == b["recording_sha256"] == hashlib.sha256(WAV_A).hexdigest()
        assert a["recording_id"] != b["recording_id"]  # separate evidence records
        assert a["model_identity"]["training_dataset"] != b["model_identity"]["training_dataset"]


def test_fewer_than_two_modalities_rejected(client_factory):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        for bits in ((), ("sensor",), ("acoustic",), ("bearing",)):
            fields, files = bits_parts(bits)
            body = assert_error(post_mm(client, fields=fields, files=files), 422, "too_few_modalities")
            assert body["error"]["details"]["present"] == [k for k in ("sensor", "acoustic", "bearing") if k in bits]
        assert rows(client) == []


# ---------------------------------------------------------------- machine compatibility matrix


def test_unknown_machine_rejected_before_inference(client_factory):
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        fields, files = bits_parts(("sensor", "acoustic", "bearing"))
        fields[0] = ("machine_id", "ghost-99")
        body = assert_error(post_mm(client, fields=fields, files=files), 422, "unknown_machine")
        assert body["error"]["details"]["machine_id"] == "ghost-99"
        assert MACHINE not in json.dumps(body["error"]["details"])  # no fleet inventory leak
        # matrix runs before inference: nothing was scored
        assert [s.calls for s in stubs] == [[], [], []]
        assert rows(client) == []


def test_acoustic_null_profile_mismatch(client_factory):
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        machine = client.app.state.telltale.fleet.machines[MACHINE]
        saved = machine.spec.acoustic_profile
        machine.spec.acoustic_profile = None
        try:
            body = assert_error(
                post_mm(client, fields=bits_parts(("sensor", "acoustic"))[0], files=bits_parts(("sensor", "acoustic"))[1]),
                422,
                "acoustic_profile_mismatch",
            )
            assert body["error"]["details"]["machine_id"] == MACHINE
            assert [s.calls for s in stubs] == [[], [], []]
            assert rows(client) == []
        finally:
            machine.spec.acoustic_profile = saved


def test_missing_models_dependency_errors(client_factory):
    with client_factory() as client:
        # sensor model missing
        state = client.app.state.telltale
        stubs = install_all(state)
        state.registry.sensor = None
        body = assert_error(
            post_mm(client, fields=bits_parts(("sensor", "acoustic"))[0], files=bits_parts(("sensor", "acoustic"))[1]),
            503,
            "dependency_unavailable",
        )
        assert "telltale train-sensor" in body["error"]["message"]
        assert [s.calls for s in stubs] == [[], [], []]
        # no sound model at all
        state = client.app.state.telltale
        install_fakes(state, stubs[0], None, stubs[2])
        state.registry.acoustic = {}
        state.registry.acoustic_identity = {}
        body = assert_error(
            post_mm(client, fields=bits_parts(("sensor", "acoustic"))[0], files=bits_parts(("sensor", "acoustic"))[1]),
            503,
            "dependency_unavailable",
        )
        assert "telltale train-sound" in body["error"]["message"]
        # profile exists but its model is not loaded -> legacy 404 dependency_error
        state.registry.acoustic = {"pump": FakeAcoustic()}
        state.registry.acoustic_identity = {"pump": dict(FAKE_IDENTITY)}
        body = assert_error(
            post_mm(client, fields=bits_parts(("sensor", "acoustic"))[0], files=bits_parts(("sensor", "acoustic"))[1]),
            404,
            "dependency_error",
        )
        assert "fan" in body["error"]["message"]
        # bearing model missing
        install_fakes(state, stubs[0], stubs[1], None)
        state.registry.bearing = None
        body = assert_error(
            post_mm(client, fields=bits_parts(("sensor", "bearing"))[0], files=bits_parts(("sensor", "bearing"))[1]),
            503,
            "dependency_unavailable",
        )
        assert "telltale train-bearing" in body["error"]["message"]
        assert rows(client) == []


def test_profile_selects_model_dynamic_machine(client_factory):
    """Acoustic follows the machine's DECLARED profile (lth-b1 declared 'pump', no pump model)."""
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        fields, files = bits_parts(("sensor", "acoustic"))
        fields[0] = ("machine_id", "lth-b1")
        body = assert_error(post_mm(client, fields=fields, files=files), 404, "dependency_error")
        assert "pump" in body["error"]["message"]
        assert [s.calls for s in stubs] == [[], [], []]


def test_enrolled_sound_model_refused(client_factory):
    with client_factory() as client:
        stubs = install_fakes(
            client.app.state.telltale,
            FakeSensor(),
            FakeAcoustic(enrolled={"line-a": {"views": {}, "calibration": []}}),
            FakeBearing(),
        )
        body = assert_error(
            post_mm(client, fields=bits_parts(("sensor", "acoustic"))[0], files=bits_parts(("sensor", "acoustic"))[1]),
            409,
            "enrolled_reference_active",
        )
        assert "line-a" in json.dumps(body["error"])
        assert [s.calls for s in stubs] == [[], [], []]


# ---------------------------------------------------------------- upload safety


def test_corrupt_and_empty_acoustic_rejected(client_factory):
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        # no sensor bit: the acoustic boundary fails before ANY inference runs
        for payload in (b"garbage-not-audio" * 500, b""):
            body = assert_error(
                post_mm(
                    client,
                    fields=bits_parts(("acoustic", "bearing"))[0],
                    files=[
                        ("acoustic_file", ("clip.wav", payload, "audio/wav")),
                        ("bearing_file", ("clip.wav", WAV_B, "audio/wav")),
                    ],
                ),
                422,
                "invalid_audio",
            )
            assert "libsndfile" not in json.dumps(body)  # decoder internals never leak
        assert [s.calls for s in stubs] == [[], [], []]
        assert rows(client) == []


def test_oversized_uploads_rejected(client_factory):
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        big = b"x" * (26 * 1024 * 1024 + 4096)  # 27.3 MB single upload: over the 25 MB limit
        body = assert_error(
            post_mm(
                client,
                fields=bits_parts(("sensor",))[0],
                files=[("acoustic_file", ("clip.wav", big, "audio/wav"))],
            ),
            413,
            "file_too_large",
        )
        assert "25 MB" in body["error"]["message"]
        # oversized Bearing upload (acoustic already read): same rejection, zero rows
        fields, files = bits_parts(("sensor", "acoustic", "bearing"))
        files[1] = ("bearing_file", ("clip.wav", big, "audio/wav"))
        assert_error(post_mm(client, fields=fields, files=files), 413, "file_too_large")
        assert [s.calls for s in stubs] == [[], [], []]
        assert rows(client) == []


def test_too_short_uploads_rejected(client_factory):
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        # acoustic: deployed MIN_SECONDS rule (0.5 s); bearing never scored after it
        body = assert_error(
            post_mm(
                client,
                fields=bits_parts(("acoustic", "bearing"))[0],
                files=[
                    ("acoustic_file", ("clip.wav", wav_bytes(seconds=0.3), "audio/wav")),
                    ("bearing_file", ("clip.wav", WAV_B, "audio/wav")),
                ],
            ),
            422,
            "invalid_audio",
        )
        assert "0.5 s" in body["error"]["message"]
        # bearing: deployed segment_seconds rule (1.0 s); classifier never ran
        body = assert_error(
            post_mm(
                client,
                fields=bits_parts(("acoustic", "bearing"))[0],
                files=[
                    ("acoustic_file", ("clip.wav", WAV_A, "audio/wav")),
                    ("bearing_file", ("clip.wav", wav_bytes(seconds=0.3), "audio/wav")),
                ],
            ),
            422,
            "invalid_audio",
        )
        assert "1.0 s" in body["error"]["message"]
        assert stubs[2].calls == []
        assert rows(client) == []


def test_non_finite_uploads_rejected(client_factory):
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        nan_wav = wav_bytes(seconds=2.0, with_nan=True)
        assert_error(
            post_mm(
                client,
                fields=bits_parts(("acoustic", "bearing"))[0],
                files=[
                    ("acoustic_file", ("clip.wav", nan_wav, "audio/wav")),
                    ("bearing_file", ("clip.wav", WAV_B, "audio/wav")),
                ],
            ),
            422,
            "invalid_audio",
        )
        assert_error(
            post_mm(
                client,
                fields=bits_parts(("acoustic", "bearing"))[0],
                files=[
                    ("acoustic_file", ("clip.wav", WAV_A, "audio/wav")),
                    ("bearing_file", ("clip.wav", nan_wav, "audio/wav")),
                ],
            ),
            422,
            "invalid_audio",
        )
        assert stubs[2].calls == []  # classifier never scored a non-finite upload
        assert rows(client) == []


def test_acoustic_matlab_rejected_but_bearing_accepts_mat(client_factory):
    with client_factory() as client:
        stubs = install_all(client.app.state.telltale)
        mat = mat_bytes()
        # acoustic keeps the slice-2 rule: MATLAB is refused, even inside a .wav name
        for name, payload in (("clip.mat", mat), ("clip.wav", b"MATLAB 5.0 MAT-file" + b"\x00" * 64), ("clip.txt", WAV_A)):
            assert_error(
                post_mm(
                    client,
                    fields=bits_parts(("acoustic", "bearing"))[0],
                    files=[
                        ("acoustic_file", (name, payload, "audio/wav")),
                        ("bearing_file", ("clip.wav", WAV_B, "audio/wav")),
                    ],
                ),
                422,
                "invalid_audio",
            )
        assert [s.calls for s in stubs] == [[], [], []]  # boundary rules fire before inference
        # bearing keeps the original path rule: decode success only, .mat IS accepted
        doc = create_mm(client, ("sensor", "acoustic", "bearing"), bearing=mat)
        assert modal(doc, "bearing")["output"]["filename"] == "clip.wav"
        assert stubs[2].calls  # classifier really ran with the .mat upload


# ---------------------------------------------------------------- inference + storage failures


def test_sensor_inference_failure_sanitized_no_row(client_factory):
    with client_factory() as client:
        sensor_stub, acoustic_stub, bearing_stub = install_all(
            client.app.state.telltale, sensor={"error": RuntimeError("boom at C:\\secret\\model.txt")}
        )
        body = assert_error(
            post_mm(client, fields=bits_parts(("sensor", "acoustic"))[0], files=bits_parts(("sensor", "acoustic"))[1]),
            500,
            "internal_error",
        )
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert "secret" not in body["error"]["message"] and "Traceback" not in body["error"]["message"]
        # fixed order: the failure aborts before the later modalities run
        assert sensor_stub.calls and acoustic_stub.calls == [] and bearing_stub.calls == []
        assert rows(client) == []


def test_audio_inference_failures_sanitized_no_row(client_factory):
    with client_factory() as client:
        acoustic_stub, bearing_stub = install_all(
            client.app.state.telltale,
            acoustic={"error": RuntimeError("onnx session died")},
            bearing={"error": RuntimeError("lightgbm died")},
        )[1:]
        body = assert_error(
            post_mm(client, fields=bits_parts(("acoustic",))[0], files=bits_parts(("acoustic", "bearing"))[1]),
            500,
            "internal_error",
        )
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert "died" not in json.dumps(body)
        assert acoustic_stub.calls and bearing_stub.calls == []  # failure aborts later modalities
        assert rows(client) == []
        # classifier failure -> same sanitized 500, nothing saved (acoustic healthy again)
        acoustic_stub.error = None
        assert_error(
            post_mm(client, fields=bits_parts(("acoustic", "bearing"))[0], files=bits_parts(("acoustic", "bearing"))[1]),
            500,
            "internal_error",
        )
        assert bearing_stub.calls and rows(client) == []


def test_audio_caused_valueerror_maps_to_422(client_factory):
    with client_factory() as client:
        install_all(
            client.app.state.telltale,
            acoustic={"error": ValueError("too weird")},
            bearing={"error": ValueError("Audio too short; need at least 1.0 s")},
        )
        # detector-side validation -> 422 invalid_audio (audio problem, not inference)
        assert_error(
            post_mm(client, fields=bits_parts(("acoustic", "bearing"))[0], files=bits_parts(("acoustic", "bearing"))[1]),
            422,
            "invalid_audio",
        )
        # classifier-side validation -> 422 invalid_audio
        assert_error(
            post_mm(client, fields=bits_parts(("sensor", "bearing"))[0], files=bits_parts(("sensor", "bearing"))[1]),
            422,
            "invalid_audio",
        )
        assert rows(client) == []


def test_storage_failures_produce_zero_rows(client_factory, monkeypatch, tmp_path):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        store = client.app.state.telltale.assessments
        assert store is not None

        def boom(*args, **kwargs):
            raise RuntimeError("insert failed at C:\\secret\\telltale.db")

        monkeypatch.setattr(store, "insert", boom)
        body = assert_error(post_mm(client), 500, "internal_error")
        assert body["error"]["message"] == "Unexpected server error; nothing was saved."
        assert "secret" not in body["error"]["message"]
        assert rows(client) == []
        monkeypatch.undo()
        # storage unavailable entirely -> 503; count rows straight from SQLite (the
        # list route would itself be unavailable), nothing was ever written.
        client.app.state.telltale.assessments = None
        assert_error(post_mm(client), 503, "storage_unavailable")
        with sqlite3.connect(tmp_path / "state" / "telltale.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0] == 0


# ---------------------------------------------------------------- resource cleanup + retention


def test_upload_resources_closed_on_every_path(client_factory, monkeypatch):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        closed: list[bool] = []
        original = assessments_routes._bounded_upload_bytes

        async def spy(upload, limit):
            try:
                return await original(upload, limit)
            finally:
                closed.append(upload.file.closed)

        monkeypatch.setattr(assessments_routes, "_bounded_upload_bytes", spy)
        # success path: both uploads read and closed
        create_mm(client, ("sensor", "acoustic", "bearing"))
        assert closed == [True, True], closed
        closed.clear()
        # boundary failure inside the threadpool decoder (non-finite): both reads closed
        nan_wav = wav_bytes(seconds=2.0, with_nan=True)
        assert_error(
            post_mm(
                client,
                fields=bits_parts(("acoustic", "bearing"))[0],
                files=[
                    ("acoustic_file", ("clip.wav", WAV_A, "audio/wav")),
                    ("bearing_file", ("clip.wav", nan_wav, "audio/wav")),
                ],
            ),
            422,
            "invalid_audio",
        )
        assert closed == [True, True], closed
        monkeypatch.undo()
        assert len(rows(client)) == 1  # the success persisted, the failure left nothing


def test_hash_only_retention(client_factory, tmp_path):
    with client_factory() as client:
        install_all(client.app.state.telltale)
        doc = create_mm(client, ("sensor", "acoustic", "bearing"))
        a, b = modal(doc, "acoustic"), modal(doc, "bearing")
        db = tmp_path / "state" / "telltale.db"
        assert db.is_file()
        raw = db.read_bytes()
        for m in (a, b):
            assert m["recording_sha256"].encode() in raw  # hash identity IS retained
        # no recording bytes, and no audio file anywhere under the data dir
        assert b"RIFF" not in raw and WAV_A[:16] not in raw and b"MATLAB" not in raw
        audio_files = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".ogg", ".mat"}]
        assert audio_files == []


# ---------------------------------------------------------------- legacy routes unchanged


def test_legacy_routes_unchanged(client_factory):
    with client_factory() as client:
        # legacy sensor route (JSON body)
        resp = client.post(
            URL_BASE,
            json={"reading": READING, "machine_id": MACHINE, "source_mode": "simulated"},
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        legacy = resp.json()
        assert legacy["expected_modalities"] == ["sensor"]
        assert legacy["source_mode"] == "simulated"
        assert "MIXED_SOURCE_PROVENANCE" not in flag_codes(legacy)
        # legacy acoustic route (single file part, real fan fixture model)
        resp = client.post(
            URL_BASE + "/acoustic",
            data={"machine_id": MACHINE},
            files=[("file", ("clip.wav", WAV_A, "audio/wav"))],
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["acoustic"]
        # legacy bearing route (single file part, real fixture classifier)
        resp = client.post(
            URL_BASE + "/bearing",
            data={"machine_id": MACHINE},
            files=[("file", ("clip.wav", WAV_B, "audio/wav"))],
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["bearing"]
        # the new route then works on the same live app and does not disturb them
        doc = create_mm(client, ("sensor", "acoustic", "bearing"))
        assert doc["expected_modalities"] == ["sensor", "acoustic", "bearing"]
        assert len(rows(client)) == 4


def test_processing_cap_disclosure_flag(client_factory):
    """Existing QualityCode, one flag, per-modality numbers; original+processed kept."""
    with client_factory() as client:
        acoustic = FakeAcoustic(result={**FAKE_ACOUSTIC_RESULT, "duration_s": 1.0})
        acoustic.metadata["max_audio_seconds"] = 1.0
        bearing = FakeBearing(result={**FAKE_BEARING_RESULT, "duration_s": 1.0})
        bearing.metadata["max_audio_seconds"] = 1.0
        install_fakes(client.app.state.telltale, None, acoustic, bearing)
        doc = create_mm(client, ("acoustic", "bearing"))  # 2.0 s uploads, 1.0 s scored
        flags = [f for f in doc["quality_flags"] if f["code"] == "AUDIO_TRUNCATED"]
        assert len(flags) == 1
        assert "acoustic" in flags[0]["explanation"] and "bearing" in flags[0]["explanation"]
        pairs = {(e["source"], e["kind"]): e["value"] for e in doc["evidence"]}
        assert pairs[("acoustic", "audio_duration_original_s")] == 2.0
        assert pairs[("acoustic", "audio_duration_processed_s")] == 1.0
        assert pairs[("bearing", "audio_duration_original_s")] == 2.0
        assert pairs[("bearing", "audio_duration_processed_s")] == 1.0


def test_real_fixture_models_api_vs_direct(client_factory, models_dir):
    """The combined document must equal DIRECT calls of the same deployed fixture
    models on the same bytes (bit-identical values, no re-derivation). This is the
    only test using the real models: every other test injects deterministic fakes."""
    with client_factory() as client:
        state = client.app.state.telltale
        predictor = state.registry.sensor
        detector = state.registry.acoustic.get("fan")
        classifier = state.registry.bearing
        assert predictor is not None and detector is not None and classifier is not None
        fields, files = bits_parts(("sensor", "acoustic", "bearing"))
        resp = post_mm(client, fields=fields, files=files)
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        by = {m["kind"]: m for m in doc["modalities"]}

        direct_s = predictor.assess([dict(READING)])[0]
        out_s = by["sensor"]["output"]  # no filename is added to the sensor output
        for key in ("failure_probability", "risk_level", "decision_threshold", "failure_modes"):
            assert out_s[key] == direct_s[key], key

        direct_a = detector.analyze(io.BytesIO(WAV_A))
        out_a = by["acoustic"]["output"]
        for key in (
            "score",
            "threshold",
            "severity",
            "p_value",
            "unusual_percentile",
            "calibration",
            "status",
            "is_anomalous",
            "duration_s",
            "sample_rate",
            "peak_time_s",
        ):
            assert out_a[key] == direct_a[key], key
        assert out_a["filename"] == "clip.wav"  # only addition: sanitized display metadata

        reading = state.fleet.machines[MACHINE].reading or {}
        speed = reading.get("rotational_speed_rpm")
        rpm = float(speed) if speed is not None else None
        direct_b = classifier.diagnose(io.BytesIO(WAV_B), rpm)
        out_b = by["bearing"]["output"]
        for key in (
            "predicted_class",
            "label",
            "confidence",
            "prediction_set",
            "prediction_set_labels",
            "coverage",
            "uncertain",
            "rpm_used",
            "rpm_assumed",
            "segment_agreement",
            "segments",
            "duration_s",
            "probabilities",
        ):
            assert out_b[key] == direct_b[key], key
        assert out_b["filename"] == "clip.wav"

        # identity of every deployed artifact on disk
        ident_a = state.registry.acoustic_identity["fan"]
        assert by["acoustic"]["model_identity"]["artifact_fingerprint"] == ident_a["artifact_fingerprint"]
        assert ident_a["artifact_fingerprint"] == fingerprint_dir(models_dir / "acoustic" / "fan")
        assert by["acoustic"]["reference_id"] == "sha256:" + ident_a["reference_fingerprint"]
        assert ident_a["preprocessing_fingerprint"] is not None
        ident_b = state.registry.bearing_identity
        assert ident_b is not None
        assert by["bearing"]["model_identity"]["artifact_fingerprint"] == ident_b["artifact_fingerprint"]
        assert ident_b["artifact_fingerprint"] == fingerprint_dir(models_dir / "bearing")
        assert by["bearing"]["reference_id"] == "sha256:" + ident_b["reference_fingerprint"]
        assert ident_b["preprocessing_fingerprint"] is not None
        # hash identity of each uploaded part, linked 1:1
        assert by["acoustic"]["recording_sha256"] == hashlib.sha256(WAV_A).hexdigest()
        assert by["bearing"]["recording_sha256"] == hashlib.sha256(WAV_B).hexdigest()
