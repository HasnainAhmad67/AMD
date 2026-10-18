"""Endpoint slice-3 tests: trusted BEARING assessment creation (multipart upload).

Covers the approved contract: strict multipart boundary (unknown/duplicate/forged
parts rejected, auth before body parsing), bearing-only machine scope (no fan
fallback, no dynamically enabled other types, no fused sensor+acoustic
probability), upload safety (bounded reads; decode success is the ONLY format
rule - .mat is this dataset's native format and IS accepted, unlike the fan
slice; corrupt/empty/too-short/non-finite rejected; the DEPLOYED segment-duration
rule; 60 s processing-cap disclosure), hash-only recording provenance, model +
preprocessing + calibration identity (deterministic and reload-consistent),
conservative assess/v1 bearing status (never critical), failure atomicity (no
row on any failure) and resource cleanup. Legacy /acoustic/diagnose behaviour is
asserted unchanged.

Mock-based tests inject a deterministic FakeBearing classifier; the API/direct
comparison test runs the REAL fixture-trained tiny bearing model from conftest.
No downloads, no GPUs, no network: everything runs on the validated venv with all
temporary state on D: (TEMP/TMP pointed there by the test run).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import re
import sqlite3
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import scipy.io as sio
import soundfile as sf
from fastapi.testclient import TestClient

from telltale_api.api.routes.assessments import _bounded_upload_bytes
from telltale_api.core.config import Settings
from telltale_api.main import create_app
from telltale_api.services.registry import ModelRegistry, fingerprint_dir

URL_B = "/api/v1/assessments/bearing"
URL_BASE = "/api/v1/assessments"

TOKENS = json.dumps({"alice": "tok-alice-9f3d2c7e", "bob": "tok-bob-77aa11bb"})
AUTH_ALICE = {"Authorization": "Bearer tok-alice-9f3d2c7e"}
AUTH_BOB = {"Authorization": "Bearer tok-bob-77aa11bb"}

UPLOAD_LIMIT_MB = 25  # Settings.max_upload_mb default

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
    "probabilities": [
        {"class_name": "inner_race", "label": "Inner race defect", "probability": 0.61},
        {"class_name": "normal", "label": "Healthy bearing", "probability": 0.39},
    ],
}

UNCERTAIN_RESULT = {
    **FAKE_BEARING_RESULT,
    "uncertain": True,
    "prediction_set": ["normal", "inner_race"],
    "prediction_set_labels": ["Healthy bearing", "Inner race defect"],
}

# conformal set contains faults although "normal" is predicted: ambiguous, not healthy
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


def wav_bytes(
    seconds: float = 2.0, sr: int = 16000, amplitude: float = 0.05, with_nan: bool = False
) -> bytes:
    n = max(1, int(round(seconds * sr)))
    y = np.full(n, amplitude, dtype=np.float32)
    subtype = "PCM_16"
    if with_nan:
        y = y.astype(np.float64)
        y[0] = np.nan
        subtype = "FLOAT"  # keeps the NaN sample exactly
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype=subtype, format="WAV")  # BytesIO has no extension to infer from
    return buf.getvalue()


def mat_bytes(seconds: float = 2.0, sr: int = 44100, freq: float = 900.0) -> bytes:
    """A valid MATLAB .mat recording - the bearing dataset's own container format."""
    t = np.arange(int(sr * seconds)) / sr
    buf = io.BytesIO()
    sio.savemat(buf, {"x": (0.4 * np.sin(2 * np.pi * freq * t))[:, None]})
    return buf.getvalue()


def manifest_hash(files) -> str:
    """Independent replication of the registry manifest format (no shared code)."""
    lines = []
    for f in sorted(files, key=lambda p: p.name):
        lines.append(f"{f.name} {hashlib.sha256(f.read_bytes()).hexdigest()}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


@pytest.fixture()
def client_factory(models_dir, tmp_path):
    """Build a TestClient over the session fixture models (caller enters the context)."""

    def _make(**over):
        over.setdefault("api_tokens", TOKENS)
        settings = Settings(
            models_dir=over.pop("models_dir", models_dir),
            data_dir=over.pop("data_dir", tmp_path),
            simulator_enabled=True,
            simulator_interval_s=3600,
            llm_base_url=None,
            **over,
        )
        return TestClient(create_app(settings))

    return _make


def install_fake(state, stub: FakeBearing | None = None, identity: dict | None = None) -> FakeBearing:
    stub = stub or FakeBearing()
    state.registry.bearing = stub
    state.registry.bearing_identity = dict(identity) if identity else dict(FAKE_BEARING_IDENTITY)
    return stub


def post_audio(client, *, fields=None, files=None, auth=AUTH_ALICE, extra_headers=None):
    data = {"machine_id": "cnc-a1"} if fields is None else fields
    if files is None:
        files = [("file", ("clip.wav", wav_bytes(), "audio/wav"))]
    headers = dict(auth) if auth else {}
    headers.update(extra_headers or {})
    return client.post(URL_B, data=data, files=files, headers=headers)


def raw_multipart(fields, files, boundary: str = "----telltale-raw-b1"):
    """Hand-crafted multipart body: exact control over duplicate/extra parts
    (httpx's data+files combination silently normalizes list-of-tuples data)."""
    out = bytearray()
    for name, value in fields:
        out += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        ).encode()
    for name, filename, payload, ctype in files:
        out += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        out += payload + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def post_raw(client, fields, files, auth=AUTH_ALICE):
    body, ctype = raw_multipart(fields, files)
    return client.post(URL_B, content=body, headers={**auth, "Content-Type": ctype})


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


def create_ok(client, wav: bytes | None = None, **kw) -> tuple[dict, bytes]:
    wav = wav if wav is not None else wav_bytes()
    files = [("file", ("clip.wav", wav, "audio/wav"))]
    resp = post_audio(client, files=files, **kw)
    assert resp.status_code == 201, resp.text
    return resp.json(), wav


def modal(doc: dict) -> dict:
    return doc["modalities"][0]


def flag_codes(doc: dict) -> set[str]:
    return {f["code"] for f in doc["quality_flags"]}


def evidence_kinds(doc: dict) -> list[str]:
    return [e["kind"] for e in doc["evidence"]]


# ---------------------------------------------------------------- happy path (mock)


def test_create_ok_stub(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        stub = install_fake(state)
        wav = wav_bytes()
        resp = post_audio(client, files=[("file", ("clip.wav", wav, "audio/wav"))])
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        assert resp.headers["Location"] == f"/api/v1/assessments/{doc['assessment_id']}"
        assert re.fullmatch(r"[0-9a-f]{32}", resp.headers["X-Request-ID"])
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", doc["assessment_id"]
        )
        m = modal(doc)
        # bearing-only server profile + uploaded provenance
        assert doc["expected_modalities"] == ["bearing"]
        assert doc["source_mode"] == "uploaded"
        assert doc["evidence_completeness"] == {"expected": 1, "present": 1, "ratio": 1.0}
        assert m["kind"] == "bearing"
        assert m["source_mode"] == "uploaded"
        assert m["illustrative"] is False  # uploaded is NOT automatically illustrative
        assert m["availability"] == "available" and m["reason_code"] == "ok"
        # assess/v1 conservative bearing mapping: normal only (never critical)
        assert doc["overall_status"] == "normal"
        # typed ML output preserved unchanged (+ sanitized display filename)
        out = m["output"]
        assert out["predicted_class"] == "normal"
        assert out["confidence"] == 0.87
        assert out["coverage"] == 0.9
        assert out["prediction_set"] == ["normal"]
        assert out["probabilities"] == FAKE_BEARING_RESULT["probabilities"]
        assert out["rpm_used"] == 1200.0 and out["rpm_assumed"] is False
        assert out["segment_agreement"] == 0.95 and out["segments"] == 3
        assert out["duration_s"] == 2.0
        assert out["filename"] == "clip.wav"
        # recording provenance: opaque id + content hash (hash-only retention)
        assert UUID(m["recording_id"])
        assert m["recording_sha256"] == hashlib.sha256(wav).hexdigest()
        assert m["reference_id"] == "sha256:" + FAKE_BEARING_IDENTITY["reference_fingerprint"]
        assert m["model_identity"]["artifact_fingerprint"] == FAKE_BEARING_IDENTITY["artifact_fingerprint"]
        assert m["model_identity"]["preprocessing_fingerprint"] == FAKE_BEARING_IDENTITY[
            "preprocessing_fingerprint"
        ]
        assert m["limitations"] == ["fixture demo bearing classifier"]
        # evidence: class confidence + durations (no score/p-value exists for this model)
        assert evidence_kinds(doc) == [
            "class_confidence",
            "audio_duration_original_s",
            "audio_duration_processed_s",
        ]
        by_kind = {e["kind"]: e for e in doc["evidence"]}
        assert by_kind["class_confidence"]["value"] == 0.87
        assert by_kind["class_confidence"]["unit"] == "probability"
        assert by_kind["audio_duration_original_s"]["value"] == pytest.approx(2.0, abs=0.01)
        assert by_kind["audio_duration_processed_s"]["value"] == 2.0
        # flags: limitations present; no timestamp fabrication; NOT illustrative
        codes = flag_codes(doc)
        assert "MODEL_LIMITATIONS_PRESENT" in codes
        assert "TIMESTAMP_MISSING" in codes  # observed_at omitted, never fabricated
        assert "SOURCE_ILLUSTRATIVE" not in codes
        assert "AUDIO_TRUNCATED" not in codes
        assert "BEARING_UNCERTAIN" not in codes  # a plain normal set is not uncertain
        # flow: inference ran exactly once with the machine's LIVE TELEMETRY speed
        reading = state.fleet.machines["cnc-a1"].reading or {}
        speed = reading.get("rotational_speed_rpm")
        assert speed is not None, "the simulator must tick at startup"
        assert len(stub.calls) == 1
        assert stub.calls[0][1] == float(speed)  # rpm is server-sourced, never client-supplied


def test_create_with_observed_at(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = post_audio(
            client,
            fields={"machine_id": "cnc-a1", "observed_at": "2026-10-15T12:30:00+02:00"},
        )
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        m = modal(doc)
        assert m["observed_at"].startswith("2026-10-15T10:30:00")  # +02:00 -> UTC
        assert all(e["observed_at"] == m["observed_at"] for e in doc["evidence"])
        assert "TIMESTAMP_MISSING" not in flag_codes(doc)


@pytest.mark.parametrize("bad", ["2026-10-15T12:30:00", "not-a-timestamp", "2026-13-45T99:00:00Z"])
def test_observed_at_invalid_rejected(client_factory, bad):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = post_audio(client, fields={"machine_id": "cnc-a1", "observed_at": bad})
        body = assert_error(resp, 422, "request_validation")
        assert "observed_at" in json.dumps(body["error"])


# ---------------------------------------------------------------- auth + owner isolation


def test_auth_required_and_owner_isolation(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        # missing / invalid auth: rejected before the body is parsed
        assert_error(post_audio(client, auth=None), 401, "unauthorized")
        assert_error(post_audio(client, auth={"Authorization": "Bearer wrong-token"}), 401, "unauthorized")
        # a completely malformed body is never parsed without auth either
        garbage = client.post(
            URL_B,
            content=b"not-a-multipart-body",
            headers={"Content-Type": "multipart/form-data; boundary=x"},
        )
        assert_error(garbage, 401, "unauthorized")
        # even a 25 MB body is refused at auth, before any upload processing
        big = b"RIFF-not-really" * (UPLOAD_LIMIT_MB * 1024 * 1024 // 15 + 1)
        huge = post_audio(client, files=[("file", ("clip.wav", big, "audio/wav"))], auth=None)
        assert_error(huge, 401, "unauthorized")
        doc, _ = create_ok(client)  # alice
        got = client.get(f"{URL_BASE}/{doc['assessment_id']}", headers=AUTH_ALICE)
        assert got.status_code == 200 and got.json() == doc
        # bob: cross-owner GET is 404 (never data), list is always owner-scoped
        assert_error(
            client.get(f"{URL_BASE}/{doc['assessment_id']}", headers=AUTH_BOB), 404, "not_found"
        )
        assert client.get(URL_BASE, headers=AUTH_BOB).json()["items"] == []
        listing = client.get(URL_BASE, headers=AUTH_ALICE).json()
        assert [i["assessment_id"] for i in listing["items"]] == [doc["assessment_id"]]
        assert listing["limit"] == 50


# ---------------------------------------------------------------- machine scope (bearing only)


def test_unknown_machine_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = post_audio(client, fields={"machine_id": "ghost-99"})
        body = assert_error(resp, 422, "unknown_machine")
        assert body["error"].get("details") == {"machine_id": "ghost-99"}
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_machine_type_is_fleet_consistency_not_model_selection(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        # omitted -> allowed: the bearing classifier has no machine-type dimension
        assert post_audio(client).status_code == 201
        # matching the machine's recorded type -> allowed
        resp = post_audio(client, fields={"machine_id": "cnc-a1", "machine_type": "fan"})
        assert resp.status_code == 201, resp.text
        # contradicting the fleet record -> rejected as inconsistent metadata (never a
        # fallback, and machine_type never selects a different model)
        resp = post_audio(client, fields={"machine_id": "cnc-a1", "machine_type": "pump"})
        body = assert_error(resp, 422, "machine_type_mismatch")
        assert body["error"]["details"] == {"machine_type": "pump", "acoustic_profile": "fan"}
        resp = post_audio(client, fields={"machine_id": "lth-b1", "machine_type": "fan"})
        body = assert_error(resp, 422, "machine_type_mismatch")
        assert body["error"]["details"] == {"machine_type": "fan", "acoustic_profile": "pump"}
        # an empty value is rejected outright
        resp = post_audio(client, fields={"machine_id": "cnc-a1", "machine_type": ""})
        assert_error(resp, 422, "request_validation")
        assert len(stub.calls) == 2  # exactly the two accepted uploads ran


def test_bearing_only_no_fan_fallback_no_dynamic_types(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        stub = install_fake(state)
        # remove every acoustic model: the bearing slice must not care
        state.registry.acoustic = {}
        state.registry.acoustic_identity = {}
        doc, _ = create_ok(client)
        assert doc["expected_modalities"] == ["bearing"]
        assert [m["kind"] for m in doc["modalities"]] == ["bearing"]
        # the sensor model IS loaded in the fixture and is still never fused in
        assert state.registry.sensor is not None
        kinds = evidence_kinds(doc)
        assert "failure_probability" not in kinds and "anomaly_score" not in kinds
        assert len(stub.calls) == 1


# ---------------------------------------------------------------- strict multipart boundary


def test_unknown_and_forged_fields_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        forged_fields = (
            "score",
            "status",
            "owner",
            "expected_modalities",
            "recording_id",
            "model_identity",
            "source_mode",
            "reference_id",
            "assessment_id",
            "rpm",
        )
        for forged in forged_fields:
            resp = post_audio(client, fields={"machine_id": "cnc-a1", forged: "0.99"})
            body = assert_error(resp, 422, "request_validation")
            assert forged in json.dumps(body["error"]), forged
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_duplicate_fields_and_files_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        wav = wav_bytes()
        # control: the hand-crafted body shape itself is accepted
        ok = post_raw(client, [("machine_id", "cnc-a1")], [("file", "clip.wav", wav, "audio/wav")])
        assert ok.status_code == 201, ok.text
        # duplicate text parts are rejected, never collapsed/ignored
        dup = post_raw(
            client,
            [("machine_id", "cnc-a1"), ("machine_id", "cnc-a1")],
            [("file", "clip.wav", wav, "audio/wav")],
        )
        body = assert_error(dup, 422, "request_validation")
        assert "duplicate" in json.dumps(body["error"])
        # a second file part is rejected too
        two_files = post_raw(
            client,
            [("machine_id", "cnc-a1")],
            [("file", "a.wav", wav, "audio/wav"), ("file", "b.wav", wav, "audio/wav")],
        )
        assert_error(two_files, 422, "request_validation")
        assert len(client.get(URL_BASE, headers=AUTH_ALICE).json()["items"]) == 1  # only the control


def test_too_many_fields_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        fields = {
            "machine_id": "cnc-a1",
            "machine_type": "fan",
            "observed_at": "2026-10-15T12:00:00Z",
            "junk_one": "x",
            "junk_two": "y",  # 5 text parts > max_fields=4
        }
        assert_error(post_audio(client, fields=fields), 422, "request_validation")


def test_missing_file_field_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = client.post(URL_B, data={"machine_id": "cnc-a1"}, headers=AUTH_ALICE)
        assert_error(resp, 422, "request_validation")
        # a TEXT part named "file" is not an upload
        resp = client.post(URL_B, data={"machine_id": "cnc-a1", "file": "not-a-file"}, headers=AUTH_ALICE)
        assert_error(resp, 422, "request_validation")


# ---------------------------------------------------------------- upload safety


def test_oversized_upload_with_declared_length(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        big = b"RIFF-not-really" * (UPLOAD_LIMIT_MB * 1024 * 1024 // 15 + 1)  # > 25 MB
        assert len(big) > UPLOAD_LIMIT_MB * 1024 * 1024
        resp = post_audio(client, files=[("file", ("clip.wav", big, "audio/wav"))])
        assert_error(resp, 413, "file_too_large")
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_oversized_streamed_without_truthful_length(client_factory):
    """Content-Length is never trusted: a chunked body with no (or a false small)
    length must still be stopped by the bounded per-file read."""
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        boundary = "----telltale-boundary-42"
        head = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="machine_id"\r\n\r\ncnc-a1\r\n'
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        big = b"\x00" * (UPLOAD_LIMIT_MB * 1024 * 1024 + 512 * 1024)  # > 25 MB
        headers = {**AUTH_ALICE, "Content-Type": f"multipart/form-data; boundary={boundary}"}
        # (a) no Content-Length at all (streamed body)
        resp = client.post(URL_B, content=iter([head, big, tail]), headers=headers)
        assert_error(resp, 413, "file_too_large")
        # (b) a LYING small Content-Length: still rejected by the bounded read
        lying = {**headers, "Content-Length": "64"}
        resp = client.post(URL_B, content=iter([head, big, tail]), headers=lying)
        assert_error(resp, 413, "file_too_large")
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_empty_file_rejected(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", b"", "audio/wav"))])
        assert_error(resp, 422, "invalid_audio")
        assert stub.calls == []  # never reached the model


def test_corrupt_audio_rejected(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", b"garbage-not-audio" * 500, "audio/wav"))])
        body = assert_error(resp, 422, "invalid_audio")
        assert "libsndfile" not in json.dumps(body)  # decoder internals never leak
        assert stub.calls == []


def test_decode_is_the_only_format_rule(client_factory):
    """No extension allowlist exists on the original bearing path: content decides.
    Undecodable content is rejected whatever its name; decodable content is
    accepted whatever its name (this gap is reported, not silently invented)."""
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        # text content under a .wav name -> cannot decode -> rejected
        resp = post_audio(client, files=[("file", ("clip.wav", b"just some plain text " * 100, "audio/wav"))])
        assert_error(resp, 422, "invalid_audio")
        # a valid WAV under a non-audio extension -> accepted (decode succeeded)
        resp = post_audio(client, files=[("file", ("notes.txt", wav_bytes(), "text/plain"))])
        assert resp.status_code == 201, resp.text
        assert modal(resp.json())["output"]["filename"] == "notes.txt"
        assert len(stub.calls) == 1  # only the decodable upload reached the model


def test_matlab_native_format_accepted(client_factory):
    """.mat is the bearing dataset's NATIVE format: accepted by content, and the
    missing audio header yields no original-duration evidence (never fabricated)."""
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        mat = mat_bytes()
        resp = post_audio(client, files=[("file", ("rec.mat", mat, "application/octet-stream"))])
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        kinds = evidence_kinds(doc)
        assert "audio_duration_original_s" not in kinds  # no libsndfile header for .mat
        assert "audio_duration_processed_s" in kinds
        assert "AUDIO_TRUNCATED" not in flag_codes(doc)  # truncation only when provable
        assert modal(doc)["recording_sha256"] == hashlib.sha256(mat).hexdigest()
        # the container is detected by its MAGIC BYTES, not by the file name
        resp = post_audio(client, files=[("file", ("sneaky.wav", mat, "audio/wav"))])
        assert resp.status_code == 201, resp.text
        assert "audio_duration_original_s" not in evidence_kinds(resp.json())


def test_too_short_audio_rejected(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", wav_bytes(seconds=0.3), "audio/wav"))])
        body = assert_error(resp, 422, "invalid_audio")
        assert "1.0 s" in body["error"]["message"]  # the DEPLOYED segment_seconds, not a constant
        assert stub.calls == []  # rejected before scoring


def test_non_finite_samples_rejected(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", wav_bytes(with_nan=True), "audio/wav"))])
        assert_error(resp, 422, "invalid_audio")
        assert stub.calls == []  # rejected before scoring: no fabricated result


# ---------------------------------------------------------------- duration-cap disclosure


def test_duration_cap_truncation_disclosed(client_factory):
    with client_factory() as client:
        stub = FakeBearing(result={**FAKE_BEARING_RESULT, "duration_s": 60.0})
        install_fake(client.app.state.telltale, stub)
        long_wav = wav_bytes(seconds=65.0)
        doc, _ = create_ok(client, wav=long_wav)
        out = modal(doc)["output"]
        assert out["duration_s"] == 60.0  # original typed ML output preserved unchanged
        by_kind = {e["kind"]: e for e in doc["evidence"]}
        assert 64.5 <= by_kind["audio_duration_original_s"]["value"] <= 65.5
        assert by_kind["audio_duration_processed_s"]["value"] == 60.0
        assert "AUDIO_TRUNCATED" in flag_codes(doc)
        flag = [f for f in doc["quality_flags"] if f["code"] == "AUDIO_TRUNCATED"][0]
        assert flag["severity"] == "info" and "60" in flag["explanation"]


def test_no_truncation_flag_when_within_cap(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        doc, _ = create_ok(client)  # 2 s <= 60 s cap
        assert "AUDIO_TRUNCATED" not in flag_codes(doc)


# ---------------------------------------------------------------- identity + reference provenance


def test_model_and_reference_identity_fields(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        install_fake(state)
        doc, _ = create_ok(client)
        m = modal(doc)
        ident = m["model_identity"]
        assert ident["name"] == "bearing-lightgbm"
        assert ident["training_dataset"] == "mendeley_n9y9c7xrz3"
        # the immutable calibration reference is SEPARATE from the coverage label
        assert m["reference_id"] == "sha256:" + FAKE_BEARING_IDENTITY["reference_fingerprint"]
        assert m["output"]["coverage"] == 0.9  # the label stays in the typed output
        # identity is cache-aligned with the loaded classifier (no per-request hashing)
        assert state.registry.bearing_identity["reference_fingerprint"] == FAKE_BEARING_IDENTITY[
            "reference_fingerprint"
        ]


def test_missing_identity_stays_incomplete_not_invented(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        install_fake(state)
        state.registry.bearing_identity = None  # fingerprinting failed at load
        doc, _ = create_ok(client)
        m = modal(doc)
        assert m["model_identity"] is None
        assert m["reference_id"] is None
        assert "MODEL_IDENTITY_INCOMPLETE" in flag_codes(doc)


def test_registry_identity_deterministic_and_error_consistent(models_dir, tmp_path):
    r1 = ModelRegistry(models_dir).load()
    r2 = ModelRegistry(models_dir).load()
    assert r1.bearing_identity is not None and r2.bearing_identity is not None
    assert r1.bearing_identity == r2.bearing_identity  # deterministic across loads
    ident = r1.bearing_identity
    model_dir = models_dir / "bearing"
    assert ident["artifact_fingerprint"] == fingerprint_dir(model_dir)
    files = [p for p in model_dir.iterdir() if p.is_file()]
    assert ident["artifact_fingerprint"] == manifest_hash(files)
    # reference identity = the metadata.json CALIBRATION record (conformal q/coverage live there)
    assert ident["reference_fingerprint"] == manifest_hash([model_dir / "metadata.json"])
    assert ident["preprocessing_fingerprint"] is not None  # code identity available here
    assert ident["training_dataset"] == "mendeley_n9y9c7xrz3"
    # a corrupt bearing dir yields NO classifier and NO identity, and records the error
    broken = tmp_path / "models" / "bearing"
    broken.mkdir(parents=True)
    (broken / "metadata.json").write_text((model_dir / "metadata.json").read_text("utf-8"), "utf-8")
    r3 = ModelRegistry(tmp_path / "models").load()
    assert r3.bearing is None and r3.bearing_identity is None
    assert "bearing" in r3.errors


def test_reload_keeps_identity_and_snapshot_consistent(client_factory, models_dir):
    with client_factory() as client:
        state = client.app.state.telltale
        before = dict(state.registry.bearing_identity)
        assert before, "fixture bearing model must be loaded with an identity"
        assert before["artifact_fingerprint"] == fingerprint_dir(models_dir / "bearing")
        resp = client.post("/api/v1/models/reload")  # legacy route, no auth
        assert resp.status_code == 200, resp.text
        assert state.registry.bearing_identity == before  # deterministic across reloads
        doc, _ = create_ok(client)
        m = modal(doc)
        assert m["model_identity"]["artifact_fingerprint"] == before["artifact_fingerprint"]
        assert m["model_identity"]["preprocessing_fingerprint"] == before["preprocessing_fingerprint"]
        assert m["reference_id"] == "sha256:" + before["reference_fingerprint"]
        assert m["model_identity"]["trained_at"] is not None  # created_at metadata, parsed


# ---------------------------------------------------------------- model availability + failures


def test_missing_model_dependency_errors(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        state.registry.bearing = None
        state.registry.bearing_identity = None
        resp = post_audio(client)
        body = assert_error(resp, 503, "dependency_unavailable")
        assert "telltale train-bearing" in body["error"]["message"]
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_inference_failure_sanitized_and_no_row(client_factory):
    with client_factory() as client:
        stub = FakeBearing(error=RuntimeError("boom at C:\\secret\\models\\model.txt"))
        install_fake(client.app.state.telltale, stub)
        resp = post_audio(client)
        body = assert_error(resp, 500, "internal_error")
        assert "secret" not in resp.text and "model.txt" not in resp.text  # no paths leaked
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_audio_caused_valueerror_is_422_no_row(client_factory):
    with client_factory() as client:
        stub = FakeBearing(error=ValueError("Audio too short: need >= 1.0s"))
        install_fake(client.app.state.telltale, stub)
        resp = post_audio(client)
        assert_error(resp, 422, "invalid_audio")
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_storage_failure_no_row(client_factory, monkeypatch):
    with client_factory() as client:
        state = client.app.state.telltale
        install_fake(state)

        def boom(*a, **kw):
            raise sqlite3.OperationalError("disk I/O error at D:\\state\\telltale.db")

        monkeypatch.setattr(state.assessments, "insert", boom, raising=True)
        resp = post_audio(client)
        body = assert_error(resp, 500, "internal_error")
        assert "D:\\" not in resp.text and "sqlite" not in resp.text.lower()
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


# ---------------------------------------------------------------- atomicity + cleanup + retention


def test_rejected_requests_never_create_rows(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        post_audio(client, auth=None)  # 401
        post_audio(client, fields={"machine_id": "ghost"})  # 422 unknown_machine
        post_audio(client, files=[("file", ("c.wav", b"junk", "audio/wav"))])  # 422 invalid_audio
        post_audio(client, fields={"machine_id": "cnc-a1", "score": "0.9"})  # 422 forged field
        post_audio(client, fields={"machine_id": "cnc-a1", "machine_type": "pump"})  # 422 mismatch
        doc, _ = create_ok(client)  # the only success
        listing = client.get(URL_BASE, headers=AUTH_ALICE).json()
        assert [i["assessment_id"] for i in listing["items"]] == [doc["assessment_id"]]


def test_upload_resources_always_closed():
    async def run() -> tuple[bool, bool]:
        up_ok = __import__("starlette").datastructures.UploadFile(file=io.BytesIO(b"data"), filename="a.wav")
        await _bounded_upload_bytes(up_ok, limit=1024)
        closed_ok = up_ok.file.closed  # starlette 1.7: closure state lives on the wrapped file
        up_big = __import__("starlette").datastructures.UploadFile(
            file=io.BytesIO(b"x" * 64), filename="a.wav"
        )
        with pytest.raises(Exception) as exc:
            await _bounded_upload_bytes(up_big, limit=16)
        return closed_ok, up_big.file.closed and "file_too_large" in str(exc.value.code)

    closed_on_success, closed_on_error = asyncio.run(run())
    assert closed_on_success and closed_on_error


def test_old_snapshot_unchanged_by_new_creates(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        first, _ = create_ok(client)
        second, _ = create_ok(client, wav=wav_bytes(seconds=3.0))
        assert second["assessment_id"] != first["assessment_id"]
        again = client.get(f"{URL_BASE}/{first['assessment_id']}", headers=AUTH_ALICE)
        assert again.status_code == 200 and again.json() == first


def test_restart_persistence_same_document(client_factory, tmp_path):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        doc, _ = create_ok(client)
    with client_factory(data_dir=tmp_path) as client2:  # same DB file, new process-like state
        install_fake(client2.app.state.telltale)
        got = client2.get(f"{URL_BASE}/{doc['assessment_id']}", headers=AUTH_ALICE)
        assert got.status_code == 200 and got.json() == doc
        # owner scoping survives the restart
        assert client2.get(f"{URL_BASE}/{doc['assessment_id']}", headers=AUTH_BOB).status_code == 404


def test_hash_only_retention_no_audio_on_disk(client_factory, tmp_path):
    """Hash-only retention: raw bytes are fingerprinted, then discarded - the DB
    carries the sha256 but never the payload, and no audio file is written."""
    with client_factory() as client:  # data_dir == tmp_path: inspect exactly what was written
        install_fake(client.app.state.telltale)
        doc, wav = create_ok(client)
        assert modal(doc)["recording_sha256"] == hashlib.sha256(wav).hexdigest()
        audio_exts = {".wav", ".flac", ".ogg", ".mp3", ".mat", ".aiff", ".au"}
        stray = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts]
        assert stray == []  # no raw audio anywhere under the data dir
        dbs = [p for p in tmp_path.rglob("*.db")]
        assert dbs, "the assessment store database must live under data_dir"
        blob = b"".join(p.read_bytes() for p in dbs)
        assert hashlib.sha256(wav).hexdigest().encode() in blob  # the HASH is recorded
        assert b"RIFF" not in blob and wav not in blob  # the payload never is


# ---------------------------------------------------------------- conservative assess/v1 mapping


@pytest.mark.parametrize(
    "result,status,reason,flag_on",
    [
        (FAULT_RESULT, "watch", "bearing:fault_classes_in_set", True),
        (UNCERTAIN_RESULT, "unknown", "bearing:classifier_uncertain", True),
        (AMBIGUOUS_RESULT, "unknown", "bearing:ambiguous_normal_with_faults", True),
    ],
)
def test_status_mapping_conservative(client_factory, result, status, reason, flag_on):
    with client_factory() as client:
        install_fake(client.app.state.telltale, stub=FakeBearing(result=result))
        doc, _ = create_ok(client)
        # the bearing mapping never reports "critical" and never silently "normal"
        assert doc["overall_status"] == status
        assert doc["overall_status"] != "critical"
        assert reason in doc["status_reasons"]
        # any non-normal bearing indication is flagged as a conservative review state
        assert ("BEARING_UNCERTAIN" in flag_codes(doc)) is flag_on


def test_no_fabricated_score_pvalue_or_fusion(client_factory):
    """This classifier has NO anomaly score, NO conformal p-value and NO fused
    sensor probability; none may ever appear in the evidence or the output."""
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        doc, _ = create_ok(client)
        kinds = evidence_kinds(doc)
        assert kinds == ["class_confidence", "audio_duration_original_s", "audio_duration_processed_s"]
        assert "anomaly_score" not in kinds and "conformal_p_value" not in kinds
        assert "failure_probability" not in kinds  # no sensor+acoustic fusion either
        out = modal(doc)["output"]
        assert "score" not in out and "p_value" not in out
        assert doc["expected_modalities"] == ["bearing"]  # bearing-only profile


def test_filename_is_display_only_never_label(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)  # predicts "normal"
        files = [("file", ("inner_race_defect_CONFIRMED.wav", wav_bytes(), "audio/wav"))]
        resp = post_audio(client, files=files)
        assert resp.status_code == 201, resp.text
        out = modal(resp.json())["output"]
        assert out["filename"] == "inner_race_defect_CONFIRMED.wav"  # display metadata only
        assert out["predicted_class"] == "normal"  # the class comes from the audio, never the name
        # path components are stripped: the name is never a storage path
        resp = post_audio(client, files=[("file", ("../../evil/x.wav", wav_bytes(), "audio/wav"))])
        assert resp.status_code == 201, resp.text
        assert modal(resp.json())["output"]["filename"] == "x.wav"
        assert len(stub.calls) == 2


# ---------------------------------------------------------------- legacy behaviour preserved


def test_legacy_bearing_diagnose_unchanged(client_factory):
    """The legacy no-auth /acoustic/diagnose route keeps its exact behaviour
    (it stays outside this slice's auth boundary and contract)."""
    with client_factory() as client:
        state = client.app.state.telltale
        assert state.registry.bearing is not None  # real fixture model
        resp = client.post(
            "/api/v1/acoustic/diagnose",
            files={"file": ("b.wav", wav_bytes(seconds=3.0), "audio/wav")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        expected_keys = {
            "filename",
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
            "recommendations",
            "model",
        }
        assert expected_keys == set(body)  # exact legacy response shape (response_model)
        assert body["filename"] == "b.wav"
        assert body["predicted_class"] in {"normal", "inner_race", "ball"}
        assert body["recommendations"]  # playbook still attached on the legacy route
        assert body["rpm_assumed"] is True and body["rpm_used"] == 1200.0  # no speed given
        assert body["model"]["dataset"] == "mendeley_n9y9c7xrz3"
        # a client-supplied rpm stays accepted here (deliberately outside the new contract)
        resp = client.post(
            "/api/v1/acoustic/diagnose",
            files={"file": ("b.wav", wav_bytes(seconds=3.0), "audio/wav")},
            data={"rpm": "1500"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["rpm_used"] == 1500.0 and resp.json()["rpm_assumed"] is False
        # machine_id still reads the live telemetry speed
        reading = state.fleet.machines["cnc-a1"].reading or {}
        speed = reading.get("rotational_speed_rpm")
        if speed is not None:
            resp = client.post(
                "/api/v1/acoustic/diagnose",
                files={"file": ("b.wav", wav_bytes(seconds=3.0), "audio/wav")},
                data={"machine_id": "cnc-a1"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["rpm_used"] == float(speed)


# ---------------------------------------------------------------- REAL fixture classifier (not a mock)


def test_real_fixture_detector_api_vs_direct(client_factory, models_dir):
    """The stored output must equal a DIRECT call of the same deployed fixture
    classifier on the same bytes (bit-identical values, no re-derivation)."""
    with client_factory() as client:
        state = client.app.state.telltale
        classifier = state.registry.bearing
        assert classifier is not None, "conftest fixture bearing model must be loaded"
        wav = wav_bytes(seconds=2.0)
        doc, _ = create_ok(client, wav=wav)
        reading = state.fleet.machines["cnc-a1"].reading or {}
        speed = reading.get("rotational_speed_rpm")
        rpm = float(speed) if speed is not None else None
        direct = classifier.diagnose(io.BytesIO(wav), rpm)
        out = modal(doc)["output"]
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
            assert out[key] == direct[key], key
        assert out["filename"] == "clip.wav"  # only addition: sanitized display metadata
        # identity matches the real artifacts on disk
        model_dir = models_dir / "bearing"
        ident = state.registry.bearing_identity
        assert ident is not None
        assert modal(doc)["model_identity"]["artifact_fingerprint"] == ident["artifact_fingerprint"]
        assert ident["artifact_fingerprint"] == fingerprint_dir(model_dir)
        files = [p for p in model_dir.iterdir() if p.is_file()]
        assert ident["artifact_fingerprint"] == manifest_hash(files)
        # reference/calibration identity = the metadata.json record (conformal q lives there)
        assert ident["reference_fingerprint"] == manifest_hash([model_dir / "metadata.json"])
        assert modal(doc)["reference_id"] == "sha256:" + ident["reference_fingerprint"]
        assert ident["preprocessing_fingerprint"] is not None  # code identity available here
        assert ident["training_dataset"] == "mendeley_n9y9c7xrz3"
        assert modal(doc)["recording_sha256"] == hashlib.sha256(wav).hexdigest()


# ---------------------------------------------------------------- cross-slice regression


def test_sensor_and_fan_acoustic_slices_still_work(client_factory):
    """Cross-slice regression: slice 1 (sensor JSON) and slice 2 (fan acoustic)
    are untouched by the bearing slice."""
    with client_factory() as client:
        state = client.app.state.telltale
        body = {
            "reading": {
                "product_quality": "M",
                "air_temperature_k": 302.4,
                "process_temperature_k": 310.6,
                "rotational_speed_rpm": 1352.0,
                "torque_nm": 54.2,
                "tool_wear_min": 212.0,
            },
            "machine_id": "cnc-a1",
            "source_mode": "simulated",
        }
        resp = client.post(URL_BASE, json=body, headers=AUTH_ALICE)
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["sensor"]
        # fan acoustic creation (real fixture model, unchanged slice)
        resp = client.post(
            "/api/v1/assessments/acoustic",
            data={"machine_id": "cnc-a1"},
            files=[("file", ("clip.wav", wav_bytes(), "audio/wav"))],
            headers=AUTH_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["expected_modalities"] == ["acoustic"]
        # the bearing route is still its own slice: one bearing modality only
        install_fake(state)
        doc, _ = create_ok(client)
        assert doc["expected_modalities"] == ["bearing"]
        assert [m["kind"] for m in doc["modalities"]] == ["bearing"]
