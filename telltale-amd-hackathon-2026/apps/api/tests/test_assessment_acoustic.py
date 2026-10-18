"""Endpoint slice-2 tests: trusted FAN acoustic assessment creation (multipart upload).

Covers the approved contract: strict multipart boundary (unknown/duplicate/forged
parts rejected, auth before body parsing), fan-only machine scope, upload safety
(bounded reads, allowlist backed by successful decoding, MATLAB rejected, corrupt/
empty/too-short/non-finite rejected), 30 s processing-cap disclosure, hash-only
recording provenance, model + reference identity, enrolled-reference refusal,
deterministic acoustic-only profile semantics, failure atomicity (no row on any
failure) and resource cleanup. Legacy /acoustic/analyze behaviour is asserted
unchanged.

Mock-based tests inject a deterministic FakeAcoustic detector; the API/direct
comparison test runs the REAL fixture-trained tiny fan ensemble from conftest.
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
import soundfile as sf
from fastapi.testclient import TestClient

from telltale_api.api.routes.assessments import _bounded_upload_bytes
from telltale_api.core.config import Settings
from telltale_api.main import create_app
from telltale_api.services.registry import ModelRegistry, fingerprint_dir

URL_A = "/api/v1/assessments/acoustic"
URL_BASE = "/api/v1/assessments"

TOKENS = json.dumps({"alice": "tok-alice-9f3d2c7e", "bob": "tok-bob-77aa11bb"})
AUTH_ALICE = {"Authorization": "Bearer tok-alice-9f3d2c7e"}
AUTH_BOB = {"Authorization": "Bearer tok-bob-77aa11bb"}

UPLOAD_LIMIT_MB = 25  # Settings.max_upload_mb default

FAKE_ACOUSTIC_RESULT = {
    "machine_type": "fan",
    "method": "ensemble",
    "score": 1.5,
    "threshold": 1.34,
    "severity": 1.1194,
    "p_value": 0.03,
    "unusual_percentile": 97.0,
    "calibration": "global",
    "status": "anomalous",
    "is_anomalous": True,
    "duration_s": 2.0,
    "sample_rate": 16000,
    "peak_time_s": 1.0,
    "frame_errors": {"times": [0.5, 1.0], "values": [1.4, 1.5]},
    "spectrogram": {"times": [0.5], "band_hz": [125.0], "values": [[-20.0]]},
    "band_deviation": {"band_hz": [125.0], "z": [0.1]},
    "waveform": {"duration_s": 2.0, "min": [-0.1], "max": [0.1]},
}

FAKE_IDENTITY = {
    "name": "acoustic-fan-ensemble",
    "version": "9.9.9-fake",
    "trained_at": None,
    "artifact_fingerprint": "a" * 64,
    "preprocessing_fingerprint": "b" * 64,
    "training_dataset": "mimii_due",
    "reference_fingerprint": "c" * 64,
}


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


def wav_bytes(seconds: float = 2.0, sr: int = 16000, amplitude: float = 0.05, with_nan: bool = False) -> bytes:
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


def install_fake(state, stub: FakeAcoustic | None = None, identity: dict | None = None) -> FakeAcoustic:
    stub = stub or FakeAcoustic()
    state.registry.acoustic = {"fan": stub}
    state.registry.acoustic_identity = {"fan": dict(identity or FAKE_IDENTITY)}
    return stub


def post_audio(client, *, fields=None, files=None, auth=AUTH_ALICE, extra_headers=None):
    data = {"machine_id": "cnc-a1"} if fields is None else fields
    if files is None:
        files = [("file", ("clip.wav", wav_bytes(), "audio/wav"))]
    headers = dict(auth) if auth else {}
    headers.update(extra_headers or {})
    return client.post(URL_A, data=data, files=files, headers=headers)


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
    return client.post(URL_A, content=body, headers={**auth, "Content-Type": ctype})


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


# ---------------------------------------------------------------- happy path (mock)


def test_create_ok_stub(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        stub = install_fake(state)
        doc, wav = create_ok(client)
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", doc["assessment_id"]
        )
        m = modal(doc)
        # acoustic-only server profile + uploaded provenance
        assert doc["expected_modalities"] == ["acoustic"]
        assert doc["source_mode"] == "uploaded"
        assert doc["evidence_completeness"] == {"expected": 1, "present": 1, "ratio": 1.0}
        assert m["source_mode"] == "uploaded"
        assert m["illustrative"] is False  # uploaded is NOT automatically illustrative
        assert doc["overall_status"] == "critical"  # anomalous -> critical (assess/v1)
        # typed ML output preserved unchanged (plus sanitized display filename)
        out = m["output"]
        assert out["score"] == FAKE_ACOUSTIC_RESULT["score"]
        assert out["p_value"] == FAKE_ACOUSTIC_RESULT["p_value"]
        assert out["status"] == "anomalous"
        assert out["calibration"] == "global"
        assert out["duration_s"] == 2.0
        assert out["filename"] == "clip.wav"
        # recording provenance: opaque id + content hash (hash-only retention)
        assert UUID(m["recording_id"])
        assert m["recording_sha256"] == hashlib.sha256(wav).hexdigest()
        assert m["reference_id"] == "sha256:" + FAKE_IDENTITY["reference_fingerprint"]
        assert m["model_identity"]["artifact_fingerprint"] == FAKE_IDENTITY["artifact_fingerprint"]
        assert m["model_identity"]["preprocessing_fingerprint"] == FAKE_IDENTITY["preprocessing_fingerprint"]
        assert m["limitations"] == ["fixture demo sound model"]
        # evidence: score + p-value + durations, distinct kinds
        kinds = [e["kind"] for e in doc["evidence"]]
        assert kinds == ["anomaly_score", "conformal_p_value", "audio_duration_original_s", "audio_duration_processed_s"]
        by_kind = {e["kind"]: e for e in doc["evidence"]}
        assert by_kind["anomaly_score"]["value"] == 1.5 and by_kind["anomaly_score"]["unit"] == "score"
        assert by_kind["conformal_p_value"]["value"] == 0.03
        assert by_kind["audio_duration_original_s"]["value"] == pytest.approx(2.0, abs=0.01)
        assert by_kind["audio_duration_processed_s"]["value"] == 2.0
        # flags: limitations present; no timestamp fabrication; NOT illustrative; NOT truncated
        codes = flag_codes(doc)
        assert "MODEL_LIMITATIONS_PRESENT" in codes
        assert "TIMESTAMP_MISSING" in codes  # observed_at omitted, never fabricated
        assert "SOURCE_ILLUSTRATIVE" not in codes
        assert "AUDIO_TRUNCATED" not in codes
        # flow: inference ran exactly once, in the threadpool, on the real bytes
        assert len(stub.calls) == 1


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
        doc, _ = create_ok(client)  # alice
        got = client.get(f"{URL_BASE}/{doc['assessment_id']}", headers=AUTH_ALICE)
        assert got.status_code == 200 and got.json() == doc
        # bob: cross-owner GET is 404 (never data), list is always owner-scoped
        assert_error(
            client.get(f"{URL_BASE}/{doc['assessment_id']}", headers=AUTH_BOB), 404, "not_found"
        )
        listing_bob = client.get(URL_BASE, headers=AUTH_BOB).json()
        assert listing_bob["items"] == []
        listing_alice = client.get(URL_BASE, headers=AUTH_ALICE).json()
        assert [i["assessment_id"] for i in listing_alice["items"]] == [doc["assessment_id"]]
        assert listing_alice["limit"] == 50


# ---------------------------------------------------------------- machine scope (fan only)


def test_unknown_machine_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = post_audio(client, fields={"machine_id": "ghost-99"})
        body = assert_error(resp, 422, "unknown_machine")
        assert body["error"].get("details") == {"machine_id": "ghost-99"}
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_profile_mismatch_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        # cnc-a1 has acoustic_profile fan; claiming pump is a mismatch, never a fallback
        resp = post_audio(client, fields={"machine_id": "cnc-a1", "machine_type": "pump"})
        body = assert_error(resp, 422, "acoustic_profile_mismatch")
        assert body["error"]["details"] == {"machine_type": "pump", "acoustic_profile": "fan"}


def test_unsupported_type_no_fan_fallback(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        # lth-b1 is a pump machine: unsupported here, and never silently scored as fan
        resp = post_audio(client, fields={"machine_id": "lth-b1"})
        assert_error(resp, 422, "unsupported_machine_type")
        assert stub.calls == []  # no inference happened


def test_explicit_type_still_needs_profile_match(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        # explicit fan on a fan machine is fine
        resp = post_audio(client, fields={"machine_id": "cnc-a1", "machine_type": "fan"})
        assert resp.status_code == 201, resp.text


# ---------------------------------------------------------------- strict multipart boundary


def test_unknown_and_forged_fields_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        for forged in ("score", "status", "owner", "expected_modalities", "recording_id", "model_identity"):
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
        assert_error(post_audio(client, fields=None, files=[("file", ("a.txt", b"hi", "text/plain"))]), 422, "invalid_audio")
        assert len(client.get(URL_BASE, headers=AUTH_ALICE).json()["items"]) == 1  # only the control create


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
        resp = client.post(URL_A, data={"machine_id": "cnc-a1"}, headers=AUTH_ALICE)
        assert_error(resp, 422, "request_validation")
        # a TEXT part named "file" is not an upload
        resp = client.post(URL_A, data={"machine_id": "cnc-a1", "file": "not-a-file"}, headers=AUTH_ALICE)
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
        resp = client.post(URL_A, content=iter([head, big, tail]), headers=headers)
        assert_error(resp, 413, "file_too_large")
        # (b) a LYING small Content-Length: still rejected by the bounded read
        lying = {**headers, "Content-Length": "64"}
        resp = client.post(
            URL_A, content=iter([head, big, tail]), headers=lying
        )
        assert_error(resp, 413, "file_too_large")
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_empty_file_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", b"", "audio/wav"))])
        assert_error(resp, 422, "invalid_audio")


def test_corrupt_audio_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", b"garbage-not-audio" * 500, "audio/wav"))])
        body = assert_error(resp, 422, "invalid_audio")
        assert "libsndfile" not in json.dumps(body)  # decoder internals never leak


def test_unsupported_and_matlab_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        for name in ("clip.txt", "clip.mp3", "clip.mat", "clip"):
            resp = post_audio(client, files=[("file", (name, wav_bytes(), "audio/wav"))])
            assert_error(resp, 422, "invalid_audio")
        # MATLAB magic bytes inside a .wav name are still refused (extension alone is not trusted)
        resp = post_audio(client, files=[("file", ("clip.wav", b"MATLAB 5.0 MAT-file" + b"\x00" * 64, "audio/wav"))])
        assert_error(resp, 422, "invalid_audio")


def test_too_short_audio_rejected(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", wav_bytes(seconds=0.3), "audio/wav"))])
        assert_error(resp, 422, "invalid_audio")


def test_non_finite_samples_rejected(client_factory):
    with client_factory() as client:
        stub = install_fake(client.app.state.telltale)
        resp = post_audio(client, files=[("file", ("clip.wav", wav_bytes(with_nan=True), "audio/wav"))])
        assert_error(resp, 422, "invalid_audio")
        assert stub.calls == []  # rejected before scoring: no fabricated result


def test_upload_resources_always_closed():
    async def run() -> tuple[bool, bool]:
        up_ok = __import__("starlette").datastructures.UploadFile(file=io.BytesIO(b"data"), filename="a.wav")
        await _bounded_upload_bytes(up_ok, limit=1024)
        closed_ok = up_ok.file.closed  # starlette 1.7: closure state lives on the wrapped file
        up_big = __import__("starlette").datastructures.UploadFile(file=io.BytesIO(b"x" * 64), filename="a.wav")
        with pytest.raises(Exception) as exc:
            await _bounded_upload_bytes(up_big, limit=16)
        return closed_ok, up_big.file.closed and "file_too_large" in str(exc.value.code)

    closed_on_success, closed_on_error = asyncio.run(run())
    assert closed_on_success and closed_on_error


# ---------------------------------------------------------------- duration-cap disclosure


def test_duration_cap_truncation_disclosed(client_factory):
    with client_factory() as client:
        stub = FakeAcoustic(result={**FAKE_ACOUSTIC_RESULT, "duration_s": 30.0})
        install_fake(client.app.state.telltale, stub)
        long_wav = wav_bytes(seconds=35.0)
        doc, _ = create_ok(client, wav=long_wav)
        m = modal(doc)
        out = m["output"]
        assert out["duration_s"] == 30.0  # original typed ML output preserved unchanged
        by_kind = {e["kind"]: e for e in doc["evidence"]}
        assert 34.5 <= by_kind["audio_duration_original_s"]["value"] <= 35.5
        assert by_kind["audio_duration_processed_s"]["value"] == 30.0
        assert "AUDIO_TRUNCATED" in flag_codes(doc)
        flag = [f for f in doc["quality_flags"] if f["code"] == "AUDIO_TRUNCATED"][0]
        assert flag["severity"] == "info" and "30" in flag["explanation"]


def test_no_truncation_flag_when_within_cap(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        doc, _ = create_ok(client)  # 2 s <= 30 s cap
        assert "AUDIO_TRUNCATED" not in flag_codes(doc)


def test_duration_evidence_without_invention(client_factory):
    """A header without reliable duration info yields no original-duration evidence
    (never fabricated); the processed duration always comes from the model output."""
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        doc, _ = create_ok(client)
        kinds = [e["kind"] for e in doc["evidence"]]
        assert "audio_duration_original_s" in kinds  # reliable header probe succeeded here
        assert "audio_duration_processed_s" in kinds


# ---------------------------------------------------------------- identity + reference provenance


def test_model_and_reference_identity_fields(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        install_fake(state)
        doc, _ = create_ok(client)
        m = modal(doc)
        ident = m["model_identity"]
        assert ident["training_dataset"] == "mimii_due"
        # the immutable reference identity is SEPARATE from the calibration label
        assert m["reference_id"] == "sha256:" + FAKE_IDENTITY["reference_fingerprint"]
        assert m["output"]["calibration"] == "global"  # label stays in the typed output
        # identity is cache-aligned with the loaded detector (no per-request hashing)
        assert state.registry.acoustic_identity["fan"]["reference_fingerprint"] == FAKE_IDENTITY[
            "reference_fingerprint"
        ]


def test_missing_identity_stays_incomplete_not_invented(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        install_fake(state, identity=None)
        state.registry.acoustic_identity = {"fan": None}  # fingerprinting failed at load
        doc, _ = create_ok(client)
        m = modal(doc)
        assert m["model_identity"] is None
        assert m["reference_id"] is None
        assert "MODEL_IDENTITY_INCOMPLETE" in flag_codes(doc)


def test_enrolled_reference_state_refused(client_factory):
    with client_factory() as client:
        stub = FakeAcoustic(enrolled={"line-a": {"views": {}, "calibration": []}})
        install_fake(client.app.state.telltale, stub)
        resp = post_audio(client)
        assert_error(resp, 409, "enrolled_reference_active")
        assert stub.calls == []  # no inference, no snapshot
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


# ---------------------------------------------------------------- model availability + failures


def test_missing_model_dependency_errors(client_factory):
    with client_factory() as client:
        state = client.app.state.telltale
        state.registry.acoustic = {}
        state.registry.acoustic_identity = {}
        assert_error(post_audio(client), 503, "dependency_unavailable")  # no sound model at all
        state.registry.acoustic = {"pump": FakeAcoustic()}
        state.registry.acoustic_identity = {"pump": dict(FAKE_IDENTITY)}
        assert_error(post_audio(client), 404, "dependency_error")  # fan specifically missing
        assert client.get(URL_BASE, headers=AUTH_ALICE).json()["items"] == []


def test_inference_failure_sanitized_and_no_row(client_factory):
    with client_factory() as client:
        stub = FakeAcoustic(error=RuntimeError("boom at C:\\secret\\models\\embedder_0.onnx"))
        install_fake(client.app.state.telltale, stub)
        resp = post_audio(client)
        body = assert_error(resp, 500, "internal_error")
        assert "secret" not in resp.text and "embedder" not in resp.text  # no paths leaked
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


def test_rejected_requests_never_create_rows(client_factory):
    with client_factory() as client:
        install_fake(client.app.state.telltale)
        post_audio(client, auth=None)  # 401
        post_audio(client, fields={"machine_id": "ghost"})  # 422
        post_audio(client, files=[("file", ("c.wav", b"junk", "audio/wav"))])  # 422
        post_audio(client, fields={"machine_id": "cnc-a1", "score": "0.9"})  # 422
        doc, _ = create_ok(client)  # the only success
        listing = client.get(URL_BASE, headers=AUTH_ALICE).json()
        assert [i["assessment_id"] for i in listing["items"]] == [doc["assessment_id"]]


# ---------------------------------------------------------------- immutability + persistence


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


# ---------------------------------------------------------------- REAL fixture detector (not a mock)


def test_real_fixture_detector_api_vs_direct(client_factory, models_dir):
    """The stored output must equal a DIRECT call of the same deployed fixture
    detector on the same bytes (bit-identical values, no re-derivation)."""
    with client_factory() as client:
        state = client.app.state.telltale
        detector = state.registry.acoustic.get("fan")
        assert detector is not None, "conftest fixture fan model must be loaded"
        wav = wav_bytes(seconds=2.0)
        doc, _ = create_ok(client, wav=wav)
        direct = detector.analyze(io.BytesIO(wav))
        out = modal(doc)["output"]
        assert out["score"] == direct["score"]
        assert out["p_value"] == direct["p_value"]
        assert out["status"] == direct["status"]
        assert out["calibration"] == direct["calibration"]
        assert out["duration_s"] == direct["duration_s"]
        assert out["is_anomalous"] == direct["is_anomalous"]
        # identity matches the real artifacts on disk
        model_dir = models_dir / "acoustic" / "fan"
        ident = state.registry.acoustic_identity["fan"]
        assert modal(doc)["model_identity"]["artifact_fingerprint"] == ident["artifact_fingerprint"]
        assert ident["artifact_fingerprint"] == fingerprint_dir(model_dir)
        files = [p for p in model_dir.iterdir() if p.is_file()]
        assert ident["artifact_fingerprint"] == manifest_hash(files)
        bank = model_dir / detector.metadata["bank_file"]
        assert ident["reference_fingerprint"] == manifest_hash([bank, model_dir / "metadata.json"])
        assert modal(doc)["reference_id"] == "sha256:" + ident["reference_fingerprint"]
        assert ident["preprocessing_fingerprint"] is not None  # code identity available here
        assert modal(doc)["recording_sha256"] == hashlib.sha256(wav).hexdigest()


def test_registry_identity_deterministic_and_error_consistent(models_dir, tmp_path):
    r1 = ModelRegistry(models_dir).load()
    r2 = ModelRegistry(models_dir).load()
    assert r1.acoustic_identity["fan"] == r2.acoustic_identity["fan"]  # deterministic across loads
    assert r1.acoustic_identity["fan"]["artifact_fingerprint"] == fingerprint_dir(models_dir / "acoustic" / "fan")
    assert r1.acoustic_identity["fan"]["preprocessing_fingerprint"] is not None
    # a corrupt model dir yields NO detector and NO identity, and records the error
    broken = tmp_path / "models" / "acoustic" / "fan"
    broken.mkdir(parents=True)
    (broken / "metadata.json").write_text((models_dir / "acoustic" / "fan" / "metadata.json").read_text("utf-8"), "utf-8")
    r3 = ModelRegistry(tmp_path / "models").load()
    assert "fan" not in r3.acoustic and "fan" not in r3.acoustic_identity
    assert "acoustic/fan" in r3.errors


# ---------------------------------------------------------------- legacy behaviour preserved


def test_legacy_analyze_unchanged(client_factory):
    """The legacy no-auth analyze route keeps its response shape and behaviour."""
    with client_factory() as client:
        state = client.app.state.telltale
        assert state.registry.acoustic.get("fan") is not None  # real fixture model
        resp = client.post(
            "/api/v1/acoustic/analyze",
            files={"file": ("clip.wav", wav_bytes(), "audio/wav")},
            data={"machine_id": "cnc-a1"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for key in ("machine_type", "score", "status", "duration_s", "machine_id", "filename", "recommendations", "model"):
            assert key in body, key
        assert body["machine_id"] == "cnc-a1" and body["filename"] == "clip.wav"
        assert body["machine_type"] == "fan"


def test_sensor_slice_still_works(client_factory):
    """Cross-slice regression: the JSON sensor route is untouched by slice 2."""
    with client_factory() as client:
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
        doc = resp.json()
        assert doc["expected_modalities"] == ["sensor"]
        assert modal(doc)["recording_sha256"] is None  # additive field defaults harmlessly
