import pytest

from .conftest import tone, wav

READING = {
    "product_quality": "L",
    "air_temperature_k": 302.0,
    "process_temperature_k": 310.5,
    "rotational_speed_rpm": 1400,
    "torque_nm": 55.0,
    "tool_wear_min": 210,
}


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["models"]["sensor"] is True


def test_models_status(client):
    body = client.get("/api/v1/models").json()
    assert body["sensor"]["loaded"]
    assert "failure" in body["sensor"]["test_metrics"]


def test_predict(client):
    r = client.post("/api/v1/sensors/predict", json=READING)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["recommendations"]
    assert body["contributions"][0]["feature"]


def test_predict_validation(client):
    r = client.post("/api/v1/sensors/predict", json={**READING, "product_quality": "Z"})
    assert r.status_code == 422


def test_what_if_default_range(client):
    r = client.post("/api/v1/sensors/what-if", json={"reading": READING, "parameter": "torque_nm", "steps": 10})
    assert r.status_code == 200, r.text
    assert len(r.json()["points"]) == 10


def test_fleet_flow(client):
    overview = client.get("/api/v1/machines/overview").json()
    assert overview["machines"] >= 1
    machines = client.get("/api/v1/machines").json()
    mid = machines[0]["id"]
    detail = client.get(f"/api/v1/machines/{mid}").json()
    assert detail["latest_reading"] is not None
    assert detail["assessment"]["recommendations"]
    r = client.post(f"/api/v1/machines/{mid}/maintenance", json={"action": "replace_tool"})
    assert r.status_code == 200
    s = client.post(f"/api/v1/machines/{mid}/summary").json()
    assert s["source"] == "template" and s["summary"]
    assert client.get("/api/v1/machines/nope").status_code == 404


def test_alerts_lifecycle(client):
    state = client.app.state.telltale
    alert = state.alerts.create(machine_id="cnc-a1", machine_name="CNC Mill A1", source="sensor",
                                severity="critical", title="t", detail="d")
    assert client.post(f"/api/v1/alerts/{alert.id}/acknowledge").json()["status"] == "acknowledged"
    assert client.post(f"/api/v1/alerts/{alert.id}/resolve", json={"note": "fixed"}).json()["note"] == "fixed"
    assert client.post("/api/v1/alerts/missing/resolve").status_code == 404


def test_acoustic_analyze(client):
    if "fan" not in client.get("/api/v1/health").json()["acoustic_machine_types"]:
        pytest.skip("torch not installed; no acoustic model trained")
    files = {"file": ("clip.wav", wav(tone(400, seed=99)), "audio/wav")}
    r = client.post("/api/v1/acoustic/analyze", files=files, data={"machine_id": "cnc-a1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["machine_type"] == "fan"
    assert body["method"] == "ensemble" and 0 < body["p_value"] <= 1
    assert body["status"] in {"normal", "watch", "anomalous"}
    assert client.get("/api/v1/machines/cnc-a1").json()["acoustic"] is not None


def test_acoustic_rejects_garbage(client):
    if "fan" not in client.get("/api/v1/health").json()["acoustic_machine_types"]:
        pytest.skip("no acoustic model")
    r = client.post("/api/v1/acoustic/analyze", files={"file": ("x.wav", b"not audio", "audio/wav")},
                    data={"machine_type": "fan"})
    assert r.status_code == 422


def test_unknown_sound_model_is_a_clear_error(client):
    r = client.post("/api/v1/acoustic/analyze", files={"file": ("x.wav", wav(tone(100)), "audio/wav")},
                    data={"machine_type": "valve"})
    assert r.status_code in (404, 503)
    assert "telltale train-sound" in r.json()["detail"]


def test_bearing_diagnose_accepts_wav_and_mat(client):
    import io

    import numpy as np
    import scipy.io as sio

    r = client.post("/api/v1/acoustic/diagnose", files={"file": ("b.wav", wav(tone(300, seconds=3.0)), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] in {"normal", "inner_race", "ball"}
    assert body["recommendations"]

    t = np.arange(44_100 * 2) / 44_100
    buf = io.BytesIO()
    sio.savemat(buf, {"x": (0.4 * np.sin(2 * np.pi * 900 * t))[:, None]})
    r = client.post("/api/v1/acoustic/diagnose", files={"file": ("rec.mat", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 200, r.text
    assert r.json()["predicted_class"] == "inner_race"


def test_samples_listing_parses_multiword_labels(client, tmp_path):
    samples = client.app.state.telltale.settings.samples_dir
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "bearing_inner_race_0.wav").write_bytes(wav(tone(300)))
    listed = {s["name"]: s for s in client.get("/api/v1/acoustic/samples").json()}
    assert listed["bearing_inner_race_0.wav"]["machine_type"] == "bearing"
    assert listed["bearing_inner_race_0.wav"]["label"] == "inner_race"


def test_enroll_routes(client):
    if "fan" not in client.get("/api/v1/health").json()["acoustic_machine_types"]:
        pytest.skip("no acoustic model")
    files = [("files", (f"n{i}.wav", wav(tone(400, seconds=2.0, seed=300 + i)), "audio/wav")) for i in range(3)]
    r = client.post("/api/v1/acoustic/enroll", files=files, data={"machine_id": "cnc-a1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "cnc-a1" and body["clips"] == 3 and body["machine_type"] == "fan" and not body["calibrated"]
    listed = client.get("/api/v1/acoustic/enrolled?machine_type=fan").json()
    assert any(e["name"] == "cnc-a1" for e in listed)
    assert client.delete("/api/v1/acoustic/enrolled/fan/cnc-a1").status_code == 200
    assert client.delete("/api/v1/acoustic/enrolled/fan/cnc-a1").status_code == 404
    bad = client.post("/api/v1/acoustic/enroll", files=[("files", ("x.wav", b"junk", "audio/wav"))], data={"machine_type": "fan"})
    assert bad.status_code == 422


def test_alerts_survive_restart(models_dir, tmp_path):
    from fastapi.testclient import TestClient

    from telltale_api.core.config import Settings
    from telltale_api.main import create_app

    settings = Settings(models_dir=models_dir, data_dir=tmp_path, simulator_interval_s=3600)
    with TestClient(create_app(settings)) as c:
        alert = c.app.state.telltale.alerts.create(machine_id="cnc-a1", machine_name="CNC Mill A1", source="sensor",
                                                   severity="critical", title="persist me", detail="d")
        c.post(f"/api/v1/alerts/{alert.id}/acknowledge")
    assert (tmp_path / "state" / "telltale.db").is_file()
    with TestClient(create_app(settings)) as c:  # a fresh app = a restart
        again = [a for a in c.get("/api/v1/alerts").json() if a["id"] == alert.id]
        assert again and again[0]["status"] == "acknowledged" and again[0]["title"] == "persist me"


def test_tool_life_countdown(client):
    state = client.app.state.telltale
    for _ in range(5):  # a few ticks of history so the pace can be measured
        state.fleet.tick()
    life = client.get("/api/v1/machines/cnc-a2").json()["tool_life"]
    assert life["replace_at_min"] == 200
    assert life["remaining_use_min"] == max(0, 200 - life["wear_min"])
    assert life["eta_minutes"] is None or life["eta_minutes"] >= 0


def test_bearing_diagnose_uses_rpm(client):
    clip = wav(tone(300, seconds=3.0))
    auto = client.post("/api/v1/acoustic/diagnose", files={"file": ("b.wav", clip, "audio/wav")}).json()
    given = client.post("/api/v1/acoustic/diagnose", files={"file": ("b.wav", clip, "audio/wav")}, data={"rpm": "1800"}).json()
    from_machine = client.post("/api/v1/acoustic/diagnose", files={"file": ("b.wav", clip, "audio/wav")},
                               data={"machine_id": "cnc-a1"}).json()
    assert auto["rpm_assumed"] is True and auto["rpm_used"] == 1200
    assert given["rpm_assumed"] is False and given["rpm_used"] == 1800
    live = client.get("/api/v1/machines/cnc-a1").json()["latest_reading"]["rotational_speed_rpm"]
    assert from_machine["rpm_used"] == live
