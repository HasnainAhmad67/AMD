"""A brand-new checkout has no models: the API must train the sensor model by itself."""

import time

from fastapi.testclient import TestClient

from telltale_api.core.config import Settings
from telltale_api.main import create_app


def test_sensor_model_is_trained_automatically_on_first_start(tmp_path):
    settings = Settings(
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        simulator_interval_s=3600,
        sensor_autotrain_offline=True,  # tests must not need the internet
    )
    with TestClient(create_app(settings)) as client:
        deadline = time.time() + 180
        health = client.get("/api/v1/health").json()
        while not health["models"]["sensor"] and time.time() < deadline:
            assert health["sensor_bootstrap"]["status"] in {"running", "done"}, health
            time.sleep(1)
            health = client.get("/api/v1/health").json()

        assert health["models"]["sensor"], health
        assert health["sensor_bootstrap"]["status"] == "done"
        assert (tmp_path / "models" / "sensor" / "metadata.json").is_file()
        caveats = client.get("/api/v1/models").json()["sensor"]["caveats"]
        assert "SIMULATED" in caveats[0]


def test_autotrain_can_be_switched_off(tmp_path):
    settings = Settings(models_dir=tmp_path, data_dir=tmp_path, simulator_interval_s=3600, sensor_autotrain=False)
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/v1/health").json()
        assert health["models"]["sensor"] is False
        assert health["sensor_bootstrap"]["status"] == "idle"
        assert client.post("/api/v1/sensors/predict", json={}).status_code == 422
