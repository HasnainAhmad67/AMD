import math

import pandas as pd
import pytest

from telltale_ml.paths import default_config
from telltale_ml.sensor.features import engineer
from telltale_ml.sensor.predictor import SensorPredictor
from telltale_ml.sensor.schema import FEATURES
from telltale_ml.sensor.synthetic import generate
from telltale_ml.sensor.train import train_sensor_models
from telltale_ml.utils import load_config

READING = {
    "product_quality": "M",
    "air_temperature_k": 300.0,
    "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 100.0,
}


def test_engineer_columns_and_physics():
    X = engineer(pd.DataFrame([READING]))
    assert list(X.columns) == list(FEATURES)
    row = X.iloc[0]
    assert row["quality_code"] == 1
    assert row["temp_delta_k"] == pytest.approx(10.0)
    assert row["power_w"] == pytest.approx(40.0 * 1500.0 * 2 * math.pi / 60)
    assert row["strain_nm_min"] == pytest.approx(4000.0)


def test_engineer_rejects_unknown_quality():
    with pytest.raises(ValueError):
        engineer(pd.DataFrame([{**READING, "product_quality": "X"}]))


def test_synthetic_rules_fire():
    df = generate(5000, seed=1)
    assert 0.01 < df["machine_failure"].mean() < 0.25
    for code in ("HDF", "PWF", "OSF"):
        assert df[code].sum() > 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    cfg = load_config(default_config("sensor"))
    cfg["lightgbm"]["n_estimators"] = 150
    out = tmp_path_factory.mktemp("sensor")
    train_sensor_models(generate(6000, seed=3), cfg, out, "synthetic")
    return SensorPredictor(out)


def test_predictor_assess(trained):
    [res] = trained.assess([READING])
    assert 0.0 <= res["failure_probability"] <= 1.0
    assert res["risk_level"] in {"low", "medium", "high"}
    assert {m["code"] for m in res["failure_modes"]} <= {"TWF", "HDF", "PWF", "OSF"}
    assert len(res["contributions"]) == len(FEATURES)


def test_overstrain_is_riskier(trained):
    safe = trained.assess([READING])[0]["failure_probability"]
    risky = trained.assess([{**READING, "torque_nm": 65.0, "tool_wear_min": 230.0}])[0]["failure_probability"]
    assert risky > safe


def test_what_if(trained):
    points = trained.what_if(READING, "torque_nm", [20, 40, 60, 75])
    assert [p["value"] for p in points] == [20, 40, 60, 75]
    with pytest.raises(ValueError):
        trained.what_if(READING, "temp_delta_k", [1.0])


def test_unreliable_modes_never_flag(trained):
    for code, meta in trained.metadata["models"]["modes"].items():
        assert "reliable" in meta and "val_f1" in meta, code
    fresh = SensorPredictor(trained.model_dir)  # don't mutate the shared fixture
    fresh.mode_reliable = {code: False for code in fresh.mode_reliable}
    [res] = fresh.assess([{**READING, "torque_nm": 70.0, "tool_wear_min": 240.0}])
    assert not any(m["flagged"] for m in res["failure_modes"])
    assert all(not m["reliable"] for m in res["failure_modes"])
