"""Test fixtures: train tiny models on synthetic data into a temp dir, then boot the app against it."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from telltale_ml.paths import default_config
from telltale_ml.sensor.synthetic import generate
from telltale_ml.sensor.train import train_sensor_models
from telltale_ml.utils import load_config

from telltale_api.core.config import Settings
from telltale_api.main import create_app

SR = 16_000


def tone(freq: float, seconds: float = 2.0, seed: int = 0, noise: float = 0.01) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


def wav(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV")
    return buf.getvalue()


def _train_tiny_bearing_model(root, tmp_path_factory):
    """3 classes x 2 setups x 3 speeds of synthetic .mat recordings in the real dataset's folder layout."""
    import os

    import scipy.io as sio
    from telltale_ml.acoustic.train_fault import discover_labelled, train_fault_model

    raw = tmp_path_factory.mktemp("bearing_raw")
    t = np.arange(44_100 * 2) / 44_100
    for setup in ("Datasets 1- with stethoscope", "Datasets 2- without stethoscope"):
        for case, freq in (("Case 1 - Inner race fault", 900), ("Case 4 - Healthy bearing", 300), ("Case 3 - Ball fault", 2500)):
            folder = raw / setup / case
            folder.mkdir(parents=True)
            for rpm in (600, 1200, 1800):
                sio.savemat(folder / f"{rpm}rpm.mat", {"x": (0.4 * np.sin(2 * np.pi * freq * t))[:, None]})
    cfg = load_config(default_config("bearing"))
    cfg["lightgbm"]["n_estimators"] = 30
    files, unlabelled = discover_labelled(raw, cfg["classes"])
    old = os.environ.get("TELLTALE_DATA_DIR")
    os.environ["TELLTALE_DATA_DIR"] = str(tmp_path_factory.mktemp("bearing_data"))  # demo clips go here
    try:
        train_fault_model(files, cfg, root / "bearing", unlabelled)
    finally:
        if old is None:
            os.environ.pop("TELLTALE_DATA_DIR", None)
        else:
            os.environ["TELLTALE_DATA_DIR"] = old


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    cfg = load_config(default_config("sensor"))
    cfg["lightgbm"]["n_estimators"] = 120
    train_sensor_models(generate(5000, seed=11), cfg, root / "sensor", "synthetic")

    _train_tiny_bearing_model(root, tmp_path_factory)

    # Tiny v2 ("ensemble") sound model if torch is available; otherwise acoustic tests are skipped.
    try:
        import torch  # noqa: F401
        from telltale_ml.acoustic.data import Clip
        from telltale_ml.acoustic.train_ensemble import train_ensemble_model
    except ImportError:
        return root
    clip_dir = tmp_path_factory.mktemp("clips")
    clips = []
    for i in range(14):
        p = clip_dir / f"n{i}.wav"
        sf.write(p, tone(400, seconds=3.0, seed=i), SR)
        clips.append(Clip(p, "normal", "train" if i < 11 else "test", "section_00"))
    for i in range(2):
        p = clip_dir / f"a{i}.wav"
        sf.write(p, tone(400, seconds=3.0) + tone(2500, seconds=3.0, noise=0.3, seed=5 + i), SR)
        clips.append(Clip(p, "anomaly", "test", "section_00"))
    cfg = {
        "seed": 0, "method": "ensemble",
        "features": {"sample_rate": SR, "n_mels": 32},
        "train": {"epochs": 2, "device": "cpu", "num_workers": 1},
        "ensemble": {"calibration_fraction": 0.2, "batch_size": 8, "crop_frames": 32, "emb_dim": 16,
                     "window_frames": 32, "window_hop_frames": 16},
        "threshold_percentile": 95, "watch_fraction": 0.85, "max_audio_seconds": 30,
    }
    train_ensemble_model(clips, cfg, root / "acoustic" / "fan", "synthetic", "fan")
    return root


@pytest.fixture()
def client(models_dir, tmp_path):
    settings = Settings(
        models_dir=models_dir,
        data_dir=tmp_path,
        simulator_enabled=True,
        simulator_interval_s=3600,  # tick once at startup, then stay still
        llm_base_url=None,
    )
    with TestClient(create_app(settings)) as c:
        yield c
