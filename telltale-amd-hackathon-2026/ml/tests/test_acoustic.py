import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from telltale_ml.acoustic.data import discover_mimii, discover_mimii_due
from telltale_ml.acoustic.fault import FaultFeatureConfig, label_from_path, segment_features
from telltale_ml.acoustic.features import LogMelConfig, load_audio, log_mel, pool, stack_frames

SR = 16_000


def tone(freq: float, seconds: float = 2.0, sr: int = SR, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


def wav_bytes(y: np.ndarray, sr: int = SR, channels: int = 1) -> io.BytesIO:
    buf = io.BytesIO()
    data = np.stack([y] * channels, axis=1) if channels > 1 else y
    sf.write(buf, data, sr, format="WAV")
    buf.seek(0)
    return buf


def test_stack_frames_layout():
    lm = np.arange(12, dtype=np.float32).reshape(3, 4)  # 3 mels, 4 frames
    vecs = stack_frames(lm, 2)
    assert vecs.shape == (3, 6)
    np.testing.assert_array_equal(vecs[0], [0, 4, 8, 1, 5, 9])
    assert stack_frames(lm, 5).shape == (0, 15)


def test_load_audio_multichannel_and_resample():
    y = load_audio(wav_bytes(tone(440, sr=8000), sr=8000, channels=8), SR, channel=0)
    assert abs(len(y) - 2 * SR) < 10


def test_log_mel_shape():
    cfg = LogMelConfig()
    lm = log_mel(tone(1000), cfg)
    assert lm.shape[0] == cfg.n_mels
    assert stack_frames(lm, cfg.frames).shape[1] == cfg.input_dim


def test_pool():
    assert pool(np.arange(10.0), 5, axis=0).shape == (5,)
    assert pool(np.ones((4, 300)), 100, axis=1).shape == (4, 100)


def test_label_from_path():
    kw = {"inner_race": ["inner", "ir"], "outer_race": ["outer", "or"], "normal": ["normal"]}
    assert label_from_path(Path("Datasets 1/Inner race/rec1.wav"), kw) == "inner_race"
    assert label_from_path(Path("d2/OR_1500rpm.wav"), kw) == "outer_race"
    assert label_from_path(Path("d2/motor.wav"), kw) is None  # "or" must be a whole token
    assert label_from_path(Path("normal/inner.wav"), kw) is None  # ambiguous


def test_segment_features():
    cfg = FaultFeatureConfig(sample_rate=SR)
    feats = segment_features(tone(300, seconds=3.0), cfg)
    assert feats.shape == (5, 2 * cfg.n_mels + 7 + cfg.envelope_bins)
    assert np.isfinite(feats).all()


def test_discover_layouts(tmp_path):
    due = tmp_path / "due" / "dev_data" / "fan"
    (due / "train").mkdir(parents=True)
    (due / "source_test").mkdir()
    sf.write(due / "train" / "section_00_source_train_normal_0000_x.wav", tone(100), SR)
    sf.write(due / "source_test" / "section_00_source_test_anomaly_0000.wav", tone(100), SR)
    clips = discover_mimii_due(tmp_path / "due", "fan")
    assert {(c.split, c.label) for c in clips} == {("train", "normal"), ("test", "anomaly")}

    mimii = tmp_path / "mimii" / "6dB" / "pump" / "id_00"
    for lbl, n in (("normal", 4), ("abnormal", 1)):
        (mimii / lbl).mkdir(parents=True)
        for i in range(n):
            sf.write(mimii / lbl / f"{i:08d}.wav", tone(100), SR)
    clips = discover_mimii(tmp_path / "mimii", "pump")
    assert sum(c.split == "test" and c.label == "normal" for c in clips) == 1
    assert sum(c.split == "train" for c in clips) == 3


def test_train_export_and_detect(tmp_path):
    """End-to-end: tiny autoencoder -> ONNX -> AcousticDetector."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from telltale_ml.acoustic.data import Clip
    from telltale_ml.acoustic.detector import AcousticDetector
    from telltale_ml.acoustic.train_anomaly import train_anomaly_model

    clips = []
    for i in range(12):
        p = tmp_path / f"n{i}.wav"
        sf.write(p, tone(500, seed=i), SR)
        clips.append(Clip(p, "normal", "train" if i < 9 else "test", "id_00"))
    for i in range(3):
        p = tmp_path / f"a{i}.wav"
        sf.write(p, tone(500, seed=100 + i) + tone(3000, noise=0.2, seed=i), SR)
        clips.append(Clip(p, "anomaly", "test", "id_00"))

    cfg = {
        "seed": 0,
        "features": {"sample_rate": SR, "n_mels": 32, "frames": 3},
        "train": {"val_fraction": 0.25, "vectors_per_clip": 32, "epochs": 3, "batch_size": 64,
                  "learning_rate": 1e-3, "device": "cpu", "num_workers": 1},
        "model": {"hidden": 32, "bottleneck": 4, "depth": 2},
        "threshold_percentile": 95, "watch_fraction": 0.85, "max_audio_seconds": 30,
    }
    meta = train_anomaly_model(clips, cfg, tmp_path / "model", "test", "fan")
    assert (tmp_path / "model" / "model.onnx").exists()
    assert meta["test_metrics"]["overall"]["roc_auc"] is not None

    det = AcousticDetector(tmp_path / "model")
    res = det.analyze(wav_bytes(tone(500, seed=50)))
    assert res["status"] in {"normal", "watch", "anomalous"}
    assert len(res["spectrogram"]["values"]) == 32


# --------------------------------------------------------------------------- bearing (Mendeley layout, .mat files)
def make_fake_bearing_dataset(root: Path, sr: int = 44_100, seconds: float = 2.0) -> None:
    """Mimic the real archive: Datasets {1,2} / Case N - <class> / <speed> rpm .mat (variable "x")."""
    import scipy.io as sio

    cases = {"Case 1 - Inner race fault": 900.0, "Case 4 - Healthy bearing": 300.0,
             "Case 6 - Combined faults (Inner, outer, ball)": 2500.0}
    setups = {"Datasets 1- Data acquisition using the stethoscope": 1.0,
              "Datasets 2- Data acquisition without using the stethoscope": 0.2}
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(0)
    for setup, gain in setups.items():
        for case, freq in cases.items():
            folder = root / "Sound Datasets" / setup / case
            folder.mkdir(parents=True)
            for rpm in (612, 1236, 1809):
                x = gain * (0.5 * np.sin(2 * np.pi * freq * (1 + rpm / 20_000) * t) + 0.05 * rng.standard_normal(len(t)))
                sio.savemat(folder / f"speed - {rpm} rpm.mat", {"x": x[:, None]})


def test_label_priority_for_combined_faults():
    kw = {"inner_race": ["inner"], "outer_race": ["outer"], "ball": ["ball"], "combined": ["combined"]}
    path = Path("Datasets 1/Case 6 - Combined faults (Inner, outer, ball)/6-Combined-600rpm.mat")
    assert label_from_path(path, kw) is None  # ambiguous without a priority
    assert label_from_path(path, kw, priority=("combined",)) == "combined"


def test_condition_of():
    from telltale_ml.acoustic.train_fault import condition_of

    ev = {"setup_regex": r"Datasets (\d)", "rpm_regex": r"(\d{3,4})\s*rpm", "rpm_bands": [600, 1200, 1800]}
    assert condition_of(Path("x/Datasets 1- foo/Case 1/Second speed - 1236rpm.mat"), ev) == "1-1200"
    assert condition_of(Path("x/Datasets 2- bar/Case 5/5-Looseness-600rpm-ws.mat"), ev) == "2-600"
    assert condition_of(Path("x/other/clip.wav"), ev) is None


def test_load_audio_reads_matlab_files(tmp_path):
    import scipy.io as sio

    sio.savemat(tmp_path / "rec.mat", {"x": (0.5 * np.ones(44_100))[:, None]})
    y = load_audio(tmp_path / "rec.mat", 22_050)  # .mat files are assumed 44.1 kHz -> resampled
    assert abs(len(y) - 22_050) < 10
    with open(tmp_path / "rec.mat", "rb") as fh:  # uploads arrive as file objects
        assert len(load_audio(io.BytesIO(fh.read()), 44_100)) == 44_100


def test_train_bearing_end_to_end(tmp_path, monkeypatch):
    from telltale_ml.acoustic.fault import FaultClassifier
    from telltale_ml.acoustic.train_fault import discover_labelled, train_fault_model
    from telltale_ml.paths import default_config
    from telltale_ml.utils import load_config

    monkeypatch.setenv("TELLTALE_DATA_DIR", str(tmp_path / "data"))
    make_fake_bearing_dataset(tmp_path / "raw")
    cfg = load_config(default_config("bearing"))
    cfg["lightgbm"]["n_estimators"] = 40
    files, unlabelled = discover_labelled(tmp_path / "raw", cfg["classes"])
    assert not unlabelled and {lbl for _, lbl in files} == {"inner_race", "normal", "combined"}

    meta = train_fault_model(files, cfg, tmp_path / "model", unlabelled)
    assert meta["split_strategy"] == "leave_one_condition_out"
    assert meta["test_metrics"]["folds"] == 6 and meta["test_metrics"]["cross_setup"]["folds"] == 2
    assert meta["demo_holdout"] == "1-1200" and meta["final_model_recordings"] == 15
    assert meta["test_metrics"]["accuracy"] > 0.6  # distinct tones per class are easy
    assert sorted(meta["demo_samples"]) == ["bearing_combined_0.wav", "bearing_inner_race_0.wav", "bearing_normal_0.wav"]
    assert 0 < meta["conformal"]["q"] <= 1 and meta["conformal"]["coverage"] == 0.9

    res = FaultClassifier(tmp_path / "model").diagnose(tmp_path / "data" / "samples" / "bearing_normal_0.wav")
    assert res["predicted_class"] == "normal"
    assert "normal" in res["prediction_set"]


# --------------------------------------------------------------------------- sound model v2 (ensemble)
def test_conformal_p_value_is_calibrated():
    from telltale_ml.acoustic.knn_scorer import conformal_p_value, severity_from_p

    rng = np.random.default_rng(0)
    cal = rng.normal(size=500)
    fresh_normals = rng.normal(size=4000)
    p = np.array([conformal_p_value(s, cal) for s in fresh_normals])
    assert abs((p <= 0.05).mean() - 0.05) < 0.02  # false-alarm rate matches the level for normal data
    assert conformal_p_value(10.0, cal) == 1 / 501  # the smallest possible p-value
    assert severity_from_p(0.05, 0.05) == pytest.approx(1.0)
    assert severity_from_p(1.0, 0.05) == 0.0


def test_knn_scorer_section_agnostic():
    from telltale_ml.acoustic.knn_scorer import EnsembleScorer, l2_normalize, leave_one_out_stats

    rng = np.random.default_rng(1)
    a = l2_normalize(rng.normal(size=(50, 8)) + np.array([5, 0, 0, 0, 0, 0, 0, 0]))
    b = l2_normalize(rng.normal(size=(50, 8)) + np.array([0, 5, 0, 0, 0, 0, 0, 0]))
    ca, cb = rng.normal(size=(50, 4)), rng.normal(size=(50, 4)) + 4
    banks = {"arc0__00__source": a, "arc0__01__source": b, "cls__00__source": ca, "cls__01__source": cb}
    stats = {"arc0": {"00": list(leave_one_out_stats("arc0", a)), "01": list(leave_one_out_stats("arc0", b))},
             "cls": {"00": list(leave_one_out_stats("cls", ca)), "01": list(leave_one_out_stats("cls", cb))}}
    scorer = EnsembleScorer(banks, stats, {"arc": 0.5, "cls": 0.5})
    normal_like = scorer.score({"arc0": b[:5], "cls": cb[:5]})  # resembles section 01 -> low scores, section unknown
    weird = scorer.score({"arc0": l2_normalize(np.tile([0, 0, 0, 0, 0, 0, 0, 5.0], (5, 1))), "cls": np.full((5, 4), -10.0)})
    assert weird.min() > normal_like.max()


def test_train_ensemble_export_and_detect(tmp_path):
    """End-to-end v2: CNN embedder -> ONNX -> banks -> calibration -> AcousticDetector on an uploaded clip."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from telltale_ml.acoustic.data import Clip
    from telltale_ml.acoustic.detector import AcousticDetector
    from telltale_ml.acoustic.train_ensemble import train_ensemble_model

    clips = []
    for sec, f0 in (("section_00", 400), ("section_01", 1200)):
        for i in range(16):
            p = tmp_path / f"{sec}_n{i}.wav"
            sf.write(p, tone(f0, seconds=3.0, seed=i), SR)
            clips.append(Clip(p, "normal", "train" if i < 12 else "test", sec, "source"))
        for i in range(4):
            p = tmp_path / f"{sec}_a{i}.wav"
            sf.write(p, tone(f0, seconds=3.0, seed=50 + i) + tone(3000, seconds=3.0, noise=0.3, seed=i), SR)
            clips.append(Clip(p, "anomaly", "test", sec, "source"))
    cfg = {
        "seed": 0, "method": "ensemble",
        "features": {"sample_rate": SR, "n_mels": 32},
        "train": {"epochs": 2, "device": "cpu", "num_workers": 1},
        "ensemble": {"calibration_fraction": 0.25, "batch_size": 8, "crop_frames": 32, "emb_dim": 16, "n_models": 2,
                     "window_frames": 32, "window_hop_frames": 16},
        "threshold_percentile": 95, "watch_fraction": 0.85, "max_audio_seconds": 30,
    }
    meta = train_ensemble_model(clips, cfg, tmp_path / "model", "test", "fan")
    assert (tmp_path / "model" / "embedder_1.onnx").exists() and (tmp_path / "model" / "bank.npz").exists()
    assert [e["branch"] for e in meta["embedders"]] == ["arc0", "arc1"]
    assert meta["scorer"] == "ensemble" and meta["test_metrics"]["dcase_hmean"] is not None
    assert len(meta["calibration_scores"]) == 6  # 25% of 12 normal clips x 2 sections held out

    det = AcousticDetector(tmp_path / "model")
    res = det.analyze(wav_bytes(tone(400, seconds=3.0, seed=99)))
    assert res["method"] == "ensemble" and 0 < res["p_value"] <= 1
    assert res["status"] in {"normal", "watch", "anomalous"}
    assert len(res["frame_errors"]["values"]) >= 2  # sliding-window timeline


def test_enrollment_adds_condition_and_calibration(tmp_path):
    """Teach a new normal condition: no retraining, persisted, own calibration once >= 19 clips."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from telltale_ml.acoustic.data import Clip
    from telltale_ml.acoustic.detector import MIN_CONDITION_CALIBRATION, AcousticDetector
    from telltale_ml.acoustic.train_ensemble import train_ensemble_model

    clips = []
    for i in range(14):
        p = tmp_path / f"n{i}.wav"
        sf.write(p, tone(400, seconds=2.0, seed=i), SR)
        clips.append(Clip(p, "normal", "train" if i < 11 else "test", "section_00", "source"))
    p = tmp_path / "a0.wav"
    sf.write(p, tone(400, seconds=2.0) + tone(3000, seconds=2.0, noise=0.3), SR)
    clips.append(Clip(p, "anomaly", "test", "section_00", "source"))
    cfg = {"seed": 0, "features": {"sample_rate": SR, "n_mels": 32}, "train": {"epochs": 1, "device": "cpu", "num_workers": 1},
           "ensemble": {"calibration_fraction": 0.2, "batch_size": 8, "crop_frames": 32, "emb_dim": 16,
                        "window_frames": 32, "window_hop_frames": 16},
           "threshold_percentile": 95, "watch_fraction": 0.85, "max_audio_seconds": 30}
    train_ensemble_model(clips, cfg, tmp_path / "model", "test", "fan")

    det = AcousticDetector(tmp_path / "model")
    new_machine = [wav_bytes(tone(1700, seconds=2.0, seed=200 + i)) for i in range(MIN_CONDITION_CALIBRATION + 1)]
    before = det.analyze(wav_bytes(tone(1700, seconds=2.0, seed=999)))
    status = det.enroll("press-7", new_machine[:5])
    assert status["clips"] == 5 and not status["calibrated"]
    status = det.enroll("press-7", new_machine[5:])  # adds to the existing clips
    assert status["clips"] == MIN_CONDITION_CALIBRATION + 1 and status["calibrated"]
    after = det.analyze(wav_bytes(tone(1700, seconds=2.0, seed=999)))
    assert after["score"] < before["score"]  # the new machine's normal sound is now familiar
    assert after["calibration"] == "enrolled:press-7"
    assert AcousticDetector(tmp_path / "model").enrollment_status("press-7")["clips"] == MIN_CONDITION_CALIBRATION + 1
    with pytest.raises(ValueError):
        det.enroll("bad name!", new_machine[:1])
    assert det.clear_enrollment("press-7") and not (tmp_path / "model" / "enrolled" / "press-7.npz").exists()
