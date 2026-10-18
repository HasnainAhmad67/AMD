# Datasets

All raw data lives in `data/raw/` (git-ignored). `telltale-download list` prints this table.

| Key | Source | Size | Licence | Used by |
|---|---|---|---|---|
| `ai4i` | [UCI #601: AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601) | 0.5 MB | CC BY 4.0 | Sensor failure-risk model |
| `mimii-due` | [Zenodo 4740355: MIMII DUE](https://zenodo.org/records/4740355) | ~1.1 GB per machine type | **CC BY-NC-SA 4.0** | Sound anomaly detector (default) |
| `mimii` | [Zenodo 3384388: MIMII](https://zenodo.org/records/3384388) | 7–11 GB per type and SNR | CC BY-SA 4.0 | Sound anomaly detector (more data) |
| `bearing` | [Mendeley n9y9c7xrz3: rolling-element bearing sounds](https://data.mendeley.com/datasets/n9y9c7xrz3/1) | 75 MB | CC BY 4.0 | Bearing fault classifier |

> **Licence note:** MIMII DUE is **non-commercial** (CC BY-NC-SA). That is fine for a hackathon, but switch to MIMII (CC BY-SA) before any commercial use.

---

## AI4I 2020 (sensor)

10,000 rows, one per machining process. Columns: `UDI, Product ID, Type (L/M/H), Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF`. About 3.4% of rows are failures.

Failure modes come from published rules, which we also use for the demo simulator (`ml/src/telltale_ml/sensor/synthetic.py`):

| Code | Mode | Rule |
|---|---|---|
| TWF | Tool wear | Tool is replaced or fails at a random wear between 200 and 240 min |
| HDF | Heat dissipation | (process − air temperature) < 8.6 K **and** speed < 1380 rpm |
| PWF | Power | torque × ω < 3500 W **or** > 9000 W |
| OSF | Overstrain | tool wear × torque > 11,000 (L) / 12,000 (M) / 13,000 (H) Nm·min |
| RNF | Random | 0.1% at random. **Not modelled**, because there is nothing to learn. |

**Honesty caveat:** the data is synthetic and the labels follow rules. Because we engineer `temp_delta_k`, `power_w` and `strain_nm_min`, the model can recover those rules almost exactly. High scores here say the pipeline works; they do not prove real-world accuracy. Say so in the pitch.

## MIMII DUE (sound) — recommended

This is the DCASE 2021 Task 2 development and additional-training set. It covers 5 machine types: `fan, gearbox, pump, slider, valve`. Each type has sections `00–05`, and each section has a **source** and a **target** domain (different speed, load or noise). Clips are 10 s, mono, 16 kHz.

```
data/raw/mimii_due/<type>/train/section_00_source_train_normal_0000_<attrs>.wav
data/raw/mimii_due/<type>/source_test/section_00_source_test_anomaly_0003.wav
data/raw/mimii_due/<type>/target_test/section_02_target_test_normal_0011.wav
```

Labels come from the file names. Training uses `train/` (normal clips only). Evaluation uses `*_test/`. Metrics are reported overall and per `section/domain`, so domain-shift weakness is visible.

**Results (fan):** the shipped "ensemble" model (ArcFace CNN embeddings + classical spectra, nearest-neighbour scoring) reaches a DCASE score of about **0.60–0.65** with the section *unknown*. The autoencoder baseline reaches 0.58, and the official DCASE 2021 AE baseline about 0.66 / 0.62 source/target AUC. Full experiment log: [`model-improvements.md`](model-improvements.md).

## MIMII (sound) — bigger, older

There are 4 types (`fan, pump, slider, valve`), model IDs `00, 02, 04, 06`, and 3 SNR levels (−6, 0, 6 dB of factory noise). Clips are 10 s, **8-channel**, 16 kHz; we use channel 0.

```
data/raw/mimii/6dB/<type>/id_00/normal/00000000.wav
data/raw/mimii/6dB/<type>/id_00/abnormal/00000000.wav
```

There is no official split. For each machine ID we hold out as many normal clips as there are abnormal ones, which is the DCASE 2020 convention.

## Bearing sounds (Mendeley) — supervised faults

Sound recordings of SKF 6004 ball bearings on a Gunt PT 500 test rig, at 44.1 kHz. `telltale-download bearing` fetches the whole 75 MB archive automatically through Mendeley's public download endpoint.

```
data/raw/bearing_sound/Sound Datasets of a Rolling Element Bearing under Various Operating Conditions/
  Datasets 1- Data acquisition using the stethoscope/
    Case 1 - Inner race fault/       First speed - 648 rpm.mat, Second speed - 1236rpm.mat, Third speed - 1812rpm.mat
    Case 2 - Outer race fault/       ...
    Case 3 - Ball fault/
    Case 4 - Healthy bearing/
    Case 5 - Looseness fault/        6-Looseness-600rpm.mat, ...
    Case 6 - Combined faults (Inner, outer, ball)/
  Datasets 2- Data acquisition without using the stethoscope/
    (same 6 cases, files like 1-Inner-600rpm-ws.mat)
```

- **Format:** MATLAB v5 `.mat`, one variable `x` (float64, already scaled to ±1), 2–20 s long. The files store no sample rate; the dataset page says 44.1 kHz. `load_audio` reads `.mat` directly.
- **Classes (6):** normal, inner race, outer race, ball, looseness, combined. Each has 6 recordings (2 setups × 3 speeds). The "Combined faults (Inner, outer, ball)" folder matches three keywords, so `combined` is marked `match_first` in `ml/configs/bearing.yaml`.
- **Watch out:** the two setups differ about 5× in loudness. Features are therefore loudness-independent (spectral shape, envelope spectrum, crest factor, kurtosis…), and training adds simulated-microphone copies (random EQ/gain/noise). A first version with absolute levels learned the setup instead of the fault.
- **Prediction sets:** conformal calibration on out-of-fold predictions. The API returns every class that can't be ruled out at 90% coverage.

**Evaluation** (`telltale-train-bearing`):

| Test | What it answers | Result (current model) |
|---|---|---|
| Leave-one-condition-out (6 folds; condition = setup + speed band) | "A speed/setup combination it has never heard" | ~84% segment accuracy, ~82% of whole recordings (mean of 3 seeds; chance ≈ 17%) |
| Leave-one-setup-out (2 folds) | "A different microphone setup" | ~45% (range 39–50% across seeds; 32% before augmentation + envelope features). Prediction sets still contain the true class ~94–97% of the time |

The final model is trained on 5 of the 6 conditions. The held-out one (`demo_holdout: "1-1200"`) provides the demo clips in `data/samples/bearing_<class>_0.wav`, so predictions on them in the UI are genuine.
