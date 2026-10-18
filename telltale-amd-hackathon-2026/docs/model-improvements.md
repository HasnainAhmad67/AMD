# Model improvements: research notes and experiments

This page records what we tried to make the models work better on **unseen data**, what worked, and what didn't. Every number comes from held-out data. Read the "how we measured" notes before quoting anything.

> TL;DR
> - **Sound anomaly (fan):** the new "ensemble" model raises the DCASE score from **0.58 → ~0.62** (source-domain AUC 0.66 → 0.75–0.78), measured *without* telling the model which section a clip came from.
> - **Bearing faults:** accuracy on a **new microphone setup rose from 32% → ~45%** (seed-dependent, 39–50%). On unseen speeds it rose to **~84% (82% of recordings)** with speed-aware order analysis. The model also says "not sure: inner race or ball" when it can't tell (conformal prediction sets).
> - **Calibrated outputs:** sound results are now conformal p-values ("more unusual than 98% of normal recordings") with a guaranteed false-alarm rate.
> - **New operating conditions:** "teach it this machine's normal sound" (enroll ~20 healthy clips, no retraining) cut false alarms there from **58% → 5%**, matching the guarantee.

---

## How we measured (so we don't fool ourselves)

- **Sound:** official MIMII DUE fan dev test set (sections 00–02, source and target domains, 100 normal + 100 anomalous clips each). Per (section, domain) we compute AUC and pAUC (FPR ≤ 0.1). The **DCASE score** is their harmonic mean. Headline numbers use **section-unknown scoring**: the scorer compares each clip with *all* sections' normal clips, exactly like an upload in the app. That's harder than the usual DCASE setting where the section is known.
- **Bearings:** leave-one-condition-out (hold out each setup + speed combination) and leave-one-setup-out (train on the stethoscope recordings, test on the others, and vice versa). Recordings are never split between train and test. Augmented copies only ever go on the training side.
- **No tuning on test:** thresholds and conformal calibration come from held-out *training* data. The ensemble weights (0.5/0.5) were fixed before looking at results.
- **Caveat:** we compared several methods on the same dev test set, so the winning method's score is slightly optimistic (selection bias). The truly unseen check is the DCASE *evaluation* set, or recordings you make yourself.

---

## Sound anomaly detection (MIMII DUE, fan)

### What the research says
- From DCASE 2021 onward, strong systems **train a classifier to tell recording conditions (sections, attributes) apart** and use its embeddings for anomaly scoring, instead of reconstructing the sound with an autoencoder. Angular-margin losses (ArcFace, sub-cluster AdaCos) work well. Scoring uses nearest-neighbour or GMM distances, computed **per domain** to handle domain shift.
- Recent top systems (DCASE 2024/25) add large pretrained audio models such as BEATs, fine-tuned with LoRA.
- Audio-language models / LLMs (e.g. CLAP zero-shot): we found no benchmark where they beat specialised detectors on machine anomalous sound detection. They help *explain* results, which our optional LLM briefing already does.

### Experiments (DCASE score = harmonic mean of AUC/pAUC; higher is better)

| # | Method | Section known? | AUC source | AUC target | pAUC | **DCASE score** |
|---|---|---|---|---|---|---|
| 0 | Autoencoder (DCASE baseline, what we shipped first) | no (one model) | 0.658 | 0.612 | 0.534 | **0.576** |
| 1 | Average log-mel spectrum + domain-wise nearest neighbour (no deep learning!) | yes | 0.738 | 0.669 | 0.541 | 0.606 |
| 1b | same, mean+std / Mahalanobis variants | yes | 0.61–0.67 | 0.55–0.65 | 0.50–0.52 | 0.53–0.58 |
| 2 | ArcFace CNN embeddings + domain-wise kNN (3 seeds) | yes | 0.72–0.77 | 0.57–0.65 | 0.58–0.65 | 0.59–0.65 |
| 3 | 1 + 2 combined (0.5/0.5, z-normalised on training data) | yes | 0.786 | 0.671 | 0.640 | 0.669 |
| 4 | **1 + 2 combined, section unknown** (what the app does) | **no** | 0.75–0.78 | 0.58–0.65 | 0.56–0.61 | **0.60–0.65** |
| 4b | same, average of 3 CNN seeds | no | 0.781 | 0.585 | 0.587 | 0.618 |
| 5 | CNN trained on 12 conditions (+ MIMII DUE extra sections 03–05), 2 seeds | yes | 0.69–0.72 | 0.60 | 0.60 | 0.59–0.60 ✗ |
| 5b | same + classical, section unknown (banks: sections 00–02) | no | 0.74–0.75 | 0.50–0.51 | 0.55–0.57 | 0.55–0.57 ✗ |
| 6 | **Frozen pretrained AST** (Audio Spectrogram Transformer, AudioSet) embeddings alone | no | 0.55–0.57 | 0.53–0.55 | 0.50–0.51 | 0.51–0.53 ✗ |
| 6b | shipped CNN + classical + frozen AST (1/3 each) | no | 0.68–0.70 | 0.59–0.60 | 0.55 | 0.58–0.59 ✗ (shipped on the same scoring: 0.597) |

What we learned:
- **Simple statistics plus the right scoring rule already beat the autoencoder.** Domain-wise nearest-neighbour scoring is the key trick: target-domain clips are compared with the few target examples instead of the 1,000 source examples.
- **The CNN and the statistics fail on different sections** (statistics: good on section 00; CNN: good on 01 and 02), so combining them helps.
- **One CNN's quality varies a lot between training runs** (0.59–0.65). Averaging 3 seeds removes the worst case but didn't beat a typical single run, so the default is 1 CNN (`ensemble.n_models` in `ml/configs/acoustic.yaml`). Use 3 on an AMD GPU if you want steadier results.
- Not knowing the section costs about 0.03. That's the honest price of scoring uploads with no metadata.
- **A frozen pretrained audio model did not help.** AST embeddings were close to chance and lowered the ensemble (exp. 6). AST learned to recognise *what* a sound is ("fan", "engine"), not small changes in the *state* of one familiar machine. The leading DCASE systems **fine-tune** pretrained models (e.g. BEATs with LoRA) on the section/attribute task instead of using them frozen. Fine-tuning an 86M-parameter model was not feasible on our laptop CPU (1.3 s per clip just to run it), but it's a good AMD GPU job.
- **More training classes did not help.** Adding the 6 extra training-only conditions made the CNN *worse* on the target domain (exp. 5). The trainer therefore ignores sections without labelled test clips by default (`ensemble.only_evaluated_groups`).

### Unseen operating conditions: the hard part
The target domain (other speeds, loads or background noise, with only 3 training examples per section) is where the model struggles. With the shipped fan model, at the alert level p ≤ 0.05:

| Domain | False alarms on normal clips | Detected anomalies |
|---|---|---|
| Source (familiar conditions) | 6.7% (promised 5%: calibration works) | 27% |
| Target (new conditions) | **57%** | 68% *(meaningless with that many false alarms)* |

**Fix: few-shot enrollment plus per-condition (Mondrian) conformal calibration.** Normal clips of the new condition join the bank, with no retraining. With ≥ 19 clips, that condition gets its own calibration from leave-one-out scores. (With n calibration scores the smallest possible p-value is 1/(n+1), so 19 is the minimum for p ≤ 0.05.) Measured through the real detector, enrolling 20 normal clips per target condition and testing on all remaining clips:

| Target condition | AUC before → after | False alarms before → after | Detections before → after |
|---|---|---|---|
| Section 00 | 0.673 → 0.644 | 16% → 16% | 32% → 22% |
| Section 01 | 0.581 → 0.557 | 64% → 0% | 76% → 1% |
| Section 02 | 0.527 → **0.743** | 95% → 0% | 97% → 4% |
| **All** | 0.594 → 0.648 | **58% → 5.4%** | 68% → 9% |

Alerts become trustworthy (false-alarm rate matches the 5% promise), and the ranking improves a lot where it was worst. The honest cost: at a strict 5% false-alarm rate this detector catches few anomalies in new conditions. Use the **watch** level (p ≤ 0.15) for earlier warnings, and improve the model itself for more. In the app this is the **"Teach this machine's normal sound"** panel on each machine's Sound check tab (`POST /api/v1/acoustic/enroll`).

### What's shipped (`method: ensemble`)
`ml/src/telltale_ml/acoustic/{embedding.py, train_ensemble.py, knn_scorer.py, detector.py}`
1. Small 2-D CNN on log-mel spectrograms, trained with an **ArcFace** loss to classify section × domain. It uses random 6 s crops and frequency masking, and oversamples the rare target-domain clips. Exported to **ONNX**, so the API stays torch-free.
2. Classical branch: the clip's average log-mel spectrum, standardised.
3. Banks of normal clips per (section, domain). Score = minimum over banks of the leave-one-out-z-normalised nearest-neighbour distance, averaged across the two branches.
4. **Conformal p-values** from 10% of normal clips held out for calibration. `p ≤ 0.05` → anomalous, `p ≤ 0.15` → watch.
5. A sliding 2 s window timeline shows *when* the clip sounded odd.
6. Optional enrollment of a machine's own normal clips with per-condition (Mondrian) calibration (see above).

### Results per machine type (shipped models, same config, section unknown)
The same settings (`ml/configs/acoustic.yaml`, one CNN, 20 epochs, CPU) were trained per machine type without any per-type tuning. The numbers come from each model's `metadata.json` → `test_metrics` (DCASE dev test split, 1,200 clips per type).

| Type | DCASE score | AUC source | AUC target | pAUC | Train time (CPU) |
|---|---|---|---|---|---|
| fan | 0.605 | 0.747 | 0.601 | 0.505 | ~20 min |
| pump | 0.602 | 0.690 | 0.630 | 0.517 | ~16 min |
| slider | 0.580 | 0.792 | **0.476** | 0.501 | ~16 min |
| valve | 0.595 | 0.732 | 0.547 | 0.537 | ~22 min |
| gearbox | 0.599 | 0.623 | **0.725** | 0.538 | ~16 min |

Pump does better on the target domain than fan does, but worse on the source domain. Its weakest group is section 02 (AUC ≈ 0.63 in both domains). Slider is the clearest case of the domain-shift problem: it has the best source-domain AUC so far (0.84 on sections 01 and 02) but is **below chance on the target domain** (0.41–0.53). For slider machines under new conditions, enrollment is not optional. Valve varies most between sections: 0.92 on section 02 source, but 0.58 on section 01 in both domains. Gearbox is the only type where the *target* domain is easier than the source (0.73 vs 0.62), so "new conditions are always worse" is a tendency, not a rule. The demo clips (`telltale-download samples`) are now taken round-robin from each section's source-domain test clips, picked by position, never by score. One seed per type, so differences of ±0.03 are within CNN seed noise (see the gotcha in AGENTS.md).

### Ideas not done yet (good team projects, best on the AMD GPU)
- **Other machine types as extra classes** (pump, valve, …). Extra *fan* sections hurt (exp. 5), but other machine types are a different kind of extra data that DCASE systems commonly use.
- **Fine-tuned pretrained audio models** (BEATs, EAT, AST + LoRA) trained on the section classification task. *Frozen* AST was tested and hurt (exp. 6), so fine-tuning is the part that matters. Needs a GPU.
- **Sub-cluster AdaCos or GMM scoring** instead of 1-nearest-neighbour.
- **Attribute labels:** MIMII DUE file names hold operating attributes; classifying them too gives richer embeddings.

---

## Bearing fault classification (Mendeley bearing sounds)

### What the research says
- Treating vibration/sound as audio and reusing **pretrained audio models** (YAMNet, large audio models with LoRA) transfers well to fault diagnosis.
- **Domain generalisation** for acoustic fault diagnosis manipulates the Fourier amplitude/phase or augments the spectrum, so models ignore the recording channel.
- Classic machinery diagnostics uses the **envelope spectrum**: bearing defects produce periodic impacts whose repetition rate shows up after demodulation.

### Experiments (accuracy on held-out data; chance ≈ 17%)

| Variant | Unseen speed + setup: segments / recordings | **New mic setup: segments / recordings** |
|---|---|---|
| Absolute-level features (first version) | 33% (single split) | — |
| Loudness-independent features | 81% / 72% | 32% / 31% |
| + per-recording spectral normalisation | 68% / 72% | 27% / 28% ✗ *(removes the fault signature too)* |
| + microphone augmentation (random EQ, gain, noise) | 74% / 75% | 49% / 44% |
| envelope spectrum only | 66% / 64% | 52% / 42% |
| loudness-independent + envelope + augmentation | 74% / 72% | 50% / 47% (single run) |
| *same, mean of 3 seeds* | *76% / 74%* | *45% / 45%* |
| **+ order spectrum (envelope in shaft orders; needs rpm), mean of 3 seeds (shipped)** | **84% / 82%** | **46% / 45%** |

We traded a little accuracy on familiar setups for much better robustness on a new microphone, the case that matters for new recordings. **Order analysis** then won back the familiar-setup accuracy and more: bearing defects tick at fixed multiples of the shaft speed, so expressing the envelope spectrum in orders lines them up across speeds. It improved every one of 3 seeds by 7–8 points. It needs the rpm: the app uses a fleet machine's live speed, or an optional field in the Sound lab; if unknown, it assumes 1200 rpm and says so.

**Lesson:** single-run results on 36 recordings are noisy. The new-mic score ranged from 39% to 50% across seeds. Compare variants over several seeds.

### Conformal prediction sets (shipped)
Out-of-fold predictions calibrate a threshold so the returned **set of classes** contains the true class 90% of the time. On average the set has 2.4 classes. On a *different microphone setup* it still contained the true class **94%** of the time. In the app, an unsure diagnosis reads: *"Not sure: with 90% certainty it is one of Ball element defect or Inner race defect."*

### Ideas not done yet
- Record a few minutes of your own bearings or machines with phones. More recording setups beat any algorithm for cross-microphone accuracy.
- Pretrained audio embeddings plus a small classifier.

---

## Engineering lessons from this round
- **Downloads silently truncated on bad Wi-Fi:** the server dropped the connection without an error and the downloader saved a 320 MB piece of a 1.08 GB zip as if it were complete. The downloader now checks sizes, resumes with HTTP Range, retries while there's progress, and validates zips. On the next download it survived 4 drops and a 504 in a row.
- **API image 1.17 GB → 842 MB:** a two-stage build removes the test folders packages ship (~180 MB) and strips debug symbols from native libraries. The biggest remaining piece is llvmlite (librosa → numba), which can only go once the audio features are reimplemented without librosa (that would need retraining).

## Sensor model (AI4I 2020)
Already near the ceiling on AI4I (PR-AUC 0.94), and the data is synthetic. The improvement that matters for real machines is **real telemetry**, plus trend features (how fast wear and temperature rise), which would also make tool-wear failures predictable. Weak failure-mode models are already gated: shown as low-confidence and never alerting.

---

## Sources
- DCASE 2021 Task 2 description: <https://arxiv.org/pdf/2106.04492>
- DCASE 2024 Task 2 description: <https://arxiv.org/abs/2406.07250>, DCASE 2025: <https://arxiv.org/pdf/2506.10097>
- Sub-cluster AdaCos (DCASE 2021, Wilkinghoff): <https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Wilkinghoff_31_t2.pdf>
- ArcFace on section labels (DCASE 2022): <https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Venkatesh_99_t2.pdf>
- BEATs-based DCASE 2024 system (AITHU): <https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Lv_110_t2.pdf>
- Large audio model for bearing diagnosis (AeroGPT): <https://arxiv.org/pdf/2506.16225>
- Frequency-guided domain generalisation for acoustic fault diagnosis: <https://pmc.ncbi.nlm.nih.gov/articles/PMC13165849/>
- Conformal process monitoring: <https://arxiv.org/pdf/2512.23602>; conformal calibration review: <https://arxiv.org/pdf/2512.17048>
