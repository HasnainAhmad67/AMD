# 3-minute demo script

A tested path through the app that shows its strongest moments. Practise it twice. Timings are approximate.

## Before you go on stage (10 minutes earlier)

- [ ] `.\telltale start` (Mac/Linux: `bash telltale.sh start`). Let it run **at least 6 minutes** first, so the simulated machines drift into trouble and raise alerts.
- [ ] Models page: sensor, sound (fan) and bearing all show **loaded**.
- [ ] Laptop volume up; browser zoom 110–125% so the back row can read it.
- [ ] Allow the microphone once (Sound lab → Record), so the permission pop-up doesn't interrupt the demo.
- [ ] A healthy fan nearby (laptop fan, desk fan) for the live recording. Optional, but it's the wow moment.

---

## 0:00 — The problem (15 s, no clicks)

> "Machines warn you before they break: hotter, louder, more strain. Those warnings are spread over many signals and many machines, so they get missed. Telltale watches the telemetry **and** listens to the machines, and tells technicians what to check first."

## 0:15 — Fleet view (30 s)

Open **Fleet**.

> "Eight machines streaming live telemetry. They're ranked by health; the critical ones are on top."

Point at a **critical** card (often *CNC Mill A2*, *Lathe B1* or *Drill Cell C1*):

> "This one's failure risk is 90%+, and it shows *why*: overstrain or heat dissipation. Here's also a tool-change countdown."

## 0:45 — Explain one machine (45 s)

Click the critical machine → **Risk & explanation** tab.

> "Failure modes, and the reason for the estimate. Red bars push the risk up: here it's tool wear × torque. These are exact SHAP values from the model, not guesses."

Show **Recommended inspection**:

> "And what to check, from an inspection playbook that engineers can edit without code."

Click **What-if** → choose *Torque*, drag the slider:

> "What if we reduce the load? The risk drops below the alert line at around X Nm."

Click **Replace tool** (or **Service cooling**) at the top:

> "When the technician acts, the alert resolves and the risk falls back down live."

## 1:30 — Listen to machines (45 s)

Open **Sound lab** → *Anomaly check* → **🎙 Record 10s** next to a healthy fan (or pick a sample clip).

> "It learned what *normal* sounds like, and says how unusual this clip is. Not a raw score: 'more unusual than 38% of normal recordings', which is a calibrated p-value. The red line shows *when* it sounded odd, and the chart shows *which frequencies* differ."

If someone asks why the model doesn't flag everything: *"With the section unknown, which is how real uploads work, its alarms are calibrated to about 5% false alarms on familiar conditions. For a new machine, you teach it the machine's normal sound with about 20 clips."* Show the **Teach this machine's normal sound** panel on a machine page.

## 2:15 — Diagnose a bearing (30 s)

**Sound lab** → *Bearing diagnosis* → pick the **Inner race defect** sample.

> "A second model names the fault. Watch this: it says **'not sure: ball or inner race'**. When the model can't tell, it says so, and the set still contains the right answer about 90% of the time. That's conformal prediction: honest uncertainty instead of a confident wrong answer."

Then pick **Outer race defect** → confident and correct.

## 2:45 — Built on AMD + honesty (15 s)

Open **Models**.

> "Every model shows what it was trained on and how it scored on data it never saw, including the weak spots. Training runs on AMD Instinct GPUs through ROCm, and the server runs everything through ONNX, without PyTorch."

---

## If something goes wrong

| Problem | Do this |
|---|---|
| No alerts yet / all machines healthy | Open *Lathe B1* or *CNC Mill A2*; the risk trend chart still shows the climb. Or wait 2 minutes. |
| Microphone blocked or room too noisy | Use **Try a sample clip** instead: `fan_anomaly_1` (flagged, p ≈ 0.01) and `fan_normal_0` (normal). Not every faulty clip is caught (`fan_anomaly_0` reads normal); if a judge picks one, that's the honest accuracy, see the next section. |
| Browser shows "server is not running" | `.\telltale start` again; it takes about 20 s if already built. |
| Wi-Fi is down | Everything runs locally; nothing needs the internet once it's built. |

## Questions you'll probably get

- **"Is the data real?"** The sensor data (AI4I 2020) is a public *synthetic* dataset with rule-based labels, so its scores are optimistic, and the Models page says so. The sound data (MIMII DUE) and the bearing data are real recordings.
- **"How accurate is the sound model?"** With the section unknown, the DCASE score is 0.58–0.61 across all five machine types (fan, pump, slider, valve, gearbox), better than the official autoencoder baseline on fan. It catches many faults but not all, and it is weakest under operating conditions it barely saw in training; teaching it a machine's normal sound removes most false alarms there. See `docs/model-improvements.md` for every experiment, including what didn't work.
- **"Where does the LLM come in?"** Optional: it turns model outputs into a short briefing for technicians. It never decides the risk.
