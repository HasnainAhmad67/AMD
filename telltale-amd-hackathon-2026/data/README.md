# data — downloaded datasets

You never need to put anything here by hand. The helper commands fill this folder for you, and it is **not saved in git** because the files are big.

| Folder | What's inside | Filled by |
|---|---|---|
| `raw/ai4i2020/` | Sensor dataset (0.5 MB) | Automatically on first start, or `.\telltale train-sensor` |
| `raw/mimii_due/` | Machine sound recordings (~1.1 GB per machine type) | `.\telltale train-sound fan` |
| `raw/bearing_sound/` | Bearing sound recordings (MATLAB `.mat` files, 75 MB) | `.\telltale train-bearing` |
| `raw/_archives/` | Downloaded zip files (kept so nothing is downloaded twice) | Automatic |
| `samples/` | Short demo clips for the website's "Try a sample clip" menus (`fan_normal_0.wav`, `bearing_ball_0.wav`, ...) | `.\telltale train-sound <type>` and `.\telltale train-bearing` |
| `processed/` | Intermediate files from training | Automatic |
| `state/` | The app's alert history (`telltale.db`), so alerts survive restarts | Automatic |

Running out of disk space? You can delete `raw/_archives/` after training.
