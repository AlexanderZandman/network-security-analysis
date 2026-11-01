# Network Security Analysis — ML Botnet Detection for Suricata

Detects IoT botnet traffic with machine learning on top of [Suricata](https://suricata.io/).
Suricata turns network traffic (a pcap file or a live interface) into flow events in `eve.json`.
This program follows that file, computes features per flow, classifies each flow with an
XGBoost model trained on the [IoT-23](https://www.stratosphereips.org/datasets-iot23) dataset,
and writes alerts in Suricata's own formats (`fast.log` and `eve.json` style), so they can be
consumed by the same tooling as regular Suricata alerts.

## How it works

```
pcap / interface ──► Suricata ──► logs/eve.json ──► main.py ──► ml_fast.log
                                                     │          ml_eve.json
                                                     ▼
                                    feature extraction + XGBoost model
```

1. **Suricata** (`scripts/run_suricata.*`, `botnet-detection/config/suricata.yaml`) writes flow
   events to `botnet-detection/logs/eve.json`.
2. **`main.py`** tails `eve.json` in batches using the `suricata/eve_consumer.py` module. When it reaches the end
   of the file it waits for new lines; if the file is truncated it starts over.
3. **`BotnetDetector`** (`botnetdetector.py`) maps Suricata flow events to the Zeek-style columns
   the model was trained on (`suricata/events_mapping.py`) and computes features
   (`model/features.py`). Some features count earlier flows in the same 1-minute time window, so
   recent flows are kept as context.
4. Flows classified as malicious are written as alerts by `suricata/suricata_event_logger.py`.
   ML alerts use generator id `9000000`, so they are easy to tell apart from Suricata's own
   signatures. Each attack type has a fixed signature id:

   | sid | Attack type                      |
   |-----|----------------------------------|
   | 1   | C&C                              |
   | 2   | Attack                           |
   | 3   | DDoS                             |
   | 4   | PartOfAHorizontalPortScan        |
   | 5   | PartOfAHorizontalPortScan-Attack |
   | 6   | Okiru                            |
   | 7   | Okiru-Attack                     |
   | 8   | FileDownload                     |
   | 99  | Unknown / other                  |

### Models

Pretrained models are in `botnet-detection/model/saved-models/`. Choose one with `--model`:

- **`everything`** (default): one multiclass classifier (`everything_classifier.pkl`) that predicts
  `Benign` or an attack type directly. All `C&C-*` labels are merged into `C&C`.
- **`split`**: a binary botnet-vs-benign detector (`botnet_detector.pkl`), followed by an attack
  type classifier (`attack_classifier.pkl`) for flows flagged as botnet.

## Project layout

```
botnet-detection/
├── main.py                    # entry point: follow eve.json and raise alerts
├── botnetdetector.py          # batches flows, keeps time-window context, runs the model
├── config/suricata.yaml       # Suricata configuration used by the scripts
├── suricata/                  # eve.json reader, event parsing/mapping, alert logger
├── model/
│   ├── features.py            # dataset loading, label mapping, feature engineering
│   ├── train.py               # train and save all models
│   ├── evaluate.py            # cross-validated evaluation
│   ├── predict.py             # load models and predict
│   └── saved-models/          # pretrained models and class names
├── data_retrieval.py          # IoT-23 Zeek conn.log.labeled ──► Parquet
├── sampling.py                # stratified sample of the Parquet data
├── data_processing.py         # exploratory analysis / plots
└── botnet-exploration.ipynb   # exploration notebook
scripts/run_suricata.{sh,ps1,bat}  # run Suricata on a pcap, logs into botnet-detection/logs
```

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip)
- [Suricata](https://docs.suricata.io/en/latest/install.html) on your `PATH`
- For training: the [IoT-23 dataset](https://www.stratosphereips.org/datasets-iot23), extracted
  into `botnet-detection/datasets/`

Running the detector with the pretrained models needs no dataset: the categorical values seen
during training (such as `proto`) are stored in the models themselves.

## Installation

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running

### 1. Generate flow events with Suricata

Run Suricata on a pcap. Logs always go to `botnet-detection/logs/`, whatever the current directory:

```bash
# macOS / Linux
scripts/run_suricata.sh path/to/capture.pcap

# Windows (PowerShell)
scripts/run_suricata.ps1 path\to\capture.pcap
```

Without an argument, the scripts use a default IoT-23 capture
(`CTU-IoT-Malware-Capture-52-1`) from `botnet-detection/datasets/`. Extra arguments are passed
on to Suricata.

### 2. Run the detector

Run from the `botnet-detection` directory:

```bash
cd botnet-detection
uv run python main.py
```

This reads `logs/eve.json` from the start, then keeps following it. Alerts are appended to
`logs/ml_fast.log` and `logs/ml_eve.json`. Stop with `Ctrl+C`.

Options:

| Option               | Default                  | Description                                                   |
|----------------------|--------------------------|---------------------------------------------------------------|
| `-i`, `--input`      | `logs/eve.json`          | Suricata `eve.json` to follow                                 |
| `--from-end`         | off                      | Skip existing lines; only analyze lines appended from now on  |
| `-b`, `--batch-size` | `1000`                   | Maximum number of lines analyzed at once                      |
| `--interval`         | `3`                      | Seconds to wait for new lines at the end of the file          |
| `--threshold`        | `0.5`                    | Botnet probability threshold                                  |
| `--fast-log`         | `<input dir>/ml_fast.log`| Where to append `fast.log`-style alerts                       |
| `--alert-eve`        | `<input dir>/ml_eve.json`| Where to append `eve.json`-style alerts                       |
| `--model`            | `everything`             | `everything` (single classifier) or `split` (detector + attack classifier) |

Example: live monitoring with the split models and a stricter threshold:

```bash
uv run python main.py --from-end --model split --threshold 0.8
```

## Preparing the data

Run these from the `botnet-detection` directory.

1. Download IoT-23 and extract it so the scenarios are in
   `datasets/opt/Malware-Project/BigDataset/IoTScenarios/<scenario>/bro/conn.log.labeled`.
2. Convert the labeled Zeek logs to Parquet (one file per scenario in `datasets/iot23_parquet/`):

   ```bash
   uv run python data_retrieval.py
   ```

3. Draw a stratified sample that keeps every attack type
   (`datasets/sample/iot23_sample.parquet`):

   ```bash
   uv run python sampling.py
   ```

## Training and evaluation

Run from the `botnet-detection` directory as modules:

```bash
uv run python -m model.train      # trains all three models, writes model/saved-models/
uv run python -m model.evaluate   # cross-validated evaluation reports
```

`model/train.py` reads its training data from
`datasets/combined/combined-training-trimmed.parquet`. Training uses grouped cross-validation, so
the reported scores are out-of-fold. Results of earlier experiments (time windows, seeds) are in
`model/BINARY_notes.txt` and `model/ATTACK_notes.txt`.

After retraining, the class name files (`attack_classes.json`, `everything_classes.json`) are
written next to the models; the detector needs them to name the predicted attack types.
