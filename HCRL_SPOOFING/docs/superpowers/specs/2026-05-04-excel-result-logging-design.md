# Excel Result Logging — HCRL_SPOOFING CH Spoof

## Goal

After each evaluate script runs, append one row to a shared Excel file so all 6 NIDS models and all rounds are captured in one place with attack budget metrics alongside detection metrics.

---

## Excel File

**Path:** `HCRL_SPOOFING/results/spoof_CH_results.xlsx`

**Columns (in order):**

| Column | Source | Formula |
|--------|--------|---------|
| `round` | passed in | round number (-1 = baseline) |
| `model` | hardcoded per script | e.g. "RI_ResNet", "ResNet50" |
| `I` | hardcoded per round | injection budget |
| `M` | hardcoded per round | modification budget |
| `Pi` | hardcoded per round | prev-injection retry budget |
| `Pm` | hardcoded per round | prev-modification retry budget |
| `D` | hardcoded per round | deletion budget |
| `D_left` | computed | attack packets surviving after deletion = `TP + FN` (round 0 only; same as `total_attack` for rounds where D=0) |
| `TP` | confusion matrix | true positives (attack frames detected) |
| `FP` | confusion matrix | false positives (benign frames flagged) |
| `TN` | confusion matrix | true negatives |
| `FN` | confusion matrix | false negatives (attack frames missed) |
| `total_attack` | computed | `TP + FN` |
| `total_benign` | computed | `TN + FP` |
| `ASR` | computed | `(TP_baseline - TP_adv) / TP_baseline` — relative drop from baseline |
| `Evasion_Rate` | computed | `FN / (FN + TP)` — fraction of attack frames that evaded detection this round |

**Baseline row:** round = -1. `ASR = 0.0`, `Evasion_Rate = FN/(FN+TP)`. Must be logged for each model before running rounds 0–3.

**Duplicate guard:** if a row with the same `(round, model)` already exists, update it in place rather than append a duplicate.

---

## Budget Values (hardcoded per round)

| round | I | M | Pi | Pm | D |
|-------|---|---|----|----|---|
| -1    | 0 | 0 | 0  | 0  | 0 |
| 0     | 0 | 0 | 0  | 0  | 1 |
| 1     | x | 0 | 5  | 0  | 0 |
| 2     | 0 | n/4 | 5 | 0 | 0 |
| 3     | 0 | 0 | x  | n/4 | 0 |

`x` = variable (dependent on attack image count); `n/4` = ceil(ablation_frac × n_attack).  
For rounds where the budget is variable, pass the actual value explicitly when calling `log_result`.

---

## Shared Utility: `scripts/result_logger.py`

Single public function — no baseline params needed; reads baseline from Excel itself:

```python
def log_result(
    excel_path: str,
    round_num: int,
    model_name: str,
    I: int, M: int, Pi: int, Pm: int, D: int,
    TP: int, FP: int, TN: int, FN: int,
):
```

Internally:
1. Ensure `results/` directory exists.
2. Load workbook if exists; otherwise create with header row.
3. Check for existing row where `round == round_num` and `model == model_name`.
4. Compute `total_attack = TP+FN`, `total_benign = TN+FP`, `D_left = TP+FN`.
5. If `round_num == -1`: `ASR = 0.0`, `Evasion_Rate = FN/(FN+TP)`.
6. Else: look up round=-1 row for same model → read `baseline_TP`, `baseline_FN` → compute `ASR` and `Evasion_Rate`. Raise error if baseline row missing.
7. Append or update row, save.

Library: `openpyxl`.

## Baseline Run (per model, once)

Each evaluate script is run once with `round=-1` on **original unperturbed test data**:

| Model | Baseline input |
|-------|---------------|
| RI_ResNet | `CAN_DATA/gear_test.csv` decoded traffic |
| ResNet50 / DenseNet161 | `gear_test_images/` + `labels.txt` |
| MULSAM / Entropy_IDS / Seq_IDS | original test CSV |

---

## Changes to Each Evaluate Script

At the end of each script's `run()` function, after computing `TP, TN, FP, FN`:

```python
from scripts.result_logger import log_result

BUDGETS = {-1:(0,0,0,0,0), 0:(0,0,0,0,1), 1:(x,0,5,0,0), 2:(0,n4,5,0,0), 3:(0,0,x,n4,0)}
I, M, Pi, Pm, D = BUDGETS[pass_num]

log_result(
    excel_path="results/spoof_CH_results.xlsx",
    round_num=pass_num,
    model_name="RI_ResNet",   # hardcoded per script
    I=I, M=M, Pi=Pi, Pm=Pm, D=D,
    TP=TP, FP=FP, TN=TN, FN=FN,
)
```

Baseline is looked up automatically from the Excel's round=-1 row for the same model.

---

## 6 Models and Their Script Files

| Model | Script |
|-------|--------|
| RI_ResNet | `evaluate_spoof_CH_resnet.py` |
| ResNet50 | `evaluate_spoof_CH_resnet50.py` |
| DenseNet161 | `evaluate_spoof_CH_densenet161.py` |
| MULSAM | `evaluate_gear_CH_MULSAM.py` |
| Entropy_IDS | `evaluate_spoof_Entropy.py` |
| Seq_IDS | `evaluate_spoof_seq_based.py` |

Each gets an identical `log_result(...)` call at the end of `run()` with its own `model_name` string.

---

## No Overwrite Risk

- The Excel file is opened, modified (append or update one row), and saved each time.
- Scripts run sequentially — no concurrent write conflict.
- `cf_stats_{round}.json` files are unaffected; this design does not depend on them.
