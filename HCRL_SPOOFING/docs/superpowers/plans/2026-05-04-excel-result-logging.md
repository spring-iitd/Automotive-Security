# Excel Result Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After each of the 6 evaluate scripts runs, append one row to `results/spoof_CH_results.xlsx` capturing attack budget metrics, confusion matrix values, ASR, and Evasion_Rate.

**Architecture:** A shared `scripts/result_logger.py` module exposes a single `log_result()` function. Each evaluate script calls it at the end of `run()`. Budget values (I/M/Pi/Pm/D) are derived internally from round number and total attack count. Baseline (round=-1) is looked up from the Excel itself when computing ASR for subsequent rounds.

**Tech Stack:** Python, openpyxl

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `scripts/result_logger.py` | `log_result()` — open/create Excel, compute metrics, append/update row |
| Create | `tests/test_result_logger.py` | Unit tests for logger |
| Modify (end of `run()`) | `scripts/evaluate_spoof_CH_resnet.py` | Add `log_result(...)` call; model="RI_ResNet", round var=`pass_num` |
| Modify (end of `run()`) | `scripts/evaluate_spoof_CH_resnet50.py` | model="ResNet50", round var=`rounds` |
| Modify (end of `run()`) | `scripts/evaluate_spoof_CH_densenet161.py` | model="DenseNet161", round var=`rounds` |
| Modify (end of `run()`) | `scripts/evaluate_gear_CH_MULSAM.py` | model="MULSAM", round var=`pass_num` |
| Modify (end of `run()`) | `scripts/evaluate_spoof_Entropy.py` | model="Entropy_IDS", round var=`pass_num` |
| Modify (end of `run()`) | `scripts/evaluate_spoof_seq_based.py` | model="Seq_IDS", round var=`rounds` |

---

## Task 1: Create `result_logger.py` with tests (TDD)

**Files:**
- Create: `HCRL_SPOOFING/tests/test_result_logger.py`
- Create: `HCRL_SPOOFING/scripts/result_logger.py`

### Step 1.1 — Write failing tests

Create `HCRL_SPOOFING/tests/test_result_logger.py`:

```python
import os
import sys
import math
import tempfile
import openpyxl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.result_logger import log_result

HEADER = ["round", "model", "I", "M", "Pi", "Pm", "D", "D_left",
          "TP", "FP", "TN", "FN", "total_attack", "total_benign",
          "ASR", "Evasion_Rate"]


def _rows(path):
    wb = openpyxl.load_workbook(path)
    return list(wb.active.iter_rows(values_only=True))


def test_baseline_creates_file_with_header():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)  # ensure it doesn't exist yet
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        rows = _rows(path)
        assert rows[0] == tuple(HEADER)
    finally:
        if os.path.exists(path): os.unlink(path)


def test_baseline_row_values():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        rows = _rows(path)
        r = rows[1]
        assert r[0] == -1           # round
        assert r[1] == "RI_ResNet"  # model
        assert r[2] == 0            # I (round -1)
        assert r[6] == 0            # D (round -1)
        assert r[8] == 100          # TP
        assert r[9] == 10           # FP
        assert r[10] == 200         # TN
        assert r[11] == 5           # FN
        assert r[12] == 105         # total_attack = TP+FN
        assert r[13] == 210         # total_benign = TN+FP
        assert r[14] == 0.0         # ASR = 0.0 for baseline
        assert abs(r[15] - 5/105) < 1e-6  # Evasion_Rate = FN/(FN+TP)
    finally:
        if os.path.exists(path): os.unlink(path)


def test_adversarial_round_asr():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        log_result(path, 1,  "RI_ResNet", TP=20,  TN=200, FP=10, FN=85)
        rows = _rows(path)
        assert len(rows) == 3  # header + 2 data rows
        r1 = rows[2]
        assert r1[0] == 1
        # ASR = (100 - 20) / 100 = 0.8
        assert abs(r1[14] - 0.8) < 1e-6
        # Evasion_Rate = 85 / (85 + 20) = 85/105
        assert abs(r1[15] - 85/105) < 1e-6
    finally:
        if os.path.exists(path): os.unlink(path)


def test_round1_budgets():
    """Round 1: I=total_attack, M=0, Pi=5, Pm=0, D=0"""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        log_result(path, 1,  "RI_ResNet", TP=20,  TN=200, FP=10, FN=85)
        rows = _rows(path)
        r1 = rows[2]
        total_attack = 20 + 85  # TP + FN for round 1
        assert r1[2] == total_attack   # I = total_attack
        assert r1[3] == 0              # M = 0
        assert r1[4] == 5              # Pi = 5
        assert r1[5] == 0              # Pm = 0
        assert r1[6] == 0              # D = 0
    finally:
        if os.path.exists(path): os.unlink(path)


def test_duplicate_guard_updates_row():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        log_result(path, 1,  "RI_ResNet", TP=20,  TN=200, FP=10, FN=85)
        # Re-log same round+model with different TP — should UPDATE not duplicate
        log_result(path, 1,  "RI_ResNet", TP=25,  TN=200, FP=10, FN=80)
        rows = _rows(path)
        assert len(rows) == 3       # still 3 rows (header + 2), not 4
        assert rows[2][8] == 25     # TP updated
    finally:
        if os.path.exists(path): os.unlink(path)


def test_missing_baseline_raises():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        with pytest.raises(ValueError, match="Baseline row not found"):
            log_result(path, 1, "RI_ResNet", TP=20, TN=200, FP=10, FN=85)
    finally:
        if os.path.exists(path): os.unlink(path)


def test_round0_d_budget():
    """Round 0: D=1, all others 0"""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        log_result(path, 0,  "RI_ResNet", TP=90,  TN=200, FP=10, FN=15)
        rows = _rows(path)
        r0 = rows[2]
        assert r0[2] == 0   # I=0
        assert r0[3] == 0   # M=0
        assert r0[4] == 0   # Pi=0
        assert r0[5] == 0   # Pm=0
        assert r0[6] == 1   # D=1
    finally:
        if os.path.exists(path): os.unlink(path)
```

- [ ] **Step 1.2: Run tests — verify they all FAIL**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -m pytest tests/test_result_logger.py -v 2>&1 | head -40
```
Expected: `ModuleNotFoundError: No module named 'scripts.result_logger'`

- [ ] **Step 1.3: Implement `scripts/result_logger.py`**

Create `HCRL_SPOOFING/scripts/result_logger.py`:

```python
import os
import math
import openpyxl

HEADER = ["round", "model", "I", "M", "Pi", "Pm", "D", "D_left",
          "TP", "FP", "TN", "FN", "total_attack", "total_benign",
          "ASR", "Evasion_Rate"]

# Column indices (0-based) for lookup
_COL_ROUND = 0
_COL_MODEL = 1
_COL_TP    = 8
_COL_FN    = 11


def _budgets(round_num, n):
    """Return (I, M, Pi, Pm, D) for this round. n = total attack frames."""
    n4 = math.ceil(0.25 * n) if n > 0 else 0
    table = {
        -1: (0,  0,   0,   0,  0),
        0:  (0,  0,   0,   0,  1),
        1:  (n,  0,   5,   0,  0),
        2:  (0,  n4,  5,   0,  0),
        3:  (0,  0,   n,   n4, 0),
    }
    return table.get(round_num, (0, 0, 0, 0, 0))


def _read_baseline(ws, model_name):
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[_COL_ROUND] == -1 and row[_COL_MODEL] == model_name:
            return int(row[_COL_TP]), int(row[_COL_FN])
    raise ValueError(
        f"Baseline row not found for model '{model_name}'. "
        "Run round=-1 evaluation first."
    )


def _find_existing_row(ws, round_num, model_name):
    """Return 1-based row index if (round, model) exists, else None."""
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if row[_COL_ROUND] == round_num and row[_COL_MODEL] == model_name:
            return i
    return None


def log_result(excel_path, round_num, model_name, TP, TN, FP, FN):
    """
    Append or update one result row in the shared Excel file.

    Parameters
    ----------
    excel_path  : str   path to spoof_CH_results.xlsx (created if absent)
    round_num   : int   -1 = baseline, 0-3 = adversarial rounds
    model_name  : str   e.g. "RI_ResNet", "ResNet50"
    TP, TN, FP, FN : int  confusion matrix values (frame-level)
    """
    os.makedirs(os.path.dirname(os.path.abspath(excel_path)), exist_ok=True)

    if os.path.exists(excel_path):
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(HEADER)

    total_attack = TP + FN
    total_benign = TN + FP
    I, M, Pi, Pm, D = _budgets(round_num, total_attack)
    D_left = total_attack

    if round_num == -1:
        ASR = 0.0
    else:
        baseline_TP, baseline_FN = _read_baseline(ws, model_name)
        ASR = (baseline_TP - TP) / baseline_TP if baseline_TP > 0 else 0.0

    evasion = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    new_row = [round_num, model_name, I, M, Pi, Pm, D, D_left,
               TP, FP, TN, FN, total_attack, total_benign,
               round(ASR, 6), round(evasion, 6)]

    existing = _find_existing_row(ws, round_num, model_name)
    if existing is not None:
        for col_idx, val in enumerate(new_row, start=1):
            ws.cell(row=existing, column=col_idx, value=val)
    else:
        ws.append(new_row)

    wb.save(excel_path)
    print(f"[result_logger] Saved round={round_num} model={model_name} "
          f"ASR={ASR:.4f} Evasion={evasion:.4f} -> {excel_path}")
```

- [ ] **Step 1.4: Run tests — verify they all PASS**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -m pytest tests/test_result_logger.py -v
```
Expected: `7 passed`

- [ ] **Step 1.5: Commit**

```bash
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline add \
    HCRL_SPOOFING/scripts/result_logger.py \
    HCRL_SPOOFING/tests/test_result_logger.py
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline commit -m \
    "feat: add result_logger utility for shared Excel logging"
```

---

## Task 2: Wire `evaluate_spoof_CH_resnet.py` (RI_ResNet)

**Files:**
- Modify: `HCRL_SPOOFING/scripts/evaluate_spoof_CH_resnet.py` — add 4 lines at end of `run()`

The existing last line of `run()` (line 332):
```python
    save_preds(pass_num,tracksheet,traffic_rows,output_path,preds)
```

- [ ] **Step 2.1: Add import and log_result call**

In `evaluate_spoof_CH_resnet.py`, append after the `save_preds(...)` call at the end of `run()`:

```python
    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=pass_num,
        model_name="RI_ResNet",
        TP=TP, TN=TN, FP=FP, FN=FN,
    )
```

- [ ] **Step 2.2: Smoke-test (dry run)**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -c "
from scripts.evaluate_spoof_CH_resnet import run
# minimal params that trigger the log without full model load
print('import OK')
"
```
Expected: `import OK` (no ImportError)

- [ ] **Step 2.3: Commit**

```bash
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline add \
    HCRL_SPOOFING/scripts/evaluate_spoof_CH_resnet.py
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline commit -m \
    "feat: log RI_ResNet results to Excel after evaluation"
```

---

## Task 3: Wire `evaluate_spoof_CH_resnet50.py` (ResNet50)

**Files:**
- Modify: `HCRL_SPOOFING/scripts/evaluate_spoof_CH_resnet50.py` — add 4 lines at end of `run()`

The existing last line of `run()` (line 260):
```python
    save_preds(rounds, tracksheet, preds_dict, output_path)
```

- [ ] **Step 3.1: Add import and log_result call**

Append after `save_preds(rounds, tracksheet, preds_dict, output_path)` at end of `run()`:

```python
    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=rounds,
        model_name="ResNet50",
        TP=TP, TN=TN, FP=FP, FN=FN,
    )
```

- [ ] **Step 3.2: Smoke-test**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -c "from scripts.evaluate_spoof_CH_resnet50 import run; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 3.3: Commit**

```bash
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline add \
    HCRL_SPOOFING/scripts/evaluate_spoof_CH_resnet50.py
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline commit -m \
    "feat: log ResNet50 results to Excel after evaluation"
```

---

## Task 4: Wire `evaluate_spoof_CH_densenet161.py` (DenseNet161)

**Files:**
- Modify: `HCRL_SPOOFING/scripts/evaluate_spoof_CH_densenet161.py` — add 4 lines at end of `run()`

The existing last line of `run()` (line 282):
```python
    save_preds(rounds, tracksheet, preds_dict, output_path)
```

- [ ] **Step 4.1: Add import and log_result call**

Append after `save_preds(rounds, tracksheet, preds_dict, output_path)` at end of `run()`:

```python
    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=rounds,
        model_name="DenseNet161",
        TP=TP, TN=TN, FP=FP, FN=FN,
    )
```

- [ ] **Step 4.2: Smoke-test**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -c "from scripts.evaluate_spoof_CH_densenet161 import run; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 4.3: Commit**

```bash
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline add \
    HCRL_SPOOFING/scripts/evaluate_spoof_CH_densenet161.py
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline commit -m \
    "feat: log DenseNet161 results to Excel after evaluation"
```

---

## Task 5: Wire `evaluate_gear_CH_MULSAM.py` (MULSAM)

**Files:**
- Modify: `HCRL_SPOOFING/scripts/evaluate_gear_CH_MULSAM.py` — add 4 lines at end of `run()`

The existing last line of `run()`:
```python
    save_preds(pass_num, tracksheet, traffic_rows, output_path, preds)
```
`TP, TN, FP, FN` are available from `plot_confusion()` return value earlier in `run()`.

- [ ] **Step 5.1: Add import and log_result call**

Append after `save_preds(pass_num, tracksheet, traffic_rows, output_path, preds)` at end of `run()`:

```python
    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=pass_num,
        model_name="MULSAM",
        TP=TP, TN=TN, FP=FP, FN=FN,
    )
```

- [ ] **Step 5.2: Smoke-test**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -c "from scripts.evaluate_gear_CH_MULSAM import run; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 5.3: Commit**

```bash
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline add \
    HCRL_SPOOFING/scripts/evaluate_gear_CH_MULSAM.py
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline commit -m \
    "feat: log MULSAM results to Excel after evaluation"
```

---

## Task 6: Wire `evaluate_spoof_Entropy.py` (Entropy_IDS)

**Files:**
- Modify: `HCRL_SPOOFING/scripts/evaluate_spoof_Entropy.py` — add 4 lines at end of `run()`

The existing last line of `run()` (line 297):
```python
    save_preds(pass_num, tracksheet, df, window_indices, preds, output_path)
```
`TP, TN, FP, FN` available from `plot_confusion()` return at line 286.

- [ ] **Step 6.1: Add import and log_result call**

Append after `save_preds(pass_num, tracksheet, df, window_indices, preds, output_path)` at end of `run()`:

```python
    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=pass_num,
        model_name="Entropy_IDS",
        TP=TP, TN=TN, FP=FP, FN=FN,
    )
```

- [ ] **Step 6.2: Smoke-test**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -c "from scripts.evaluate_spoof_Entropy import run; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 6.3: Commit**

```bash
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline add \
    HCRL_SPOOFING/scripts/evaluate_spoof_Entropy.py
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline commit -m \
    "feat: log Entropy_IDS results to Excel after evaluation"
```

---

## Task 7: Wire `evaluate_spoof_seq_based.py` (Seq_IDS)

**Files:**
- Modify: `HCRL_SPOOFING/scripts/evaluate_spoof_seq_based.py` — add 4 lines at end of `run()`

The existing last line of `run()` (line 325):
```python
    save_preds(rounds, traffic_path, tracksheet, output_path, labels)
```
`TP, TN, FP, FN` available from `test()` return at line 315:
```python
    labels, TP, TN, FP, FN = test(perturbed_data, rounds, validated_tm, unique_ids)
```

- [ ] **Step 7.1: Add import and log_result call**

Append after `save_preds(rounds, traffic_path, tracksheet, output_path, labels)` at end of `run()`:

```python
    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=rounds,
        model_name="Seq_IDS",
        TP=TP, TN=TN, FP=FP, FN=FN,
    )
```

- [ ] **Step 7.2: Smoke-test**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -c "from scripts.evaluate_spoof_seq_based import run; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 7.3: Final test suite run**

```bash
cd /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline/HCRL_SPOOFING
python -m pytest tests/test_result_logger.py -v
```
Expected: `7 passed`

- [ ] **Step 7.4: Commit**

```bash
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline add \
    HCRL_SPOOFING/scripts/evaluate_spoof_seq_based.py
git -C /DATA/scratch/ipali/WISA/plaid_final/plaid_pipeline commit -m \
    "feat: log Seq_IDS results to Excel after evaluation"
```

---

## Usage After Implementation

**Run baseline for each model first (round=-1), then rounds 0-3 in order.**

The Excel `results/spoof_CH_results.xlsx` will have one row per `(round, model)` combination, with columns:
```
round | model | I | M | Pi | Pm | D | D_left | TP | FP | TN | FN | total_attack | total_benign | ASR | Evasion_Rate
```
