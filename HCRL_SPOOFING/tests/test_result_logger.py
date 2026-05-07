import os
import sys
import math
import tempfile
import openpyxl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.result_logger import log_result, HEADER


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
        assert r[HEADER.index("round")] == -1
        assert r[HEADER.index("model")] == "RI_ResNet"
        assert r[HEADER.index("I")] == 0            # I (round -1)
        assert r[HEADER.index("D")] == 0            # D (round -1)
        assert r[HEADER.index("TP")] == 100
        assert r[HEADER.index("FP")] == 10
        assert r[HEADER.index("TN")] == 200
        assert r[HEADER.index("FN")] == 5
        assert r[HEADER.index("total_attack")] == 105   # total_attack = TP+FN
        assert r[HEADER.index("total_benign")] == 210   # total_benign = TN+FP
        assert r[HEADER.index("ASR")] == 0.0            # ASR = 0.0 for baseline
        assert abs(r[HEADER.index("Evasion_Rate")] - 5/105) < 1e-6  # FN/(FN+TP)
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
        assert r1[HEADER.index("round")] == 1
        # ASR = (100 - 20) / 100 = 0.8
        assert abs(r1[HEADER.index("ASR")] - 0.8) < 1e-6
        # Evasion_Rate = 85 / (85 + 20) = 85/105
        assert abs(r1[HEADER.index("Evasion_Rate")] - 85/105) < 1e-6
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
        assert r1[HEADER.index("I")]  == total_attack   # I = total_attack
        assert r1[HEADER.index("M")]  == 0
        assert r1[HEADER.index("Pi")] == 5
        assert r1[HEADER.index("Pm")] == 0
        assert r1[HEADER.index("D")]  == 0
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
        assert rows[2][HEADER.index("TP")] == 25     # TP updated
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
        assert r0[HEADER.index("I")]  == 0
        assert r0[HEADER.index("M")]  == 0
        assert r0[HEADER.index("Pi")] == 0
        assert r0[HEADER.index("Pm")] == 0
        assert r0[HEADER.index("D")]  == 1
    finally:
        if os.path.exists(path): os.unlink(path)


def test_round2_budgets():
    """Round 2: I=0, M=ceil(0.25*n), Pi=5, Pm=0, D=0"""
    import math
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        log_result(path, 2,  "RI_ResNet", TP=30,  TN=200, FP=10, FN=75)
        rows = _rows(path)
        r2 = rows[2]
        total_attack = 30 + 75
        assert r2[HEADER.index("I")]  == 0
        assert r2[HEADER.index("M")]  == math.ceil(0.25 * total_attack)
        assert r2[HEADER.index("Pi")] == 5
        assert r2[HEADER.index("Pm")] == 0
        assert r2[HEADER.index("D")]  == 0
    finally:
        if os.path.exists(path): os.unlink(path)


def test_round3_budgets():
    """Round 3: I=0, M=0, Pi=n, Pm=ceil(0.25*n), D=0"""
    import math
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        log_result(path, -1, "RI_ResNet", TP=100, TN=200, FP=10, FN=5)
        log_result(path, 3,  "RI_ResNet", TP=10,  TN=200, FP=10, FN=95)
        rows = _rows(path)
        r3 = rows[2]
        total_attack = 10 + 95
        assert r3[HEADER.index("I")]  == 0
        assert r3[HEADER.index("M")]  == 0
        assert r3[HEADER.index("Pi")] == total_attack
        assert r3[HEADER.index("Pm")] == math.ceil(0.25 * total_attack)
        assert r3[HEADER.index("D")]  == 0
    finally:
        if os.path.exists(path): os.unlink(path)
