"""
Ground-truth comparison, copied near-verbatim from the BI-Bench paper's own harness
(Hu-Chuxuan/bi-agent, tools/run_large_models.py: _parse_value, _safe_compare, _is_scalar,
_is_missingish, _clean_flat, _normalize_df, compare_dataframes, compare_with_timeout).

Reusing their exact matching logic (order-independent, tolerant float compare, $/% parsing,
best-match cell pairing) means a pass/fail here means the same thing as a pass/fail in the
paper's own results/*.csv, so NEXUS's pilot score is comparable to their baseline numbers
rather than being a bespoke metric that merely sounds similar.
"""
import math
from collections.abc import Iterable
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd

MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "-", "--"}


def _parse_value(val):
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("$"):
            try:
                return float(val.replace("$", ""))
            except ValueError:
                return val
        if val.endswith("%"):
            try:
                return float(val.replace("%", "")) / 100
            except ValueError:
                return val
        try:
            return float(val)
        except ValueError:
            return val
    return val


def _safe_compare(a, b, rtol=1e-3, atol=0.5) -> bool:
    a, b = _parse_value(a), _parse_value(b)
    try:
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(a) == pd.to_datetime(b)
        except Exception:
            return str(a).strip() == str(b).strip()


def _is_scalar(val):
    return not isinstance(val, Iterable) or isinstance(val, (str, bytes))


def _is_missingish(x) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in MISSING_STRINGS:
        return True
    try:
        if math.isinf(float(x)):
            return True
    except Exception:
        pass
    return False


def _clean_flat(vals):
    return [0.0 if (_is_scalar(v) and _is_missingish(v)) else v for v in vals]


def _normalize_df(df, max_rows=100):
    df = df.copy().sort_values(
        by=list(df.columns), key=lambda col: col.astype(str), kind="mergesort"
    )
    return df.iloc[:max_rows].reset_index(drop=True)


def compare_dataframes(gt_list, df2) -> bool:
    for df1 in gt_list:
        if df1.shape == (1, 1):
            gt_val = df1.iloc[0, 0]
            for v2 in _clean_flat(df2.values.flatten().tolist()):
                if _safe_compare(gt_val, v2):
                    return True
            continue
        if df1.shape != df2.shape:
            continue
        flat1 = _clean_flat(_normalize_df(df1).values.flatten().tolist())
        flat2 = _clean_flat(_normalize_df(df2).values.flatten().tolist())
        used = [False] * len(flat2)
        all_matched = True
        for v1 in flat1:
            best_idx, best_score = -1, float("inf")
            for i, v2 in enumerate(flat2):
                if used[i]:
                    continue
                if _is_missingish(v1) and _is_missingish(v2):
                    best_idx, best_score = i, 0
                    break
                try:
                    a, b = float(v1), float(v2)
                    if np.isclose(a, b, rtol=1e-2, atol=0.5):
                        diff = abs(a - b)
                        if diff < best_score:
                            best_score, best_idx = diff, i
                except Exception:
                    if _safe_compare(v1, v2):
                        best_idx, best_score = i, 0
                        break
            if best_idx != -1:
                used[best_idx] = True
            else:
                all_matched = False
                break
        if all_matched and all(used):
            return True
    return False


def compare_with_timeout(gt_list, result, timeout=120) -> bool:
    def _target(q):
        try:
            q.put(compare_dataframes(gt_list, result))
        except Exception:
            q.put(False)

    q = Queue()
    p = Process(target=_target, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    return q.get()
