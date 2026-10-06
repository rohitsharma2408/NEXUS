"""
Number Verifier — runs AFTER the Business Analyst writes the answer.

Extracts every figure in the answer text and checks it against the evidence bundle
(SQL rows, ML output, RAG chunk text, and numbers in the question itself).

Each figure lands in one of three buckets:
  supported   - matches an evidence value (within the rounding the answer implies)
  derived     - not in the evidence, but equals a sum / difference / ratio / % change of
                two evidence values, or a column total / mean / min / max
  unsupported - matches nothing: possibly invented

Limits (by design): this proves a number EXISTS in the evidence. It cannot tell whether
a real number is attached to the right claim (correct revenue, wrong month).
Bare small integers (<=12) and bare years (1900-2100) are skipped: they are almost always
counts, ordinals or dates, and checking them would only add noise.
"""
import bisect
import re

_NUM = re.compile(
    r"(?<![\w.])(\$)?\s?(-?\d{1,3}(?:,\d{3})+|-?\d+)(\.\d+)?"
    r"(%|\s?(?:million|billion|thousand)\b|[KkMmBb]\b)?"
)
_MULT = {"%": 1.0, "k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
_MAX_BASE = 300  # cap on evidence values used for pairwise derivation

# Dates are not figures: "December 31", "31 December 2022", "2023-12-01" must not be checked
# (otherwise a day-of-month like 31 is reported as an unsupported number).
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE_PATTERNS = [
    re.compile(rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}}\b)?", re.I),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b(?:,?\s+\d{{4}}\b)?", re.I),
    re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?"),
]


def _mask_dates(text):
    for pat in _DATE_PATTERNS:
        text = pat.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _parse(m, apply_skips):
    dollar, whole, frac, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
    suffix = suffix.strip().lower() if suffix else None
    bare = not dollar and not frac and not suffix and "," not in whole
    n = abs(int(whole.replace(",", "")))
    soft = bool(apply_skips and bare and (n <= 12 or 1900 <= n <= 2100))
    decimals = len(frac) - 1 if frac else 0
    mult = _MULT.get(suffix, 1.0) if suffix else 1.0
    value = float(whole.replace(",", "") + (frac or "")) * mult
    unit = (10 ** -decimals) * mult
    return {"text": m.group(0).strip(), "value": value, "unit": unit, "pct": suffix == "%", "soft": soft}


def extract_numbers(text, apply_skips=True):
    out = []
    text = text or ""
    if apply_skips:
        text = _mask_dates(text)
    for m in _NUM.finditer(text):
        p = _parse(m, apply_skips)
        if p:
            out.append(p)
    return out


def _walk(obj, out):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if obj == obj:  # drop NaN
            out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(p["value"] for p in extract_numbers(_mask_dates(obj), apply_skips=False))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, out)


def _column_stats(rows):
    cols = {}
    for r in rows or []:
        if isinstance(r, dict):
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cols.setdefault(k, []).append(float(v))
                elif isinstance(v, str):
                    try:
                        cols.setdefault(k, []).append(float(v))
                    except ValueError:
                        pass
    stats = []
    for vals in cols.values():
        if len(vals) > 1:
            stats += [sum(vals), sum(vals) / len(vals), min(vals), max(vals)]
    return stats


def _derived_pool(base, extra):
    pool = list(extra)
    for a in base:
        for b in base:
            if a is b:
                continue
            pool += [a + b, a - b]
            if b:
                pool += [a / b, a / b * 100, (a - b) / b * 100]
    return sorted(pool)


def _near(sorted_vals, target, tol):
    i = bisect.bisect_left(sorted_vals, target - tol)
    return i < len(sorted_vals) and sorted_vals[i] <= target + tol


def verify_answer(answer, investigation):
    ev = []
    _walk(investigation.question, ev)
    n_question = len(ev)
    _walk(investigation.sql_findings.get("rows"), ev)
    _walk(investigation.ml_findings, ev)
    _walk(getattr(investigation, "drilldown", None), ev)
    _walk(investigation.rag_findings.get("chunks"), ev)
    direct = sorted(set(ev))

    base = list(dict.fromkeys(ev[n_question:]))[:_MAX_BASE]
    extra = _column_stats(investigation.sql_findings.get("rows"))
    pool = None  # built lazily, only if something is left unmatched

    result = {"checked": 0, "supported": [], "derived": [], "unsupported": []}
    seen = set()
    for p in extract_numbers(answer):
        key = (p["text"], p["value"])
        if key in seen:
            continue
        seen.add(key)
        if p["soft"]:
            # Counts like "1", "5" and bare years are mostly ordinals, dates and "top N".
            # Confirm them when they match the evidence exactly; never flag them otherwise.
            if _near(direct, p["value"], 0.5):
                result["checked"] += 1
                result["supported"].append(p["text"])
            continue
        result["checked"] += 1
        cands = [p["value"]] + ([p["value"] / 100] if p["pct"] else [])
        tol = p["unit"] * 1.001 + 1e-9
        if any(_near(direct, c, tol if c == p["value"] else tol / 100) for c in cands):
            result["supported"].append(p["text"])
            continue
        if pool is None:
            pool = _derived_pool(base, extra)
        dtol = min(tol, 0.01 * abs(p["value"])) + 1e-9
        if any(_near(pool, c, dtol if c == p["value"] else dtol / 100) for c in cands):
            result["derived"].append(p["text"])
        else:
            result["unsupported"].append(p["text"])
    return result
