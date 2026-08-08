#!/usr/bin/env python3
"""A/B replay: two vanga binaries + two model sets, one CSV, identical inputs.

vanga resolves models as ./models/{SYMBOL} relative to its CWD, so each side is
"a binary + a models dir"; we sandbox each side in its own CWD with a symlink to
its models. Both sides predict from byte-identical slice files, so any
difference is the model/binary, not the data.

Scoring is mistake-based — lower is better everywhere:

  signal          the side price_levels and direction jointly point at. A
                  sideways call is an abstention, not a veto — sideways + up is
                  an up signal. Only a real contradiction (up vs down) kills it.

Two questions are scored separately, because mixing them is how you end up
blaming the model for something it never claimed:

ACCURACY — each target on its own axis, over every anchor:
  price err       distance from the predicted band centre to the realised
                  EXPONENTIALLY-WEIGHTED CLOSE, which is what price_levels
                  classifies (price_levels.rs:236)
  stop err        same against the realised WEIGHTED WORST ADVERSE measured
                  from the sequence's weighted close (stop_levels.rs:441),
                  on the side the direction head implied

TRADING — a hit test along the signal side, over signals only. An order fills
on a touch, so a long target is reached from below by the path HIGH and a short
from above by the path LOW; the closing price never enters it.
  target hit/full did price trade into the zone / reach the far edge
  price bias      + zone sat beyond where price got to (over-reaching)
                  - price ran past the far edge (under-reaching), 0 inside
  stop held/blown did price stay out of the zone / run clean through it
  stop bias       + stop wider than the real drawdown, - too tight, 0 inside
  R multiple      the trade, run bar by bar: whichever order fills first wins
  edge vs coin    R minus the same trade with the side picked blind. This is
                  the only number that says the model knows anything: for a
                  driftless price the blind baseline is ~0 at every RR.

  scripts/ab_eval.py --csv data/BTCUSDT.csv --horizon 8h,16h --window 2d --gap 4h \
      --a-bin /old/vanga --a-models /old/models \
      --b-bin ./target/release/vanga --b-models ./models
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

# (label, field, higher_is_better) — what the A/B verdict scores.
METRICS = (
    ("targ err%", "price_err", False),
    ("stop err%", "stop_err", False),
    ("R multiple", "r_multiple", True),
    ("edge vs coin", "edge_r", True),
)

# Serialized field order of the 5-class targets, each mapped to a trade side.
DIR_BINS = ["dump_probability", "down_probability", "sideways_probability",
            "up_probability", "pump_probability"]
PL_BINS = ["strong_down", "moderate_down", "neutral", "moderate_up", "strong_up"]
STOP_BINS = ["extreme_adverse", "high_adverse", "moderate_adverse",
             "low_adverse", "minimal_adverse"]
SIDES = [-1, -1, 0, 1, 1]


def duration_secs(text):
    unit = text[-1].lower()
    mult = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    if unit not in mult:
        raise ValueError(f"bad duration {text!r}, expected like 4h / 2d / 30m")
    return int(text[:-1]) * mult[unit]


def parse_ts(text):
    return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()


def fmt_ts(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def load_csv(path):
    with open(path, newline="") as fh:
        lines = fh.read().splitlines()
    cols = next(csv.reader([lines[0]]))
    idx = {name: cols.index(name) for name in ("timestamp", "open", "high", "low", "close")}
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = next(csv.reader([line]))
        o, h, l, c = (float(f[idx[n]]) for n in ("open", "high", "low", "close"))
        rows.append({
            "ts": parse_ts(f[idx["timestamp"]]),
            "high": h,
            "low": l,
            "close": c,
            "ohlc4": (o + h + l + c) / 4.0,  # direction targets are built on OHLC4
            "raw": line,
        })
    return lines[0], rows


def bar_seconds(rows):
    diffs = sorted(rows[i]["ts"] - rows[i - 1]["ts"]
                   for i in range(max(1, len(rows) - 20), len(rows)))
    return diffs[len(diffs) // 2]


def sandbox(workdir, side, models_dir, symbol):
    """Give a side its own CWD whose ./models points at that side's models."""
    models_dir = os.path.abspath(models_dir)
    if not os.path.exists(os.path.join(models_dir, f"{symbol}.meta")):
        sys.exit(f"side {side}: no {symbol}.meta in {models_dir}")
    root = os.path.abspath(os.path.join(workdir, f"root{side}"))
    os.makedirs(root, exist_ok=True)
    link = os.path.join(root, "models")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(models_dir, link)
    return root


def run_side(binary, root, symbol, slice_path, horizons, out_path, fresh, min_conf):
    if not os.path.exists(out_path) or fresh:
        cmd = [binary, "predict", "-s", symbol, "-i", slice_path, "--output", out_path]
        # One run covers every trained horizon; no reason to pay model load twice.
        cmd += ["--all-horizons"] if len(horizons) > 1 else ["--horizon", horizons[0]]
        if min_conf is not None:
            cmd += ["--min-confidence", str(min_conf)]
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        log_path = out_path.replace(".json", ".log")
        if proc.returncode != 0:
            with open(log_path, "w") as fh:
                fh.write(f"$ {' '.join(cmd)}\n(cwd {root})\n\n{proc.stdout}\n{proc.stderr}")
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            return {"__error__": (" | ".join(tail) or f"exit {proc.returncode}"),
                    "__log__": log_path}
        # Exit 0 with nothing usable is the model declining to signal — the bias
        # veto at formatter.rs:781 or the min_confidence skip. That is a
        # deliberate protection, so it is counted as behaviour, not an error.
        if not read_predictions(out_path, horizons):
            with open(log_path, "w") as fh:
                fh.write(f"$ {' '.join(cmd)}\n(cwd {root})\n\n{proc.stdout}\n{proc.stderr}")
    return read_predictions(out_path, horizons)


def argmax_bin(bins, names):
    """(index, probability) of the most likely class, over a bins dict."""
    probs = [bins.get(name, {}).get("probability", 0.0) for name in names]
    best = max(range(len(probs)), key=probs.__getitem__)
    return best, probs[best]


def read_predictions(path, horizons):
    with open(path) as fh:
        payload = json.load(fh)
    items = payload if isinstance(payload, list) else [payload]
    out = {}
    for item in items:
        horizon = item.get("horizon")
        if horizon not in horizons:
            continue
        pred = {"confidence": float(item.get("confidence", 0.0))}

        direction = item.get("direction")
        if direction:
            probs = [direction.get(name, 0.0) for name in DIR_BINS]
            cls = max(range(len(probs)), key=probs.__getitem__)
            pred.update(dir_class=cls, dir_side=SIDES[cls], dir_prob=probs[cls],
                        seq_len=int(direction.get("sequence_length") or 0))

        if direction:
            pred["dir_conf"] = float(direction.get("confidence", 0.0))
        if item.get("price_levels"):
            pred["pl_conf"] = float(item["price_levels"].get("confidence", 0.0))

        levels = (item.get("price_levels") or {}).get("bins")
        if levels:
            cls, prob = argmax_bin(levels, PL_BINS)
            band = levels[PL_BINS[cls]]["price"]
            pred.update(pl_class=cls, pl_side=SIDES[cls], pl_prob=prob,
                        pl_lo=min(band), pl_hi=max(band))

        stops = (item.get("stop_levels") or {}).get("bins")
        if stops:
            cls, prob = argmax_bin(stops, STOP_BINS)
            band = stops[STOP_BINS[cls]]["price"]
            pred.update(stop_class=cls, stop_prob=prob,
                        stop_lo=min(band), stop_hi=max(band),
                        # Whole distribution kept so the stop can be chosen by
                        # tail mass instead of by mode — see stop_band_at_tail.
                        stop_bins=[(stops[n].get("probability", 0.0),
                                    min(stops[n]["price"]), max(stops[n]["price"]))
                                   for n in STOP_BINS if n in stops])

        out[horizon] = pred
    return out


def realised_acceleration(rows, k, end_idx, seq_len):
    """Reproduce the direction target's axis: horizon momentum minus sequence
    momentum on OHLC4 (src/targets/direction.rs:432). Horizon momentum is
    measured from the FIRST horizon bar, k+1 — not from the anchor. The
    volatility normalisation there scales the value but not its sign, so no
    calibration is needed to know which side the label fell on."""
    start = k - seq_len + 1
    if seq_len < 2 or start < 0 or end_idx <= k + 1:
        return None
    seq_start, seq_end = rows[start]["ohlc4"], rows[k]["ohlc4"]
    hor_start, hor_end = rows[k + 1]["ohlc4"], rows[end_idx]["ohlc4"]
    if seq_start <= 0 or hor_start <= 0:
        return None
    return (hor_end - hor_start) / hor_start - (seq_end - seq_start) / seq_start


def stop_band_at_tail(bins, tail):
    """Pick the stop band by CUMULATIVE tail mass instead of by mode.

    stop_levels bins run extreme_adverse(0) -> minimal_adverse(4), worst to
    mildest, so accumulating from index 0 gives P(adverse at least this bad).
    Return the first band where that mass reaches `tail`: beyond it lies at most
    `tail` of the model's own predicted probability.

    The argmax picks the MODE, and a stop at the modal depth is breached by the
    whole cumulative mass of the worse classes — 40-60% with five balanced
    classes. Placing a stop is a tail question, not a mode question.
    """
    cum = 0.0
    for prob, lo, hi in bins:
        cum += prob
        if cum >= tail:
            return lo, hi
    return bins[-1][1], bins[-1][2]


def exp_weighted_close(bars):
    """get_sequence_exponential_weighted_close (price_levels.rs:236-256):
    weight ((i+1)/n)^2 over closes, so recent bars dominate. This is what
    price_levels actually classifies — an average, never an extreme."""
    n = len(bars)
    num = den = 0.0
    for i, b in enumerate(bars):
        w = ((i + 1) / n) ** 2
        num += b["close"] * w
        den += w
    return num / den


def weighted_worst_adverse(bars, bullish, reference):
    """find_weighted_worst_adverse (stop_levels.rs:441-476): the low (bullish)
    or high (bearish) that maximises severity x weight, weight = 0.5 + 0.5*r^2.
    Severity is measured from the sequence's weighted close, not from entry."""
    worst, best_severity = reference, 0.0
    n = len(bars)
    for i, b in enumerate(bars):
        w = 0.5 + 0.5 * ((i + 1) / n) ** 2
        price = b["low"] if bullish else b["high"]
        severity = max(0.0, reference - price if bullish else price - reference)
        if severity * w > best_severity:
            best_severity, worst = severity * w, price
    return worst


def walk(path, entry, side, reward, risk):
    """Barrier race over the horizon: which order fills first. reward/risk are
    distances in % of entry. Returns (outcome, exit % in trade direction).

    Stop is tested before target, so a bar containing both resolves as a loss —
    intrabar sequence is unknowable at this resolution and the pessimistic read
    is the one that will not flatter the model."""
    tgt = entry * (1.0 + side * reward / 100.0)
    stp = entry * (1.0 - side * risk / 100.0)
    for b in path:
        if b["low"] <= stp if side > 0 else b["high"] >= stp:
            return "loss", -risk
        if b["high"] >= tgt if side > 0 else b["low"] <= tgt:
            return "win", reward
    return "open", (path[-1]["close"] - entry) * side / entry * 100.0


def zone_error(reached, near, far):
    """How far `reached` fell short of, or ran past, the predicted [near, far]
    zone. Everything is already in trade-direction coordinates (bigger = further
    along the axis we care about), so one formula covers long and short, target
    and stop.

      > 0  OVERESTIMATE — the zone sat beyond where price got to
      < 0  UNDERESTIMATE — price ran past the far edge
      = 0  price finished inside the zone
    """
    if reached < near:
        return near - reached
    if reached > far:
        return far - reached
    return 0.0


def measure(pred, entry, path, accel, ref_price):
    """Score one prediction against the bars that followed it, on two separate
    questions that must not be mixed:

      ACCURACY  — was the model right about what it actually predicts? Each
                  target is scored on its own axis: price_levels against the
                  realised exponentially-weighted close, stop_levels against
                  the realised weighted worst adverse. Anything else penalises
                  the model for a claim it never made.
      TRADING   — would the trade have paid? A hit test along the signal side,
                  because an order fills on a touch, not on a close.

    `path` is every bar after the anchor to the horizon end, in order — order
    matters for the barrier race. `ref_price` is the sequence's weighted close,
    which is what stop_levels measures adverse excursion from."""
    end, low = path[-1]["close"], min(r["low"] for r in path)
    high = max(r["high"] for r in path)
    m = {"move_pct": (end - entry) / entry * 100.0}

    # --- ACCURACY: each target on its own axis --------------------------------
    if "pl_lo" in pred:
        realised = exp_weighted_close(path)
        lo, hi = pred["pl_lo"], pred["pl_hi"]
        m["price_in_band"] = lo <= realised <= hi
        m["price_err"] = abs(realised - (lo + hi) / 2.0) / entry * 100.0

    if "stop_lo" in pred:
        # The band was built for the side the DIRECTION head implied
        # (formatter.rs:484-490, bullish when absent), so it must be scored on
        # that side — not on the side our signal rule ended up taking.
        bullish = pred.get("dir_side", 0) >= 0
        realised = weighted_worst_adverse(path, bullish, ref_price)
        lo, hi = pred["stop_lo"], pred["stop_hi"]
        m["stop_in_band"] = lo <= realised <= hi
        m["stop_err"] = abs(realised - (lo + hi) / 2.0) / entry * 100.0

    if accel is not None and "dir_side" in pred:
        m["accel_pct"] = accel * 100.0
        # A sideways call makes no directional claim, so it is scored as a
        # no-call rather than a miss.
        if pred["dir_side"]:
            m["dir_called"] = True
            m["dir_wrong"] = pred["dir_side"] * accel <= 0
            # dir_wrong scores the target the model was trained on (momentum
            # acceleration); dir_paid scores whether that call made money,
            # which is a different question and the one that pays rent.
            m["dir_paid"] = pred["dir_side"] * m["move_pct"] > 0
        else:
            m["dir_called"] = False

    # A sideways call abstains rather than objects, so sideways + up is an up
    # signal. Only price_levels and direction pointing opposite ways kills it.
    sides = {pred.get("pl_side", 0), pred.get("dir_side", 0)} - {0}
    side = sides.pop() if len(sides) == 1 else 0
    m["signal"] = side
    if not side:
        # No side means no trade, so no level can be right or wrong.
        return m

    m["edge_pct"] = m["move_pct"] * side
    m["win"] = m["edge_pct"] > 0

    def favour(price):
        """% of entry that `price` sits in the direction the trade profits."""
        return (price - entry) * side / entry * 100.0

    # --- TRADING: hit test along the side we would have taken ------------------
    if "pl_lo" in pred:
        # Longs reach their target from below via the path high, shorts from
        # above via the path low. sorted() puts the near edge first for both.
        near, far = sorted((favour(pred["pl_lo"]), favour(pred["pl_hi"])))
        reach = favour(high if side > 0 else low)
        m["target_reach_pct"] = reach
        m["target_hit"] = reach >= near
        m["target_full"] = reach >= far
        m["price_bias"] = zone_error(reach, near, far)

    if "stop_lo" in pred:
        # Same test mirrored: adverse coordinates, where bigger = worse for us.
        near, far = sorted((-favour(pred["stop_lo"]), -favour(pred["stop_hi"])))
        adverse = -favour(low if side > 0 else high)
        m["adverse_pct"] = adverse
        m["stop_hit"] = adverse >= near
        m["stop_blown"] = adverse > far
        m["stop_bias"] = zone_error(adverse, near, far)

    if "pl_lo" in pred and "stop_lo" in pred:
        # The combined score: run the trade. Enter at the anchor close, take the
        # near edge of each zone as the working order, and walk the bars in
        # order — whichever fills first ends it.
        reward = favour(pred["pl_lo"] if side > 0 else pred["pl_hi"])
        risk = -favour(pred["stop_hi"] if side > 0 else pred["stop_lo"])

        outcome, exit_pct = walk(path, entry, side, reward, risk)
        m["outcome"] = outcome
        m["exit_pct"] = exit_pct
        # risk <= 0 means the stop sat on the profitable side of entry — the
        # levels contradict the signal, so there is no trade to size.
        m["bad_geometry"] = risk <= 0
        if risk > 0:
            m["r_multiple"] = exit_pct / risk
            m["planned_rr"] = reward / risk
            # No-edge baseline: the SAME bars and the SAME geometry, with the
            # side chosen blind. Averaging both sides is exactly what you earn
            # with no directional information, so r_multiple - baseline_r is the
            # part of the result that came from picking a side rather than from
            # the reward/risk you happened to set. For a driftless price this
            # baseline is ~0 at every RR, which is why a better RR alone cannot
            # make a losing system profitable.
            _, flip_pct = walk(path, entry, -side, reward, risk)
            m["baseline_r"] = (exit_pct + flip_pct) / 2.0 / risk
            m["edge_r"] = m["r_multiple"] - m["baseline_r"]
    return m


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def quantile(sorted_values, q):
    """Plain nearest-rank quantile — no interpolation, so the number printed is
    a confidence the model actually produced."""
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def confidence_report(measured, horizon, labels, gate, field, title):
    """Confidence is not comparable across binaries in absolute terms — the
    scale itself changed — so report the spread and translate the live gate
    into whatever value buys the same selectivity on the other side."""
    spreads = {}
    for side in ("A", "B"):
        values = sorted(r[field] for r in measured[(side, horizon)] if field in r)
        if values:
            spreads[side] = values
    if not spreads:
        return

    print(f"\n  {title}")
    for side, values in spreads.items():
        qs = [quantile(values, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
        passed = sum(1 for v in values if v >= gate) / len(values)
        print(f"    {side}  n={len(values):<4} min {qs[0]:.3f}  p25 {qs[1]:.3f}"
              f"  med {qs[2]:.3f}  p75 {qs[3]:.3f}  max {qs[4]:.3f}"
              f"   >={gate:.2f}: {passed:>6.1%}")

    if len(spreads) != 2:
        return
    a_pass = sum(1 for v in spreads["A"] if v >= gate) / len(spreads["A"])
    desc = sorted(spreads["B"], reverse=True)
    equiv = desc[min(len(desc) - 1, max(0, round(a_pass * len(desc)) - 1))]
    print(f"    A's {gate:.2f} gate passes {a_pass:.1%} — B needs {equiv:.3f}"
          f" for the same selectivity")

    # Both sides ran the same anchors, so the pairs give the mapping directly.
    # It is a band, not a point: confidence is many-to-one over the class vector.
    pairs = sorted(b[field] for a, b in zip(measured[("A", horizon)], measured[("B", horizon)])
                   if field in a and field in b and a[field] >= gate)
    if pairs:
        print(f"    on the {len(pairs)} anchors A gates in, B's {field} spans"
              f" {pairs[0]:.3f}–{pairs[-1]:.3f} (med {quantile(pairs, 0.5):.3f})")


def prob_better(wins, trials):
    """P(the winning rate is truly above 50%) with a uniform prior, i.e. the
    posterior mass of Beta(wins+1, trials-wins+1) above 0.5. Integer parameters
    make it an exact binomial sum, so no scipy: sum_{j<=wins} C(n,j) / 2^n."""
    n = trials + 1
    return sum(math.comb(n, j) for j in range(0, wins + 1)) / (2 ** n)


def paired(measured, horizons, field):
    """(A, B) value pairs over anchors where BOTH sides produced the metric.
    Anchors are already paired by index — a row is only appended when both
    sides answered — so zip() lines up the same bar on both sides."""
    return [(a[field], b[field])
            for h in horizons
            for a, b in zip(measured[("A", h)], measured[("B", h)])
            if field in a and field in b]


def verdict(measured, horizons):
    """Which side makes smaller mistakes, and how sure can we be at this n.

    The mean says who is ahead; the sign test says whether that lead survives
    the sample size. Ties are dropped — they carry no evidence either way.
    R multiple is the verdict metric: it is the only one denominated in money."""
    print(f"\n{'=' * 72}\nVERDICT — paired anchors only (same bars, both sides answered)")
    print(f"  {'metric':>11} {'pairs':>6} {'A mean':>8} {'B mean':>8}"
          f" {'B wins':>8} {'P(B better)':>12}")

    summary = None
    for name, field, higher_better in METRICS:
        pairs = paired(measured, horizons, field)
        if not pairs:
            print(f"  {name:>11} {'0':>6}   no anchor scored this metric on both sides")
            continue
        a_mean, b_mean = mean(a for a, _ in pairs), mean(b for _, b in pairs)
        wins = sum(1 for a, b in pairs if (b > a if higher_better else b < a))
        decisive = sum(1 for a, b in pairs if a != b)
        prob = prob_better(wins, decisive)
        print(f"  {name:>11} {len(pairs):>6} {a_mean:>8.3f} {b_mean:>8.3f}"
              f" {f'{wins}/{decisive}':>8} {prob:>11.1%}")
        if field == "r_multiple":
            summary = (a_mean, b_mean, decisive, prob, higher_better)

    # The table above only says which side is less bad. This says whether either
    # has any edge at all, which is the question that decides whether retraining
    # is worth doing: same bars, same levels, only the side selection differs.
    print(f"\n  EDGE vs coin flip — is the side selection worth anything?")
    print(f"  {'side':>5} {'trades':>7} {'R':>8} {'baseline':>9} {'edge':>8}"
          f" {'beat':>9} {'P(real)':>9}")
    # A trade where both sides would have won is a tie: the side choice was
    # worth nothing, so it is evidence for neither. Only decisive trades count,
    # and MIN_DECISIVE stops a 2-sample fluke from reading as a real edge.
    MIN_DECISIVE = 15
    verdicts = {}
    for s in ("A", "B"):
        rows_s = [r for h in horizons for r in measured[(s, h)] if "edge_r" in r]
        if not rows_s:
            print(f"  {s:>5} {0:>7}   no complete trades")
            continue
        decisive = sum(1 for r in rows_s if r["edge_r"] != 0)
        beat = sum(1 for r in rows_s if r["edge_r"] > 0)
        conf = prob_better(beat, decisive)
        verdicts[s] = (mean(r["edge_r"] for r in rows_s), conf, decisive)
        print(f"  {s:>5} {len(rows_s):>7} {mean(r['r_multiple'] for r in rows_s):>8.3f}"
              f" {mean(r['baseline_r'] for r in rows_s):>9.3f}"
              f" {verdicts[s][0]:>8.3f} {f'{beat}/{decisive}':>9} {conf:>8.1%}")

    thin = [s for s, (_, _, d) in verdicts.items() if d < MIN_DECISIVE]
    if thin:
        print(f"\n  Side(s) {'+'.join(thin)} have under {MIN_DECISIVE} decisive trades —"
              " their P(real) is not usable.\n  Ties are whipsaws where both sides won;"
              " tight targets (RR < 1) produce a lot of them.")
    proven = [s for s, (e, c, d) in verdicts.items() if e > 0 and c >= 0.8 and d >= MIN_DECISIVE]
    if verdicts and not proven:
        print("\n  ⚠️  NEITHER side is shown to beat a blind coin flip on the same bars"
              "\n      and the same levels. The head-to-head below is then a comparison"
              "\n      between two things not yet distinguishable from no signal — a"
              "\n      better RR cannot rescue that, only a real directional edge can.")

    if not summary:
        print("\n  No anchor produced a complete trade on both sides — nothing to"
              " score in money terms.")
        return
    a_mean, b_mean, decisive, prob, higher_better = summary
    lead = "B" if (b_mean > a_mean if higher_better else b_mean < a_mean) else "A"
    gap = abs(b_mean - a_mean)
    conf = prob if lead == "B" else 1.0 - prob

    print(f"\n  {lead} earns {gap:.3f}R more per signal, at {conf:.0%} confidence.")
    if decisive < 20:
        print(f"  NOT DECISIVE — {decisive} decisive pairs. Widen --window or shrink"
              " --gap; aim for 50+.")
    elif conf >= 0.95:
        print(f"  {lead} is better. The lead survives the sample size.")
    elif conf >= 0.80:
        print(f"  {lead} is probably better, but this would not survive a rerun on"
              " fresh bars. Collect more anchors before acting on it.")
    else:
        print("  TOO CLOSE TO CALL — the two sides are indistinguishable on this data.")
    print("  Overlapping windows are correlated, so the true evidence is weaker"
          " than the pair count suggests.")


def cell(value, width=7, digits=2, suffix=""):
    if value is None:
        return f"{'-':>{width + len(suffix)}}"
    return f"{value:>{width}.{digits}f}{suffix}"


def bias_word(bias, over, under):
    """Turn a signed zone error into the thing that went wrong, in trading terms."""
    if bias is None:
        return ""
    return over if bias > 0 else (under if bias < 0 else "on the level")


def report(rows, label, vetoed, attempted):
    """rows: every measured anchor for one side/horizon."""
    signals = [r for r in rows if r["signal"]]
    priced = [r for r in signals if "target_hit" in r]
    stopped = [r for r in signals if "stop_hit" in r]
    traded = [r for r in signals if "r_multiple" in r]

    print(f"\n{label}")
    if attempted:
        print(f"  declined      {vetoed}/{attempted} anchors ({vetoed / attempted:.1%})"
              " — vanga's own consistency protection")

    # Accuracy first: every anchor, each target on its own axis.
    acc_p = [r for r in rows if "price_err" in r]
    acc_s = [r for r in rows if "stop_err" in r]
    if acc_p:
        hit = sum(1 for r in acc_p if r["price_in_band"]) / len(acc_p)
        print(f"  price_levels  n={len(acc_p):<4} in-band {hit:>6.1%}"
              f"  err {cell(mean(r['price_err'] for r in acc_p))}%"
              "   (vs realised weighted close)")
    if acc_s:
        hit = sum(1 for r in acc_s if r["stop_in_band"]) / len(acc_s)
        print(f"  stop_levels   n={len(acc_s):<4} in-band {hit:>6.1%}"
              f"  err {cell(mean(r['stop_err'] for r in acc_s))}%"
              "   (vs realised weighted worst adverse)")

    scored = [r for r in rows if "dir_called" in r]
    called = [r for r in scored if r["dir_called"]]
    if scored:
        wrong = (sum(1 for r in called if r["dir_wrong"]) / len(called)) if called else None
        paid = (sum(1 for r in called if r["dir_paid"]) / len(called)) if called else None
        print(f"  direction     n={len(scored):<4} called  {len(called) / len(scored):>6.1%}"
              f"  wrong {cell(wrong and wrong * 100, 5, 1, '%')}"
              f"  paid {cell(paid and paid * 100, 5, 1, '%')}")

    fire = len(signals) / len(rows) if rows else 0.0
    if not signals:
        print(f"  signals       n=0    fired   {fire:>6.1%}  (price_levels and"
              " direction contradicted each other on every anchor)")
        return
    wins = sum(1 for r in signals if r["win"]) / len(signals)
    print(f"  signals       n={len(signals):<4} fired   {fire:>6.1%}"
          f"  close-win {wins:>6.1%}"
          f"  edge {cell(mean(r['edge_pct'] for r in signals))}%")

    if priced:
        hit = sum(1 for r in priced if r["target_hit"]) / len(priced)
        full = sum(1 for r in priced if r["target_full"]) / len(priced)
        bias = mean(r["price_bias"] for r in priced)
        print(f"  target        n={len(priced):<4} hit     {hit:>6.1%}  full {full:>6.1%}"
              f"  bias {cell(bias, 6)}% {bias_word(bias, 'over-reaching', 'under-reaching')}")

    if stopped:
        held = sum(1 for r in stopped if not r["stop_hit"]) / len(stopped)
        blown = sum(1 for r in stopped if r["stop_blown"]) / len(stopped)
        bias = mean(r["stop_bias"] for r in stopped)
        print(f"  stop          n={len(stopped):<4} held    {held:>6.1%}  blown {blown:>6.1%}"
              f"  bias {cell(bias, 6)}% {bias_word(bias, 'too wide', 'too tight')}")

    if traded:
        tally = {o: sum(1 for r in traded if r["outcome"] == o)
                 for o in ("win", "loss", "open")}
        bad = sum(1 for r in signals if r.get("bad_geometry"))
        print(f"  trade         n={len(traded):<4} win {tally['win']}  loss {tally['loss']}"
              f"  open {tally['open']}"
              f"  R {cell(mean(r['r_multiple'] for r in traded))}"
              f"  planned RR {cell(mean(r['planned_rr'] for r in traded))}"
              + (f"  [{bad} unusable: stop on the wrong side of entry]" if bad else ""))
        # Ties are trades where BOTH sides would have won — with RR < 1 the two
        # targets sit inside the two stops, so a whipsaw fills both. The side
        # choice was genuinely worth nothing there, so they carry no evidence.
        decisive = sum(1 for r in traded if r["edge_r"] != 0)
        beat = sum(1 for r in traded if r["edge_r"] > 0)
        print(f"  vs coin flip  same bars, same levels, side picked blind:"
              f"  baseline R {cell(mean(r['baseline_r'] for r in traded))}"
              f"  edge {cell(mean(r['edge_r'] for r in traded))}"
              f"  beat {beat}/{decisive} decisive of {len(traded)}"
              f" ({prob_better(beat, decisive):.0%} sure real)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", help="defaults to the CSV file stem")
    ap.add_argument("--horizon", required=True,
                    help="trained horizon(s), comma separated: 8h,16h")
    ap.add_argument("--window", default="2d", help="evaluate over the last N of the file")
    ap.add_argument("--gap", default="4h", help="step between anchors")
    ap.add_argument("--a-bin", required=True)
    ap.add_argument("--a-models", required=True, help="models dir for side A (holds SYMBOL.meta)")
    ap.add_argument("--b-bin", required=True)
    ap.add_argument("--b-models", required=True, help="models dir for side B (holds SYMBOL.meta)")
    ap.add_argument("--gate", type=float, default=0.42,
                    help="live gate to translate between the two scales; reporting only,"
                         " it never filters the scoring")
    ap.add_argument("--stop-tail", type=float, default=None, metavar="P",
                    help="place the stop where the model gives at most P probability to a"
                         " worse adverse move (e.g. 0.15), instead of at the most likely"
                         " bin. Targets are NOT symmetric to this: the bins describe where"
                         " the weighted close LANDS, not the extreme price REACHES, so the"
                         " same query does not transfer to the target side")
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="passed through to the binaries only when given; omit to use"
                         " each binary's own default")
    ap.add_argument("--workdir", default="/tmp/vanga_ab")
    ap.add_argument("--out", help="write per-anchor results to this CSV")
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    args = ap.parse_args()

    symbol = args.symbol or os.path.splitext(os.path.basename(args.csv))[0]
    horizons = [h.strip() for h in args.horizon.split(",") if h.strip()]
    header, rows = load_csv(args.csv)
    bar = bar_seconds(rows)
    hbars = {h: round(duration_secs(h) / bar) for h in horizons}
    gap_bars = max(1, round(duration_secs(args.gap) / bar))
    window_bars = round(duration_secs(args.window) / bar)

    # The tail of the file has no realised outcome yet — the shortest horizon
    # reaches furthest, longer ones drop their own tail below.
    last = len(rows) - 1 - min(hbars.values())
    first = max(0, len(rows) - 1 - window_bars)
    if last < first:
        sys.exit(f"window {args.window} is shorter than the shortest horizon — nothing to score")
    anchors = sorted(range(last, first - 1, -gap_bars))

    print(f"{symbol}: {len(rows)} bars @ {bar / 3600:g}h, window {args.window}, "
          f"gap {args.gap}, horizons {'/'.join(horizons)}")
    print(f"anchors: {len(anchors)} from {fmt_ts(rows[anchors[0]]['ts'])} "
          f"to {fmt_ts(rows[anchors[-1]]['ts'])}")
    for h in horizons:
        usable = sum(1 for k in anchors if k + hbars[h] < len(rows))
        print(f"  {h}: {usable}/{len(anchors)} anchors have a realised outcome")

    os.makedirs(args.workdir, exist_ok=True)
    roots = {"A": sandbox(args.workdir, "A", args.a_models, symbol),
             "B": sandbox(args.workdir, "B", args.b_models, symbol)}
    measured = {(side, h): [] for side in ("A", "B") for h in horizons}
    per_anchor = []
    # Both sides must answer the same anchor or the comparison is scored on
    # different samples, which silently invalidates it.
    dropped = {h: 0 for h in horizons}
    failures = {"A": [], "B": []}
    # Declining to signal is a model decision (bias veto / confidence skip), so
    # it is tracked per side as a rate rather than hidden behind the pairing.
    vetoed = {(side, h): 0 for side in ("A", "B") for h in horizons}
    attempted = {h: 0 for h in horizons}

    for n, k in enumerate(anchors, 1):
        slice_path = os.path.abspath(os.path.join(args.workdir, f"{symbol}_{k}.csv"))
        if not os.path.exists(slice_path) or args.fresh:
            with open(slice_path, "w") as fh:
                fh.write(header + "\n")
                fh.write("\n".join(r["raw"] for r in rows[: k + 1]) + "\n")

        stamp = fmt_ts(rows[k]["ts"])
        entry = rows[k]["close"]
        print(f"[{n}/{len(anchors)}] {stamp}", end="", flush=True)

        preds = {}
        for side, binary in (("A", args.a_bin), ("B", args.b_bin)):
            out_path = os.path.abspath(os.path.join(args.workdir, f"{symbol}_{k}_{side}.json"))
            preds[side] = run_side(binary, roots[side], symbol, slice_path, horizons,
                                   out_path, args.fresh, args.min_confidence)
            if "__error__" in preds[side]:
                failures[side].append((stamp, preds[side]["__error__"],
                                       preds[side]["__log__"]))
                print(f"\n  ! {side} failed: {preds[side]['__error__']}", flush=True)

        for h in horizons:
            end_idx = k + hbars[h]
            if end_idx >= len(rows):
                continue
            path = rows[k + 1: end_idx + 1]
            end = rows[end_idx]["close"]
            print(f" {h}={(end - entry) / entry * 100:+.2f}%", end="")

            attempted[h] += 1
            for side in ("A", "B"):
                if not preds[side].get(h) and "__error__" not in preds[side]:
                    vetoed[(side, h)] += 1

            if not (preds["A"].get(h) and preds["B"].get(h)):
                missing = [s for s in ("A", "B") if not preds[s].get(h)]
                print(f" [{h}: no signal from {'+'.join(missing)}]", end="")
                dropped[h] += 1
                continue

            row = {"timestamp": stamp, "horizon": h}
            for side in ("A", "B"):
                pred = preds[side][h]
                if args.stop_tail is not None and pred.get("stop_bins"):
                    pred["stop_lo"], pred["stop_hi"] = \
                        stop_band_at_tail(pred["stop_bins"], args.stop_tail)
                seq_len = pred.get("seq_len", 0)
                accel = realised_acceleration(rows, k, end_idx, seq_len)
                # stop_levels measures adverse excursion from the sequence's
                # weighted close, not from entry (stop_levels.rs:405).
                seq = rows[max(0, k - seq_len + 1): k + 1]
                ref_price = exp_weighted_close(seq) if seq else entry
                m = measure(pred, entry, path, accel, ref_price)
                for key in ("confidence", "dir_conf", "pl_conf"):
                    if key in pred:
                        m[key] = pred[key]
                measured[(side, h)].append(m)
                row[side] = m
            per_anchor.append(row)
        print()

    for side in ("A", "B"):
        if failures[side]:
            stamp, err, log_path = failures[side][0]
            print(f"\n! side {side}: {len(failures[side])}/{len(anchors)} prediction runs"
                  f" failed — first at {stamp}: {err}\n  full output: {log_path}")
    if any(dropped.values()):
        detail = ", ".join(f"{h}: {n}/{attempted[h]}" for h, n in dropped.items() if n)
        print(f"\nanchors not scored because a side declined to signal ({detail}) —"
              " mistakes are compared on the anchors both sides answered")

    labels = {"A": f"A  {args.a_bin}", "B": f"B  {args.b_bin}"}
    for h in horizons:
        print(f"\n{'=' * 72}\nHORIZON {h}")
        for side in ("A", "B"):
            report(measured[(side, h)], labels[side], vetoed[(side, h)], attempted[h])
        confidence_report(measured, h, labels, args.gate, "confidence",
                          "overall confidence")
        confidence_report(measured, h, labels, args.gate, "dir_conf",
                          "direction confidence")

    # Scored on the signal set only: those are the anchors we would have traded,
    # so they are the only ones where a level being off costs money.
    print(f"\n{'=' * 72}\nSUMMARY — signals only, scored as trades")
    print(f"  {'horizon':>8} {'side':>5} {'sig':>4} {'declined':>9} | {'hit':>6}"
          f" {'targ err%':>9} {'targ bias':>9} | {'held':>6} {'stop err%':>9}"
          f" {'stop bias':>9} | {'R':>6} {'edge':>6} {'win/loss':>9}")
    for h in horizons:
        for side in ("A", "B"):
            rows_h = measured[(side, h)]
            decline = (vetoed[(side, h)] / attempted[h]) if attempted[h] else None
            signals = [r for r in rows_h if r["signal"]]
            if not signals:
                print(f"  {h:>8} {side:>5} {len(signals):>4}"
                      f" {cell(decline and decline * 100, 8, 1, '%')} | no signals")
                continue
            priced = [r for r in signals if "target_hit" in r]
            stopped = [r for r in signals if "stop_hit" in r]
            traded = [r for r in signals if "r_multiple" in r]
            acc_p = [r for r in rows_h if "price_err" in r]
            acc_s = [r for r in rows_h if "stop_err" in r]
            hit = (sum(1 for r in priced if r["target_hit"]) / len(priced)) if priced else None
            held = (sum(1 for r in stopped if not r["stop_hit"]) / len(stopped)
                    if stopped else None)
            wl = (f"{sum(1 for r in traded if r['outcome'] == 'win')}"
                  f"/{sum(1 for r in traded if r['outcome'] == 'loss')}") if traded else "-"
            print(f"  {h:>8} {side:>5} {len(signals):>4}"
                  f" {cell(decline and decline * 100, 8, 1, '%')} |"
                  f" {cell(hit and hit * 100, 5, 1, '%')}"
                  f" {cell(mean(r['price_err'] for r in acc_p), 9)}"
                  f" {cell(mean(r['price_bias'] for r in priced), 9)} |"
                  f" {cell(held and held * 100, 5, 1, '%')}"
                  f" {cell(mean(r['stop_err'] for r in acc_s), 9)}"
                  f" {cell(mean(r['stop_bias'] for r in stopped), 9)} |"
                  f" {cell(mean(r['r_multiple'] for r in traded), 6)}"
                  f" {cell(mean(r['edge_r'] for r in traded), 6)}"
                  f" {wl:>9}")

    verdict(measured, horizons)

    if args.out:
        cols = ["signal", "outcome", "r_multiple", "baseline_r", "edge_r",
                "planned_rr", "exit_pct",
                "price_err", "price_in_band", "stop_err", "stop_in_band",
                "target_hit", "target_full", "target_reach_pct", "price_bias",
                "stop_hit", "stop_blown", "adverse_pct", "stop_bias",
                "dir_called", "dir_wrong", "dir_paid", "accel_pct",
                "edge_pct", "bad_geometry",
                "confidence", "dir_conf", "pl_conf"]
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", "horizon", "move_pct"]
                       + [f"{s.lower()}_{c}" for s in ("A", "B") for c in cols])
            for r in per_anchor:
                a, b = r.get("A", {}), r.get("B", {})
                move = a.get("move_pct", b.get("move_pct", ""))
                w.writerow([r["timestamp"], r["horizon"], move]
                           + [a.get(c, "") for c in cols]
                           + [b.get(c, "") for c in cols])
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
