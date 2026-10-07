"""
Attribute-update model: training + walk-forward backtest.

Per stat group (hitting / pitching) we fit three signals for each attribute's
delta at the next monthly attribute update:

  S1  gap      — per-attribute linear response to the gap between the card's
                 rating and the rating its stats "deserve" (a Ridge projector)
  S2  gbm      — LightGBM regressor on ratings, stat windows and history
  S3  analog   — k-nearest historical (card, attribute) situations

plus a 3-class LightGBM classifier for P(up) / P(down).

Everything downstream of the base models is learned from *out-of-fold*
predictions produced walk-forward (train on updates < k, predict update k):
the stacking weights, the prediction intervals, and the map from attribute
deltas to an OVR delta with its error distribution. The backtest for update
k only ever uses parameters fitted on updates < k, so the reported metrics
are what you'd actually have seen live.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CORE_ATTRS, MODELS_DIR, PROCESSED_DIR, STAT_CUTOFF_DAYS, quicksell_value
from src.features.dataset import feature_columns
from src.features.windows import HIT_RATES, PIT_RATES

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "update_model.joblib"
SUMMARY_PATH = MODELS_DIR / "backtest_summary.json"
SIGNALS = ["s_gap", "s_gbm", "s_analog"]
INTERVAL_Q = (0.1, 0.9)  # 80% prediction interval
OVR_MOVES = list(range(-4, 5))  # integer OVR moves; tails clipped to ±4


def _projector_cols(group: str) -> list[str]:
    rates = HIT_RATES if group == "hitting" else PIT_RATES
    n = "pa" if group == "hitting" else "bf"
    return [f"{w}_{r}" for w in ("stab", "prev", "ytd", "30d") for r in rates] + [f"ytd_{n}", f"prev_{n}"]


def _analog_cols(group: str) -> list[str]:
    rates = HIT_RATES if group == "hitting" else PIT_RATES
    return (
        ["gap", "rating_before", "rating_minus_ovr", "last_delta", "prev_major_delta"]
        + [f"stab_{r}" for r in rates] + [f"30d_{r}" for r in rates]
    )


def _model_cols(group: str) -> list[str]:
    return feature_columns(group) + ["proj", "gap"]


# ═══════════════════════════════════════════════════════════════════════════
#  Base models for one group
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GroupModel:
    group: str
    projectors: dict = field(default_factory=dict)
    gap_slopes: dict = field(default_factory=dict)
    analogs: dict = field(default_factory=dict)
    gbm: object = None
    clf: object = None

    def fit(self, df: pd.DataFrame) -> "GroupModel":
        df = df.copy()
        y_after = df["rating_before"] + df["delta"]
        pcols = _projector_cols(self.group)
        for attr, sub in df.groupby("attr"):
            proj = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=3.0))
            proj.fit(sub[pcols], y_after.loc[sub.index])
            self.projectors[attr] = proj
        self._add_projection(df)

        for attr, sub in df.groupby("attr"):
            lr = LinearRegression().fit(sub[["gap"]].fillna(0), sub["delta"])
            self.gap_slopes[attr] = (float(lr.coef_[0]), float(lr.intercept_))
            knn = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                KNeighborsRegressor(n_neighbors=min(60, len(sub)), weights="distance"),
            )
            knn.fit(sub[_analog_cols(self.group)], sub["delta"])
            self.analogs[attr] = knn

        X = df[_model_cols(self.group)]
        common = dict(
            n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
        )
        self.gbm = LGBMRegressor(**common).fit(X, df["delta"], categorical_feature=["attr_code"])
        direction = np.sign(df["delta"]).astype(int) + 1  # 0=down 1=flat 2=up
        self.clf = LGBMClassifier(objective="multiclass", **common).fit(X, direction, categorical_feature=["attr_code"])
        return self

    def _add_projection(self, df: pd.DataFrame) -> None:
        pcols = _projector_cols(self.group)
        proj = pd.Series(np.nan, index=df.index)
        for attr, sub in df.groupby("attr"):
            if attr in self.projectors:
                proj.loc[sub.index] = self.projectors[attr].predict(sub[pcols])
        df["proj"] = proj
        df["gap"] = proj - df["rating_before"]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self._add_projection(df)
        out = pd.DataFrame(index=df.index)
        out["proj"], out["gap"] = df["proj"], df["gap"]
        out["s_gap"] = 0.0
        out["s_analog"] = 0.0
        for attr, sub in df.groupby("attr"):
            slope, icpt = self.gap_slopes.get(attr, (0.0, 0.0))
            out.loc[sub.index, "s_gap"] = sub["gap"].fillna(0) * slope + icpt
            if attr in self.analogs:
                out.loc[sub.index, "s_analog"] = self.analogs[attr].predict(sub[_analog_cols(self.group)])
        X = df[_model_cols(self.group)]
        out["s_gbm"] = self.gbm.predict(X)
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        out["p_down"] = proba[:, classes.index(0)] if 0 in classes else 0.0
        out["p_up"] = proba[:, classes.index(2)] if 2 in classes else 0.0
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Calibration layer (learned from out-of-fold predictions)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Calibration:
    """Stacking weights, intervals and OVR mapping for one group."""
    weights: dict = field(default_factory=lambda: {"s_gap": 0.0, "s_gbm": 1.0, "s_analog": 0.0, "intercept": 0.0})
    intervals: list = field(default_factory=list)   # [{"max_abs": x, "lo": q10, "hi": q90}]
    ovr_coef: dict = field(default_factory=dict)    # attr → weight of its delta in OVR
    ovr_intercept: float = 0.0
    ovr_clf: object = None                          # P(OVR move = k | mu, ovr)
    n_oof: int = 0

    def ensemble(self, sig: pd.DataFrame) -> pd.Series:
        w = self.weights
        return w["intercept"] + sum(sig[s] * w[s] for s in SIGNALS)

    def interval(self, pred: pd.Series) -> tuple[pd.Series, pd.Series]:
        if not self.intervals:
            return pred - 4.0, pred + 4.0
        lo, hi = pd.Series(np.nan, index=pred.index), pd.Series(np.nan, index=pred.index)
        a = pred.abs()
        prev = -1.0
        for b in self.intervals:
            m = (a > prev) & (a <= b["max_abs"])
            lo[m], hi[m] = pred[m] + b["lo"], pred[m] + b["hi"]
            prev = b["max_abs"]
        return lo.fillna(pred - 4.0), hi.fillna(pred + 4.0)

    def ovr_move_probs(self, mu: np.ndarray, ovr: np.ndarray) -> np.ndarray:
        """(n, len(OVR_MOVES)) probabilities of each integer OVR move."""
        if self.ovr_clf is None:
            return _normal_move_probs(mu, sd=1.0)
        raw = self.ovr_clf.predict_proba(_ovr_features(mu, ovr))
        probs = np.zeros((len(mu), len(OVR_MOVES)))
        for j, k in enumerate(self.ovr_clf.classes_):
            probs[:, OVR_MOVES.index(int(k))] = raw[:, j]
        return probs

    def ovr_mean(self, pivot: pd.DataFrame) -> pd.Series:
        mu = pd.Series(self.ovr_intercept, index=pivot.index)
        for attr, c in self.ovr_coef.items():
            if attr in pivot:
                mu += pivot[attr].fillna(0) * c
        return mu


def fit_calibration(oof: pd.DataFrame, group: str) -> Calibration:
    cal = Calibration(n_oof=len(oof))
    if len(oof) < 500:
        return cal

    stack = LinearRegression(positive=True).fit(oof[SIGNALS], oof["delta"])
    cal.weights = {s: float(c) for s, c in zip(SIGNALS, stack.coef_)} | {"intercept": float(stack.intercept_)}
    ens = cal.ensemble(oof)
    resid = oof["delta"] - ens

    edges = [0.25, 0.75, 1.5, 3.0, 99.0]
    prev = -1.0
    for e in edges:
        r = resid[(ens.abs() > prev) & (ens.abs() <= e)]
        if len(r) >= 50:
            cal.intervals.append({"max_abs": e, "lo": float(r.quantile(INTERVAL_Q[0])), "hi": float(r.quantile(INTERVAL_Q[1])), "n": int(len(r))})
        prev = e

    pivot, players = player_frame(pd.concat([oof, ens.rename("ens")], axis=1), group)
    if len(players) >= 100:
        attrs = [a for a in CORE_ATTRS[group] if a in pivot]
        ridge = Ridge(alpha=1.0, positive=True).fit(pivot[attrs].fillna(0), players["ovr_delta"])
        cal.ovr_coef = {a: float(c) for a, c in zip(attrs, ridge.coef_)}
        cal.ovr_intercept = float(ridge.intercept_)
        mu = cal.ovr_mean(pivot).to_numpy()
        moves = players["ovr_delta"].clip(OVR_MOVES[0], OVR_MOVES[-1]).astype(int)
        cal.ovr_clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(
            _ovr_features(mu, players["ovr_before"].to_numpy()), moves
        )
    return cal


def _ovr_features(mu: np.ndarray, ovr: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    ovr = np.asarray(ovr, dtype=float)
    return np.column_stack([mu, np.abs(mu), np.clip(mu, 0, None), ovr, ovr >= 99])


def player_frame(rows: pd.DataFrame, group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot attribute predictions to one row per (card, as_of) for OVR.

    A card's OVR is driven by its display group: pitchers by pitching attrs,
    everyone else by hitting attrs.
    """
    is_pitch = 1 if group == "pitching" else 0
    sub = rows[rows["is_pitcher_card"] == is_pitch]
    key = ["card_uuid", "as_of"]
    pivot = sub.pivot_table(index=key, columns="attr", values="ens", aggfunc="first")
    meta_cols = [c for c in ("ovr_before", "ovr_after") if c in sub]
    players = sub.groupby(key)[meta_cols].first()
    if "ovr_after" in players:
        players["ovr_delta"] = players["ovr_after"] - players["ovr_before"]
    return pivot, players.loc[pivot.index]


# ═══════════════════════════════════════════════════════════════════════════
#  Player-level distribution helpers (shared with predict.py)
# ═══════════════════════════════════════════════════════════════════════════

def _normal_move_probs(mu: np.ndarray, sd: float) -> np.ndarray:
    """Fallback before any out-of-fold data exists: discretized normal."""
    from scipy.stats import norm

    moves = np.array(OVR_MOVES)
    upper = norm.cdf((moves[None, :] + 0.5 - mu[:, None]) / sd)
    lower = norm.cdf((moves[None, :] - 0.5 - mu[:, None]) / sd)
    probs = upper - lower
    return probs / probs.sum(axis=1, keepdims=True)


def _cap_at_99(ovr: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """OVR can't exceed 99: fold impossible upward mass back onto the cap."""
    moves = np.array(OVR_MOVES)
    over = (ovr[:, None] + moves[None, :]) > 99
    if not over.any():
        return probs
    capped = (probs * over).sum(axis=1)
    probs = np.where(over, 0.0, probs)
    at_cap = (99 - ovr).clip(moves[0], moves[-1]) - moves[0]
    probs[np.arange(len(ovr)), at_cap.astype(int)] += capped
    return probs


def player_market_metrics(ovr: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    probs = _cap_at_99(ovr, probs)
    moves = np.array(OVR_MOVES)
    expected = (probs * moves).sum(axis=1)
    new_ovr = (ovr[:, None] + moves[None, :]).clip(40, 99)
    qs_now = np.array([quicksell_value(int(o)) for o in ovr])
    qs_new = np.vectorize(quicksell_value)(new_ovr)
    tiers = np.array([65, 75, 85, 90])  # Bronze/Silver/Gold/Diamond floors
    tier_now = np.searchsorted(tiers, ovr, side="right")
    tier_new = np.searchsorted(tiers, new_ovr, side="right")
    return pd.DataFrame({
        "upgrade_probability": (probs * (moves > 0)).sum(axis=1),
        "downgrade_probability": (probs * (moves < 0)).sum(axis=1),
        "tier_up_probability": (probs * (tier_new > tier_now[:, None])).sum(axis=1),
        "tier_down_probability": (probs * (tier_new < tier_now[:, None])).sum(axis=1),
        "expected_qs_change": (probs * (qs_new - qs_now[:, None])).sum(axis=1),
        "qs_upside": (probs * np.maximum(qs_new - qs_now[:, None], 0)).sum(axis=1),
        "current_qs": qs_now,
        "expected_move": expected,
        "sd": np.sqrt((probs * (moves[None, :] - expected[:, None]) ** 2).sum(axis=1)),
        "p_no_change": probs[:, OVR_MOVES.index(0)],
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Walk-forward backtest
# ═══════════════════════════════════════════════════════════════════════════

def _attr_metrics(df: pd.DataFrame) -> dict:
    y, p = df["delta"], df["ens"]
    moved = (y != 0) & (p.abs() >= 0.5)
    sse, sse0 = ((y - p) ** 2).sum(), (y ** 2).sum()
    rho = spearmanr(y, p).correlation if p.std() > 0 else 0.0
    out = {
        "attr_mae": float((y - p).abs().mean()),
        "attr_mae_baseline": float(y.abs().mean()),
        "attr_skill": float(1 - sse / sse0) if sse0 else 0.0,
        "attr_spearman": float(rho),
        "attr_direction_acc": float((np.sign(y[moved]) == np.sign(p[moved])).mean()) if moved.any() else None,
        "interval_coverage": float(((y >= df["lo"]) & (y <= df["hi"])).mean()),
    }
    for s in SIGNALS:
        out[f"{s}_mae"] = float((y - df[s]).abs().mean())
    return out


def _player_metrics(players: pd.DataFrame) -> dict:
    y, mu = players["ovr_delta"], players["mu"]
    up = (y > 0).astype(float)
    brier = float(((players["upgrade_probability"] - up) ** 2).mean())
    brier0 = float(((up.mean() - up) ** 2).mean())
    top_up = players.nlargest(25, "mu")
    top_dn = players.nsmallest(25, "mu")
    top_ev = players.nlargest(25, "expected_qs_change")
    realized_qs = players["ovr_after"].map(quicksell_value) - players["ovr_before"].map(quicksell_value)
    return {
        "ovr_mae": float((y - mu).abs().mean()),
        "ovr_mae_baseline": float(y.abs().mean()),
        "ovr_spearman": float(spearmanr(y, mu).correlation),
        "upgrade_brier": brier,
        "upgrade_brier_baseline": brier0,
        "upgrade_prob_mean": float(players["upgrade_probability"].mean()),
        "downgrade_prob_mean": float(players["downgrade_probability"].mean()),
        "top25_up_hit_rate": float((top_up["ovr_delta"] > 0).mean()),
        "top25_up_avg_ovr_delta": float(top_up["ovr_delta"].mean()),
        "top25_down_hit_rate": float((top_dn["ovr_delta"] < 0).mean()),
        "top25_ev_avg_qs_gain": float(realized_qs.loc[top_ev.index].mean()),
        "all_avg_qs_gain": float(realized_qs.mean()),
        "base_rate_up": float((y > 0).mean()),
        "base_rate_down": float((y < 0).mean()),
        "n_players": int(len(players)),
    }


def score_rows(rows: pd.DataFrame, sig: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    out = rows.join(sig)
    ens = cal.ensemble(out)
    lo, hi = cal.interval(ens)
    return pd.concat([out, pd.DataFrame({"ens": ens, "lo": lo, "hi": hi})], axis=1)


def score_players(scored: pd.DataFrame, cals: dict[str, Calibration]) -> pd.DataFrame:
    frames = []
    for group, cal in cals.items():
        pivot, players = player_frame(scored, group)
        if players.empty:
            continue
        players = players.copy()
        players["mu"] = cal.ovr_mean(pivot)
        ovr = players["ovr_before"].to_numpy(int)
        mm = player_market_metrics(ovr, cal.ovr_move_probs(players["mu"].to_numpy(), ovr))
        mm.index = players.index
        frames.append(players.join(mm))
    return pd.concat(frames) if frames else pd.DataFrame()


def walk_forward(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Returns (out-of-fold attribute rows, per-fold metrics, last fold's scored players)."""
    dates = sorted(data["as_of"].unique())
    oof_frames: list[pd.DataFrame] = []
    folds = []
    players = pd.DataFrame()
    for k in range(1, len(dates)):
        train, test = data[data["as_of"] < dates[k]], data[data["as_of"] == dates[k]]
        prior_oof = pd.concat(oof_frames) if oof_frames else pd.DataFrame()
        fold_scored, fold_raw, cals = [], [], {}
        for group in CORE_ATTRS:
            tr, te = train[train["group"] == group], test[test["group"] == group]
            sig = GroupModel(group).fit(tr).predict(te)
            fold_raw.append(te.join(sig))
            g_oof = prior_oof[prior_oof["group"] == group] if len(prior_oof) else prior_oof
            cals[group] = fit_calibration(g_oof, group)
            fold_scored.append(score_rows(te, sig, cals[group]))
        scored = pd.concat(fold_scored)
        players = score_players(scored, cals)
        metrics = {"test_update": dates[k], "train_updates": k, **_attr_metrics(scored), **_player_metrics(players)}
        folds.append(metrics)
        logger.info(
            "  fold %s: attr MAE %.3f (baseline %.3f) | OVR MAE %.3f (baseline %.3f) | top25 up hit %.0f%%",
            dates[k], metrics["attr_mae"], metrics["attr_mae_baseline"],
            metrics["ovr_mae"], metrics["ovr_mae_baseline"], 100 * metrics["top25_up_hit_rate"],
        )
        oof_frames.append(pd.concat(fold_raw))
    return pd.concat(oof_frames), folds, players


def last_update_review(players: pd.DataFrame, data: pd.DataFrame, n: int = 25) -> dict:
    """The backtest's picks for the most recent update next to what actually happened."""
    if players.empty:
        return {}
    p = players.reset_index()
    names = data.groupby("card_uuid")["player_name"].first()
    p["player_name"] = p["card_uuid"].map(names)
    p["realized_qs"] = p["ovr_after"].map(quicksell_value) - p["ovr_before"].map(quicksell_value)

    def rows(frame: pd.DataFrame) -> list[dict]:
        return [
            {
                "card_uuid": r.card_uuid, "player_name": r.player_name,
                "ovr_before": int(r.ovr_before), "ovr_after": int(r.ovr_after),
                "predicted_delta": round(float(r.mu), 2),
                "upgrade_probability": round(float(r.upgrade_probability), 3),
                "downgrade_probability": round(float(r.downgrade_probability), 3),
                "expected_qs_change": round(float(r.expected_qs_change), 1),
                "realized_qs_change": int(r.realized_qs),
            }
            for r in frame.itertuples()
        ]

    return {
        "update": str(p["as_of"].iloc[0]),
        "top_upgrades": rows(p.nlargest(n, "upgrade_probability")),
        "top_downgrades": rows(p.nlargest(n, "downgrade_probability")),
        "top_value": rows(p.nlargest(n, "expected_qs_change")),
    }


def summarize(folds: list[dict]) -> dict:
    """Mean over folds that had calibration fitted from earlier folds."""
    calibrated = folds[1:] if len(folds) > 1 else folds
    keys = [k for k in calibrated[0] if isinstance(calibrated[0][k], (int, float)) and k != "train_updates"]
    summary = {}
    for k in keys:
        vals = [f[k] for f in calibrated if f.get(k) is not None]
        summary[k] = float(np.mean(vals)) if vals else None
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def train_all(data: pd.DataFrame | None = None) -> dict:
    if data is None:
        path = PROCESSED_DIR / "training_examples.parquet"
        if path.exists():
            data = pd.read_parquet(path)
        else:
            from src.features.dataset import build_training_dataset
            data = build_training_dataset()
    if data.empty or data["as_of"].nunique() < 2:
        raise ValueError("Need at least two major attribute updates to train. Run the backfill first.")
    if LGBMRegressor is None:
        raise ImportError("lightgbm is required for training")

    logger.info("Walk-forward backtest over %d updates...", data["as_of"].nunique())
    oof, folds, last_players = walk_forward(data)

    logger.info("Fitting final models on all %d rows...", len(data))
    groups, cals = {}, {}
    for group in CORE_ATTRS:
        groups[group] = GroupModel(group).fit(data[data["group"] == group])
        cals[group] = fit_calibration(oof[oof["group"] == group], group)
        logger.info("  %s stacking weights: %s", group, {k: round(v, 3) for k, v in cals[group].weights.items()})

    importances = {
        g: dict(sorted(
            zip(_model_cols(g), (float(v) for v in m.gbm.booster_.feature_importance("gain"))),
            key=lambda kv: -kv[1],
        )[:15])
        for g, m in groups.items()
    }
    summary = {
        "trained_on": sorted(data["as_of"].unique().tolist()),
        "n_rows": int(len(data)),
        "folds": folds,
        "summary": summarize(folds),
        "stacking_weights": {g: c.weights for g, c in cals.items()},
        "ovr_weights": {g: c.ovr_coef for g, c in cals.items()},
        "stat_cutoff_days": int(data.attrs.get("lag_days", STAT_CUTOFF_DAYS)),
        "last_update_review": last_update_review(last_players, data),
        "top_features": importances,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"groups": groups, "calibration": cals, "version": 2}, MODEL_PATH, compress=3)
    text = json.dumps(_json_safe(summary), indent=2)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "backtest_summary.json").write_text(text, encoding="utf-8")
    logger.info("Model saved to %s", MODEL_PATH)
    return summary


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def load_summary() -> dict:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}
