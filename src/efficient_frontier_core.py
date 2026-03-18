"""
Comparaison des stratégies Core via frontière efficiente (données daily).

Entrée :
- outputs/core3_etf_daily_log_returns.csv  (log-rendements journaliers des 3 ETF)

Fenêtre IS  : 2019-01-01 → 2020-12-31  (calibration, sans look-ahead)
Fenêtre OOS : 2021-01-01 → 2025-12-31  (évaluation)

5 stratégies comparées :
  1. Max Sharpe  – poids optimisés (Max Sharpe) sur IS, appliqués fixement sur OOS
  2. Min Variance – poids Min Variance sur IS, fixes sur OOS
  3. Equal Weight – 1/3 chacun (benchmark naïf)
  4. Risk Parity  – poids ∝ 1/vol_IS, normalisés
  5. Max Sharpe Rolling – recalibré tous les 63j (rolling 252j, référence core_pipeline)

Sorties :
- outputs/figures/06_efficient_frontier_core.png    (nuage + 5 stratégies)
- outputs/figures/07_core_strategies_oos_perf.png   (perfs OOS cumulées)
- outputs/core_portfolio_comparison.csv             (métriques IS + OOS)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import minimize

project_root = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FrontierConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    input_csv:    Path = project_root / "outputs" / "core3_etf_daily_log_returns.csv"
    fig_dir:      Path = project_root / "outputs" / "figures"
    out_csv:      Path = project_root / "outputs" / "core_portfolio_comparison.csv"

    calib_start: str = "2019-01-01"
    calib_end:   str = "2020-12-31"
    oos_start:   str = "2021-01-01"
    oos_end:     str = "2025-12-31"

    n_sim:       int   = 15_000
    rf:          float = 0.0
    w_min:       float = 0.05
    w_max:       float = 0.50

    rolling_lookback: int = 252
    rolling_rebal:    int = 63
    equity_floor:     float = 0.30   # floor poids Equity (colonne 0), cohérent avec core_pipeline
    dpi:          int   = 160


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers statistiques (daily)
# ══════════════════════════════════════════════════════════════════════════════

def _stats_daily(log_rets: pd.DataFrame, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """μ et Σ annualisés à partir de log-rendements journaliers."""
    mu  = log_rets.mean().values * 252
    cov = log_rets.cov().values  * 252
    return mu, cov


def _port_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Tuple[float, float, float]:
    """Rendement, vol, Sharpe annualisés d'un portefeuille."""
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))
    s = r / v if v > 1e-10 else np.nan
    return r, v, s


def _batch_stats(W: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ret, vol, sharpe) pour une matrice N×k de poids."""
    rets   = W @ mu
    vols   = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    sharpe = np.where(vols > 1e-10, rets / vols, np.nan)
    return rets, vols, sharpe


def _sim_portfolios(k: int, n: int, seed: int = 42) -> np.ndarray:
    """Dirichlet random portfolios."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(k), size=n)


# ══════════════════════════════════════════════════════════════════════════════
#  Optimiseurs
# ══════════════════════════════════════════════════════════════════════════════

def _opt_max_sharpe(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n
    def neg_sr(w):
        v = np.sqrt(w @ cov @ w)
        return -(w @ mu) / v if v > 1e-12 else 1e10
    res = minimize(neg_sr, w0, method="SLSQP",
                   bounds=[(w_min, w_max)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return np.maximum(res.x, 0) / np.maximum(res.x, 0).sum() if res.success else w0


def _opt_min_var(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n
    def port_var(w):
        return w @ cov @ w
    res = minimize(port_var, w0, method="SLSQP",
                   bounds=[(w_min, w_max)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return np.maximum(res.x, 0) / np.maximum(res.x, 0).sum() if res.success else w0


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    """Poids inversement proportionnels à la vol individuelle."""
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / (vols + 1e-12)
    return inv_vol / inv_vol.sum()


# ══════════════════════════════════════════════════════════════════════════════
#  Backtest OOS
# ══════════════════════════════════════════════════════════════════════════════

def _backtest_fixed(
    log_rets_oos: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Rendements simples journaliers OOS avec poids fixes."""
    r = np.exp(log_rets_oos.values) - 1.0          # simple rets per asset
    port = r @ weights
    return pd.Series(port, index=log_rets_oos.index, name="ret")


def _backtest_rolling(
    log_rets: pd.DataFrame,
    w_min: float, w_max: float,
    lookback: int, rebal: int,
    oos_start: str, oos_end: str,
    equity_floor: float = 0.0,
) -> pd.Series:
    """Backtest rolling Max Sharpe (cohérent avec core_pipeline, equity floor inclus)."""
    dates = log_rets.index
    port_rets: List[float] = []
    port_dates: List[pd.Timestamp] = []
    for start in range(lookback, len(dates) - rebal, rebal):
        window = log_rets.iloc[start - lookback:start]
        oos    = log_rets.iloc[start:start + rebal]
        mu  = window.mean().values * 252
        cov = window.cov().values  * 252
        w   = _opt_max_sharpe(mu, cov, w_min, w_max)
        # Appliquer le floor Equity (colonne 0)
        if equity_floor > 0 and w[0] < equity_floor:
            deficit = equity_floor - w[0]
            w[0] = equity_floor
            others_sum = w[1:].sum()
            if others_sum > 1e-12:
                w[1:] -= deficit * (w[1:] / others_sum)
            w = np.clip(w, w_min, w_max)
            w /= w.sum()
        r_oos = (np.exp(oos.values) - 1.0) @ w
        port_rets.extend(r_oos.tolist())
        port_dates.extend(oos.index.tolist())
    s = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates)).sort_index()
    return s.loc[oos_start:oos_end].rename("ret")


def _perf_metrics(ret: pd.Series) -> Dict:
    """Métriques annualisées à partir de rendements simples journaliers."""
    n = len(ret)
    ann_ret = float((1 + ret).prod() ** (252 / n) - 1) if n > 0 else np.nan
    ann_vol = float(ret.std() * np.sqrt(252))
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-10 else np.nan
    cum     = (1 + ret).cumprod()
    mdd     = float(((cum / cum.cummax()) - 1).min())
    return {"ret_ann": ann_ret, "vol_ann": ann_vol, "sharpe": sharpe, "mdd": mdd}


# ══════════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Max Sharpe":         "#e6194b",
    "Min Variance":       "#3cb44b",
    "Equal Weight":       "#4363d8",
    "Risk Parity":        "#f58231",
    "Max Sharpe Rolling": "#911eb4",
}


def _plot_frontier(
    sim_rets: np.ndarray, sim_vols: np.ndarray, sim_sharpe: np.ndarray,
    strategies: Dict,
    tickers: List[str],
    fig_dir: Path, dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(sim_vols, sim_rets, c=sim_sharpe, cmap="viridis",
                    s=5, alpha=0.3, label="_nolegend_")
    plt.colorbar(sc, ax=ax, label="Sharpe (IS)")

    markers = {"Max Sharpe": "*", "Min Variance": "D", "Equal Weight": "P",
               "Risk Parity": "^", "Max Sharpe Rolling": "s"}
    for name, d in strategies.items():
        if "w" not in d:
            continue
        w = d["w"]
        ax.scatter(d["vol_is"], d["ret_is"], marker=markers.get(name, "o"),
                   s=160, zorder=5, color=COLORS.get(name, "grey"),
                   label=f"{name}  Sharpe={d['sharpe_is']:.2f}")
        for i, t in enumerate(tickers):
            ax.annotate(f"{w[i]:.0%}", (d["vol_is"], d["ret_is"]),
                        textcoords="offset points", xytext=(6, -4*(i-1)),
                        fontsize=7, color=COLORS.get(name, "grey"))

    ax.set_xlabel("Volatilité annualisée (IS 2019-2020)")
    ax.set_ylabel("Rendement annualisé (IS 2019-2020)")
    ax.set_title("Frontière efficiente – 3 ETF Core\n(calibration 2019-2020, daily)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "06_efficient_frontier_core.png", dpi=dpi)
    plt.close()


def _plot_oos_perf(strategies: Dict, fig_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Cumulative OOS ────────────────────────────────────────────────────────
    ax = axes[0]
    for name, d in strategies.items():
        if "oos_ret" not in d:
            continue
        cum = 100 * (1 + d["oos_ret"]).cumprod()
        ax.plot(cum.index, cum.values, label=name, color=COLORS.get(name))
    ax.set_title("Performance cumulée OOS 2021-2025 (base 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Métriques OOS bar chart ───────────────────────────────────────────────
    ax2 = axes[1]
    names  = list(strategies.keys())
    sharpes = [strategies[n].get("oos_sharpe", 0) for n in names]
    bars = ax2.bar(range(len(names)), sharpes,
                   color=[COLORS.get(n, "grey") for n in names])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax2.set_title("Sharpe OOS 2021-2025 par stratégie")
    ax2.set_ylabel("Sharpe")
    ax2.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "07_core_strategies_oos_perf.png", dpi=dpi)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = FrontierConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMPARAISON DES STRATÉGIES CORE (Markowitz daily)")
    print(f"  IS   : {cfg.calib_start} → {cfg.calib_end}")
    print(f"  OOS  : {cfg.oos_start} → {cfg.oos_end}")
    print("=" * 60)

    # ── Chargement ────────────────────────────────────────────────────────────
    print("\n[1] Chargement des log-rendements journaliers des 3 ETF...")
    df = pd.read_csv(cfg.input_csv, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df.sort_index()
    tickers = list(df.columns)
    print(f"  ETFs : {tickers}")
    print(f"  Période : {df.index.min().date()} → {df.index.max().date()}")

    # ── Fenêtres ──────────────────────────────────────────────────────────────
    is_df  = df.loc[cfg.calib_start:cfg.calib_end].dropna()
    oos_df = df.loc[cfg.oos_start:cfg.oos_end].dropna()
    print(f"  IS  : {len(is_df)} obs | OOS : {len(oos_df)} obs")

    mu_is, cov_is = _stats_daily(is_df, None)

    # ── Portefeuilles simulés sur IS ──────────────────────────────────────────
    print("\n[2] Simulation de la frontière (IS)...")
    W_sim  = _sim_portfolios(len(tickers), cfg.n_sim)
    s_rets, s_vols, s_sharpes = _batch_stats(W_sim, mu_is, cov_is)

    # ── 4 stratégies statiques ────────────────────────────────────────────────
    print("\n[3] Optimisation des 4 stratégies statiques sur IS...")
    strat_configs = {
        "Max Sharpe":    _opt_max_sharpe(mu_is, cov_is, cfg.w_min, cfg.w_max),
        "Min Variance":  _opt_min_var(mu_is, cov_is, cfg.w_min, cfg.w_max),
        "Equal Weight":  np.ones(len(tickers)) / len(tickers),
        "Risk Parity":   _risk_parity(cov_is),
    }

    strategies: Dict = {}
    for name, w in strat_configs.items():
        r_is, v_is, s_is = _port_stats(w, mu_is, cov_is)
        oos_ret = _backtest_fixed(oos_df, w)
        m_oos   = _perf_metrics(oos_ret)
        strategies[name] = {
            "w":          w,
            "ret_is":     r_is,  "vol_is": v_is, "sharpe_is": s_is,
            "oos_ret":    oos_ret,
            "oos_ret_ann": m_oos["ret_ann"], "oos_vol": m_oos["vol_ann"],
            "oos_sharpe":  m_oos["sharpe"],  "oos_mdd": m_oos["mdd"],
        }
        print(f"  {name:<22s}  w=[{', '.join(f'{x:.1%}' for x in w)}]  "
              f"IS Sharpe={s_is:.2f}  OOS Sharpe={m_oos['sharpe']:.2f}")

    # ── Max Sharpe Rolling ────────────────────────────────────────────────────
    print("\n[4] Backtest rolling Max Sharpe (252j lookback, 63j rebal, equity floor 30%)...")
    roll_ret_is = _backtest_rolling(
        df, cfg.w_min, cfg.w_max,
        cfg.rolling_lookback, cfg.rolling_rebal,
        cfg.calib_start, cfg.calib_end,
        equity_floor=cfg.equity_floor,
    )
    roll_ret = _backtest_rolling(
        df, cfg.w_min, cfg.w_max,
        cfg.rolling_lookback, cfg.rolling_rebal,
        cfg.oos_start, cfg.oos_end,
        equity_floor=cfg.equity_floor,
    )
    m_roll_is = _perf_metrics(roll_ret_is)
    m_roll = _perf_metrics(roll_ret)
    strategies["Max Sharpe Rolling"] = {
        "ret_is":      m_roll_is["ret_ann"],
        "vol_is":      m_roll_is["vol_ann"],
        "sharpe_is":   m_roll_is["sharpe"],
        "oos_ret":    roll_ret,
        "oos_ret_ann": m_roll["ret_ann"], "oos_vol": m_roll["vol_ann"],
        "oos_sharpe":  m_roll["sharpe"],  "oos_mdd": m_roll["mdd"],
    }
    print(f"  {'Max Sharpe Rolling':<22s}  (rolling)  "
          f"IS Sharpe={m_roll_is['sharpe']:.2f}  OOS Sharpe={m_roll['sharpe']:.2f}")

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TABLEAU COMPARATIF")
    print("=" * 60)
    rows = []
    for name, d in strategies.items():
        poids_str = f"[{', '.join(f'{x:.1%}' for x in d['w'])}]" if "w" in d else "rolling"
        row = {
            "strategie":    name,
            "poids":        poids_str,
            "IS_ret_ann":   d.get("ret_is", np.nan),
            "IS_vol_ann":   d.get("vol_is", np.nan),
            "IS_sharpe":    d.get("sharpe_is", np.nan),
            "OOS_ret_ann":  d["oos_ret_ann"],
            "OOS_vol_ann":  d["oos_vol"],
            "OOS_sharpe":   d["oos_sharpe"],
            "OOS_mdd":      d["oos_mdd"],
        }
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index("strategie")

    print(f"\n{'Stratégie':<26} {'IS Ret':>8} {'IS Vol':>8} {'IS Sh':>7} "
          f"{'OOS Ret':>8} {'OOS Vol':>8} {'OOS Sh':>7} {'OOS MDD':>9}")
    print("-" * 85)
    for name, row in comp_df.iterrows():
        print(f"{name:<26} {row['IS_ret_ann']:>7.1%} {row['IS_vol_ann']:>8.1%} "
              f"{row['IS_sharpe']:>7.2f} {row['OOS_ret_ann']:>8.1%} "
              f"{row['OOS_vol_ann']:>8.1%} {row['OOS_sharpe']:>7.2f} {row['OOS_mdd']:>8.1%}")

    best = comp_df["OOS_sharpe"].idxmax()
    print(f"\n  ★  Meilleure stratégie OOS (Sharpe) : {best}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    comp_df.to_csv(cfg.out_csv)
    print(f"\n  -> {cfg.out_csv}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[5] Génération des figures...")
    _plot_frontier(s_rets, s_vols, s_sharpes, strategies, tickers, cfg.fig_dir, cfg.dpi)
    print(f"  -> {cfg.fig_dir / '06_efficient_frontier_core.png'}")

    _plot_oos_perf(strategies, cfg.fig_dir, cfg.dpi)
    print(f"  -> {cfg.fig_dir / '07_core_strategies_oos_perf.png'}")

    print("\n  ✓  Frontière efficiente terminée.")
    return strategies, comp_df


if __name__ == "__main__":
    main()
