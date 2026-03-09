"""
plots_report.py – Génération complète des graphiques Core-Satellite (daily).

Sections :
  A. Poche Core          (7 figures)
  B. Poche Satellite     (7 figures)
    C. Core vs Satellite   (4 figures)
  D. Portefeuille total  (6 figures)
                         ─────────────
                                                 24 figures au total

Entrées (outputs/) :
  core_returns_daily_oos.csv          log-rendements journaliers Core portfolio
  core3_etf_daily_log_returns.csv     log-rendements des 3 ETF Core individuels
  Core_finaux.csv                     tickers + thèmes sélectionnés
  fond_returns_daily.csv              core_ret / sat_pocket_ret / portfolio_ret
  fond_weights.csv                    poids satellites + allocation globale
  fond_metrics.csv                    métriques scalaires
  fond_annual_perf.csv                performance annuelle
  fond_beta_rolling.csv               beta rolling 63j de la poche satellite
  satellite_selected.csv              métadata des fonds satellites
  satellite_individual_returns.csv    rendements OOS de chaque fonds satellite

Sorties : outputs/figures/  (A01... → D06...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         10,
})

project_root = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PlotConfig:
    fig_dir: Path = project_root / "outputs" / "figures"
    dpi:     int  = 150

    # inputs
    core_oos_csv:        Path = project_root / "outputs" / "core_returns_daily_oos.csv"
    core_etf_csv:        Path = project_root / "outputs" / "core3_etf_daily_log_returns.csv"
    core_finaux_csv:     Path = project_root / "outputs" / "Core_finaux.csv"
    fond_returns_csv:    Path = project_root / "outputs" / "fond_returns_daily.csv"
    fond_weights_csv:    Path = project_root / "outputs" / "fond_weights.csv"
    fond_metrics_csv:    Path = project_root / "outputs" / "fond_metrics.csv"
    fond_annual_csv:     Path = project_root / "outputs" / "fond_annual_perf.csv"
    fond_beta_roll_csv:  Path = project_root / "outputs" / "fond_beta_rolling.csv"
    sat_selected_csv:    Path = project_root / "outputs" / "satellite_selected.csv"
    sat_indiv_csv:       Path = project_root / "outputs" / "satellite_individual_returns.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cum(r: pd.Series, base: float = 100.0) -> pd.Series:
    return base * (1 + r).cumprod()

def _dd(r: pd.Series) -> pd.Series:
    w = (1 + r).cumprod()
    return w / w.cummax() - 1

def _roll_vol(r: pd.Series, w: int = 63) -> pd.Series:
    return r.rolling(w).std() * np.sqrt(252)

def _roll_sharpe(r: pd.Series, w: int = 63) -> pd.Series:
    roll_ret = r.rolling(w).mean() * 252
    roll_vol = r.rolling(w).std() * np.sqrt(252)
    return roll_ret / roll_vol.replace(0, np.nan)

def _roll_alpha_beta(y: pd.Series, x: pd.Series, w: int = 63):
    cov_  = y.rolling(w).cov(x)
    var_  = x.rolling(w).var()
    beta  = cov_ / var_.replace(0, np.nan)
    alpha = y.rolling(w).mean() - beta * x.rolling(w).mean()
    alpha_ann = (1 + alpha) ** 252 - 1
    return alpha_ann.rename("alpha_ann_roll"), beta.rename("beta_roll")

def _save(fig_dir: Path, name: str, dpi: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"  -> {fig_dir / name}")

BLOC_COLORS = {"Bloc1": "#e6194b", "Bloc2": "#3cb44b", "Bloc3": "#4363d8"}
ETF_COLORS  = ["#e6194b", "#3cb44b", "#4363d8"]


# ══════════════════════════════════════════════════════════════════════════════
#  Section A — Poche Core
# ══════════════════════════════════════════════════════════════════════════════

def plot_A01_core_etf_cum(log_etf: pd.DataFrame, core_finaux: pd.DataFrame,
                          cfg: PlotConfig) -> None:
    """A01 – Perf cumulée des 3 ETF Core individuels (OOS 2021-2025)."""
    oos = log_etf.loc["2021":"2025"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(oos.columns):
        r = np.exp(oos[col]) - 1
        theme = core_finaux.loc[core_finaux["Ticker"] == col, "Theme"].values
        lbl = f"{col}  [{theme[0] if len(theme) else '?'}]"
        _cum(r).plot(ax=ax, label=lbl, color=ETF_COLORS[i % 3])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_title("A01 – Performance cumulée des 3 ETF Core (OOS 2021-2025, base 100)")
    ax.set_xlabel("Date"); ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "A01_core_etf_cum.png", cfg.dpi)


def plot_A02_core_portfolio_cum(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A02 – Perf cumulée du portefeuille Core Max-Sharpe Rolling (OOS)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _cum(r_core).plot(ax=ax, color="#e6194b", lw=2)
    ax.set_title("A02 – Performance cumulée du Core (Max-Sharpe rolling, base 100)")
    ax.set_xlabel("Date"); ax.set_ylabel("Valeur (base 100)")
    _save(cfg.fig_dir, "A02_core_portfolio_cum.png", cfg.dpi)


def plot_A03_core_drawdown(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A03 – Drawdown du portefeuille Core."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _dd(r_core).mul(100).plot(ax=ax, color="#e6194b", lw=1.5)
    ax.fill_between(_dd(r_core).index, _dd(r_core).mul(100), 0, alpha=0.25, color="#e6194b")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("A03 – Drawdown du portefeuille Core")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown (%)")
    _save(cfg.fig_dir, "A03_core_drawdown.png", cfg.dpi)


def plot_A04_core_rolling_vol(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A04 – Volatilité rolling 63j du Core (annualisée)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _roll_vol(r_core).mul(100).plot(ax=ax, color="#e6194b")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("A04 – Volatilité rolling 63j du Core (annualisée)")
    ax.set_xlabel("Date"); ax.set_ylabel("Vol annualisée (%)")
    _save(cfg.fig_dir, "A04_core_rolling_vol.png", cfg.dpi)


def plot_A05_core_rolling_sharpe(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A05 – Sharpe rolling 63j du Core."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _roll_sharpe(r_core).plot(ax=ax, color="#e6194b")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("A05 – Sharpe rolling 63j du Core")
    ax.set_xlabel("Date"); ax.set_ylabel("Sharpe")
    _save(cfg.fig_dir, "A05_core_rolling_sharpe.png", cfg.dpi)


def plot_A06_core_annual_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """A06 – Performance annuelle du Core."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(annual.index.astype(str), annual["core"].mul(100),
                  color=["#e6194b" if v >= 0 else "#888888" for v in annual["core"]])
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, v in zip(bars, annual["core"].mul(100)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.3 if v >= 0 else -0.8),
                f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title("A06 – Performance annuelle du Core")
    ax.set_xlabel("Année"); ax.set_ylabel("Rendement (%)")
    _save(cfg.fig_dir, "A06_core_annual_bar.png", cfg.dpi)


def plot_A07_core_fees_bar(core_finaux: pd.DataFrame, cfg: PlotConfig) -> None:
    """A07 – Frais des ETF Core (expense ratio indicatif)."""
    # Expense ratios indicatifs DJSC/CBE3/IEAC
    fee_map = {"DJSC LN Equity": 35, "CBE3 LN Equity": 15, "IEAC LN Equity": 20}
    tickers = core_finaux["Ticker"].tolist()
    fees    = [fee_map.get(t, 25) for t in tickers]
    themes  = core_finaux["Theme"].tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(tickers, fees, color=ETF_COLORS[:len(tickers)])
    ax.set_ylabel("Expense ratio (bps/an)")
    ax.set_title("A07 – Frais des ETF Core (expense ratio indicatif)")
    for bar, v in zip(bars, fees):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v} bps",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels([f"{t}\n({th})" for t, th in zip(tickers, themes)], fontsize=8)
    _save(cfg.fig_dir, "A07_core_fees_bar.png", cfg.dpi)


# ══════════════════════════════════════════════════════════════════════════════
#  Section B — Poche Satellite
# ══════════════════════════════════════════════════════════════════════════════

def plot_B01_sat_fund_cum(sat_indiv: pd.DataFrame, sat_info: pd.DataFrame,
                          sat_weights: pd.Series, cfg: PlotConfig) -> None:
    """B01 – Perf cumulée de chaque fonds satellite (OOS)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(sat_weights.index):
        if t not in sat_indiv.columns:
            continue
        bloc = sat_info.loc[t, "bloc"] if t in sat_info.index else "?"
        _cum(sat_indiv[t]).plot(ax=ax, label=f"{t} [{bloc}]",
                                color=cmap(i), lw=1.5)
    ax.set_title("B01 – Performance cumulée individuelle des fonds satellite (base 100)")
    ax.set_xlabel("Date"); ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=8, ncol=2)
    _save(cfg.fig_dir, "B01_sat_fund_cum.png", cfg.dpi)


def plot_B02_sat_pocket_cum(r_sat: pd.Series, cfg: PlotConfig) -> None:
    """B02 – Perf cumulée de la poche satellite agrégée."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _cum(r_sat).plot(ax=ax, color="#3cb44b", lw=2)
    ax.set_title("B02 – Performance cumulée de la poche satellite (base 100)")
    ax.set_xlabel("Date"); ax.set_ylabel("Valeur (base 100)")
    _save(cfg.fig_dir, "B02_sat_pocket_cum.png", cfg.dpi)


def plot_B03_sat_weights_bar(sat_weights: pd.Series, sat_info: pd.DataFrame,
                              w_sat: float, cfg: PlotConfig) -> None:
    """B03 – Poids optimisés dans la poche satellite, colorés par bloc."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart poids dans la poche
    ax = axes[0]
    colors = [BLOC_COLORS.get(
        sat_info.at[t, "bloc"] if t in sat_info.index else "Inconnu", "#aaaaaa")
        for t in sat_weights.index]
    bars = ax.bar(range(len(sat_weights)), sat_weights.mul(100), color=colors)
    ax.set_xticks(range(len(sat_weights)))
    ax.set_xticklabels(sat_weights.index, rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("B03a – Poids dans la poche satellite (θᵢ)")
    ax.set_ylabel("Poids (%)")
    for bar, v in zip(bars, sat_weights.mul(100)):
        if v > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    # Pie des poids absolus dans le portefeuille total
    ax2 = axes[1]
    abs_weights = pd.concat([
        pd.Series({"Core (total)": 1 - w_sat}),
        sat_weights.mul(w_sat).rename(lambda t: t.split(" ")[0]),
    ])
    wedge_colors = ["#cccccc"] + [
        BLOC_COLORS.get(
            sat_info.at[t, "bloc"] if t in sat_info.index else "", "#aaaaaa")
        for t in sat_weights.index
    ]
    abs_weights.plot.pie(ax=ax2, colors=wedge_colors, autopct="%1.1f%%",
                         startangle=90, textprops={"fontsize": 8})
    ax2.set_title("B03b – Allocation totale du portefeuille")
    ax2.set_ylabel("")
    _save(cfg.fig_dir, "B03_sat_weights.png", cfg.dpi)


def plot_B04_sat_rolling_alpha(r_sat: pd.Series, r_core: pd.Series,
                                cfg: PlotConfig) -> None:
    """B04 – Alpha rolling 63j de la poche satellite vs Core."""
    alpha_r, _ = _roll_alpha_beta(r_sat, r_core, w=63)
    fig, ax = plt.subplots(figsize=(10, 4))
    alpha_r.mul(100).plot(ax=ax, color="#3cb44b")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(alpha_r.index, alpha_r.mul(100), 0,
                    where=alpha_r > 0, alpha=0.2, color="#3cb44b", label="Alpha > 0")
    ax.fill_between(alpha_r.index, alpha_r.mul(100), 0,
                    where=alpha_r <= 0, alpha=0.2, color="#e6194b", label="Alpha < 0")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("B04 – Alpha rolling 63j de la poche satellite vs Core (annualisé)")
    ax.set_xlabel("Date"); ax.set_ylabel("Alpha annualisé (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "B04_sat_rolling_alpha.png", cfg.dpi)


def plot_B05_sat_rolling_beta(beta_roll: pd.Series, cfg: PlotConfig) -> None:
    """B05 – Beta rolling 63j de la poche satellite vs Core."""
    fig, ax = plt.subplots(figsize=(10, 4))
    beta_roll.plot(ax=ax, color="#4363d8", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline( 0.15, color="orange", lw=1, ls=":", label="+β_max")
    ax.axhline(-0.15, color="orange", lw=1, ls=":", label="−β_max")
    ax.fill_between(beta_roll.index, beta_roll, 0, alpha=0.15, color="#4363d8")
    ax.set_title("B05 – Beta rolling 63j de la poche satellite vs Core")
    ax.set_xlabel("Date"); ax.set_ylabel("Beta")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "B05_sat_rolling_beta.png", cfg.dpi)


def plot_B06_sat_perf_annual(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """B06 – Performance annuelle de la poche satellite."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(annual.index.astype(str), annual["satellite"].mul(100),
                  color=["#3cb44b" if v >= 0 else "#888888" for v in annual["satellite"]])
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, v in zip(bars, annual["satellite"].mul(100)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.2 if v >= 0 else -0.8),
                f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title("B06 – Performance annuelle de la poche satellite")
    ax.set_xlabel("Année"); ax.set_ylabel("Rendement (%)")
    _save(cfg.fig_dir, "B06_sat_annual.png", cfg.dpi)


def plot_B07_sat_fees_bar(sat_weights: pd.Series, sat_info: pd.DataFrame,
                           cfg: PlotConfig) -> None:
    """B07 – Frais des fonds satellite pondérés."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Frais bruts de chaque fonds
    ax = axes[0]
    expense = []
    for t in sat_weights.index:
        e = sat_info.at[t, "expense_pct"] if t in sat_info.index and "expense_pct" in sat_info.columns else np.nan
        expense.append(e * 100 if not np.isnan(e) else 200)
    colors = [BLOC_COLORS.get(
        sat_info.at[t, "bloc"] if t in sat_info.index else "", "#aaaaaa")
        for t in sat_weights.index]
    ax.bar(range(len(sat_weights)), expense, color=colors)
    ax.set_xticks(range(len(sat_weights)))
    ax.set_xticklabels(sat_weights.index, rotation=35, ha="right", fontsize=8)
    ax.set_title("B07a – Expense ratio de chaque fonds satellite (bps/an)")
    ax.set_ylabel("bps/an")

    # Contribution aux frais totaux
    ax2 = axes[1]
    w_sat    = sat_weights.values
    contrib  = [w * e for w, e in zip(w_sat, expense)]
    ax2.bar(range(len(sat_weights)), contrib, color=colors)
    ax2.set_xticks(range(len(sat_weights)))
    ax2.set_xticklabels(sat_weights.index, rotation=35, ha="right", fontsize=8)
    ax2.set_title("B07b – Contribution aux frais de la poche satellite (poids×frais)")
    ax2.set_ylabel("Contribution (bps × poids)")
    _save(cfg.fig_dir, "B07_sat_fees.png", cfg.dpi)


# ══════════════════════════════════════════════════════════════════════════════
#  Section C — Core vs Satellite
# ══════════════════════════════════════════════════════════════════════════════

def plot_C01_core_vs_sat_cum(r_core: pd.Series, r_sat: pd.Series,
                              r_port: pd.Series, cfg: PlotConfig) -> None:
    """C01 – Perf cumulée : Core vs Satellite vs Portefeuille total."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _cum(r_core).plot(ax=ax, label="Core (Max-Sharpe rolling)",  color="#e6194b", lw=2)
    _cum(r_sat ).plot(ax=ax, label="Satellite (poche agrégée)",   color="#3cb44b", lw=2, ls="--")
    _cum(r_port).plot(ax=ax, label="Portefeuille total",          color="#4363d8", lw=2.5)
    ax.set_title("C01 – Performance cumulée : Core vs Satellite vs Portefeuille (base 100)")
    ax.set_xlabel("Date"); ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=10)
    _save(cfg.fig_dir, "C01_core_vs_sat_cum.png", cfg.dpi)


def plot_C02_excess_cum(r_port: pd.Series, r_core: pd.Series, cfg: PlotConfig) -> None:
    """C02 – Rendements excédentaires cumulés du portefeuille vs Core."""
    excess    = r_port - r_core
    cum_ex    = _cum(excess, base=0.0)   # base 0 : montant gagné au-dessus du core
    fig, ax = plt.subplots(figsize=(10, 4))
    cum_ex.mul(100).plot(ax=ax, color="#4363d8", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(cum_ex.index, cum_ex.mul(100), 0,
                    where=cum_ex > 0, alpha=0.2, color="#3cb44b")
    ax.fill_between(cum_ex.index, cum_ex.mul(100), 0,
                    where=cum_ex <= 0, alpha=0.2, color="#e6194b")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("C02 – Rendements excédentaires cumulés (portefeuille – core)")
    ax.set_xlabel("Date"); ax.set_ylabel("Excès cumulé (%)")
    _save(cfg.fig_dir, "C02_excess_cum.png", cfg.dpi)


def plot_C04_annual_grouped_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """C04 – Performance annuelle groupée : Portefeuille / Core / Satellite."""
    years  = annual.index.astype(str).tolist()
    x      = np.arange(len(years))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, annual["core"].mul(100),      width, label="Core",        color="#e6194b")
    ax.bar(x,         annual["portfolio"].mul(100),  width, label="Portefeuille", color="#4363d8")
    ax.bar(x + width, annual["satellite"].mul(100), width, label="Satellite",   color="#3cb44b")
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("C04 – Performance annuelle : Core vs Portefeuille vs Satellite")
    ax.set_ylabel("Rendement (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "C04_annual_grouped_bar.png", cfg.dpi)


def plot_C05_excess_annual_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """C05 – Rendement excédentaire annuel (portefeuille − core)."""
    exc = (annual["portfolio"] - annual["core"]).mul(100)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(exc.index.astype(str), exc,
                  color=["#3cb44b" if v >= 0 else "#e6194b" for v in exc])
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, v in zip(bars, exc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.1 if v >= 0 else -0.3),
                f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title("C05 – Rendement excédentaire annuel (portefeuille – core)")
    ax.set_ylabel("Excès (%)")
    _save(cfg.fig_dir, "C05_excess_annual.png", cfg.dpi)


# ══════════════════════════════════════════════════════════════════════════════
#  Section D — Portefeuille total
# ══════════════════════════════════════════════════════════════════════════════

def plot_D01_portfolio_cum(r_port: pd.Series, r_core: pd.Series, cfg: PlotConfig) -> None:
    """D01 – Perf cumulée du fonds (base 100) avec Core en référence."""
    fig, ax = plt.subplots(figsize=(11, 6))
    c_port = _cum(r_port)
    c_core = _cum(r_core)
    ax.fill_between(c_port.index, c_port, c_core, where=c_port >= c_core,
                    alpha=0.15, color="#3cb44b", label="_nolegend_")
    ax.fill_between(c_port.index, c_port, c_core, where=c_port < c_core,
                    alpha=0.15, color="#e6194b", label="_nolegend_")
    c_core.plot(ax=ax, color="#e6194b", lw=1.5, ls="--", label="Core (benchmark)")
    c_port.plot(ax=ax, color="#4363d8", lw=2.5, label="Portefeuille total")
    ax.set_title("D01 – Performance cumulée du fonds vs benchmark Core (base 100)")
    ax.set_xlabel("Date"); ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=10)
    _save(cfg.fig_dir, "D01_portfolio_cum.png", cfg.dpi)


def plot_D02_portfolio_drawdown(r_port: pd.Series, r_core: pd.Series,
                                 cfg: PlotConfig) -> None:
    """D02 – Drawdown du portefeuille total vs Core."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _dd(r_core).mul(100).plot(ax=ax, color="#e6194b", lw=1, ls="--", label="Core")
    _dd(r_port).mul(100).plot(ax=ax, color="#4363d8", lw=1.5, label="Portefeuille")
    ax.fill_between(_dd(r_port).index, _dd(r_port).mul(100), 0, alpha=0.2, color="#4363d8")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("D02 – Drawdown : Portefeuille vs Core")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D02_portfolio_drawdown.png", cfg.dpi)


def plot_D03_portfolio_annual_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """D03 – Performance annuelle du portefeuille total avec Core en référence."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(annual))
    ax.bar(x - 0.2, annual["core"].mul(100),      0.35, label="Core",        color="#e6194b", alpha=0.7)
    ax.bar(x + 0.2, annual["portfolio"].mul(100),  0.35, label="Portefeuille", color="#4363d8")
    ax.set_xticks(x); ax.set_xticklabels(annual.index.astype(str))
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("D03 – Performance annuelle : Portefeuille vs Core")
    ax.set_ylabel("Rendement (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D03_portfolio_annual.png", cfg.dpi)


def plot_D04_portfolio_vol_target(r_port: pd.Series, vol_min: float,
                                   vol_max: float, cfg: PlotConfig) -> None:
    """D04 – Volatilité rolling 63j vs bande cible."""
    roll_v = _roll_vol(r_port).mul(100)
    fig, ax = plt.subplots(figsize=(10, 4))
    roll_v.plot(ax=ax, color="#4363d8", lw=1.5, label="Vol rolling 63j")
    ax.axhline(vol_min * 100, color="orange", ls="--", lw=1, label=f"Min cible {vol_min:.0%}")
    ax.axhline(vol_max * 100, color="orange", ls="--", lw=1, label=f"Max cible {vol_max:.0%}")
    ax.fill_between(roll_v.index, vol_min * 100, vol_max * 100, alpha=0.1, color="orange")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("D04 – Volatilité rolling 63j vs bande cible")
    ax.set_xlabel("Date"); ax.set_ylabel("Vol annualisée (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D04_portfolio_vol_target.png", cfg.dpi)


def plot_D05_beta_sat_rolling(beta_roll: pd.Series, cfg: PlotConfig) -> None:
    """D05 – Beta rolling satellite vs Core (63j) avec bandes ±0.15."""
    fig, ax = plt.subplots(figsize=(10, 4))
    beta_roll.plot(ax=ax, color="#4363d8", lw=1.5, label="Beta rolling 63j")
    ax.axhline( 0,     color="black",  lw=0.8, ls="--")
    ax.axhline( 0.15,  color="orange", lw=1,   ls=":", label="±β_max = 0.15")
    ax.axhline(-0.15,  color="orange", lw=1,   ls=":")
    ax.fill_between(beta_roll.index, -0.15, 0.15, alpha=0.07, color="green",
                    label="Zone cible |β| ≤ 0.15")
    ax.set_title("D05 – Beta rolling 63j de la poche satellite vs Core")
    ax.set_xlabel("Date"); ax.set_ylabel("Beta")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D05_beta_sat_rolling.png", cfg.dpi)


def plot_D06_portfolio_dist(r_port: pd.Series, r_core: pd.Series, cfg: PlotConfig) -> None:
    """D06 – Distribution des rendements journaliers : Portefeuille vs Core."""
    fig, ax = plt.subplots(figsize=(9, 5))
    r_core.mul(100).hist(ax=ax, bins=80, alpha=0.5, color="#e6194b", label="Core", density=True)
    r_port.mul(100).hist(ax=ax, bins=80, alpha=0.5, color="#4363d8", label="Portefeuille", density=True)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("D06 – Distribution des rendements journaliers")
    ax.set_xlabel("Rendement journalier (%)"); ax.set_ylabel("Densité")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D06_portfolio_dist.png", cfg.dpi)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = PlotConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GÉNÉRATION DES GRAPHIQUES – RAPPORT CORE-SATELLITE")
    print("=" * 60)

    # ── Chargement des données ────────────────────────────────────────────────
    print("\n[1] Chargement des données...")

    # Core
    log_c_oos = pd.read_csv(cfg.core_oos_csv, index_col=0, parse_dates=True)
    log_c_oos.index = pd.DatetimeIndex(log_c_oos.index).tz_localize(None)
    r_core_oos = (np.exp(log_c_oos.iloc[:, 0]) - 1).sort_index()

    log_etf = pd.read_csv(cfg.core_etf_csv, index_col=0, parse_dates=True)
    log_etf.index = pd.DatetimeIndex(log_etf.index).tz_localize(None)

    core_finaux = pd.read_csv(cfg.core_finaux_csv)

    # Fonds total
    bt = pd.read_csv(cfg.fond_returns_csv, index_col=0, parse_dates=True)
    bt.index = pd.DatetimeIndex(bt.index).tz_localize(None)
    bt = bt.sort_index()
    r_port = bt["portfolio_ret"]
    r_core = bt["core_ret"]
    r_sat  = bt["sat_pocket_ret"]

    # Poids
    weights_df = pd.read_csv(cfg.fond_weights_csv, index_col=0)
    sat_weights = weights_df["theta_satellite"].dropna()
    sat_weights = sat_weights[sat_weights > 0]
    w_sat = float(weights_df["w_sat"].iloc[0])

    # Métriques
    metrics = pd.read_csv(cfg.fond_metrics_csv, index_col=0)["valeur"].to_dict()

    # Annuel
    annual = pd.read_csv(cfg.fond_annual_csv, index_col=0)
    annual.index = annual.index.astype(int)

    # Beta rolling
    beta_roll = pd.read_csv(cfg.fond_beta_roll_csv, index_col=0, parse_dates=True)
    beta_roll.index = pd.DatetimeIndex(beta_roll.index).tz_localize(None)
    beta_series = beta_roll.iloc[:, 0].sort_index()

    # Satellite
    sat_info = pd.read_csv(cfg.sat_selected_csv).set_index("ticker")
    sat_indiv = pd.read_csv(cfg.sat_indiv_csv, index_col=0, parse_dates=True)
    sat_indiv.index = pd.DatetimeIndex(sat_indiv.index).tz_localize(None)
    sat_indiv = sat_indiv.sort_index()

    vol_min = 0.04; vol_max = 0.15

    print(f"    Backtest : {bt.index.min().date()} → {bt.index.max().date()} ({len(bt)} obs)")

    # ── Section A — Core ────────────────────────────────────────────────────────
    print("\n[2] Section A – Poche Core (7 figures)...")
    plot_A01_core_etf_cum(log_etf, core_finaux, cfg)
    plot_A02_core_portfolio_cum(r_core_oos.loc[r_port.index.min():], cfg)
    plot_A03_core_drawdown(r_core, cfg)
    plot_A04_core_rolling_vol(r_core, cfg)
    plot_A05_core_rolling_sharpe(r_core, cfg)
    plot_A06_core_annual_bar(annual, cfg)
    plot_A07_core_fees_bar(core_finaux, cfg)

    # ── Section B — Satellite ────────────────────────────────────────────────────
    print("\n[3] Section B – Poche Satellite (7 figures)...")
    plot_B01_sat_fund_cum(sat_indiv, sat_info, sat_weights, cfg)
    plot_B02_sat_pocket_cum(r_sat, cfg)
    plot_B03_sat_weights_bar(sat_weights, sat_info, w_sat, cfg)
    plot_B04_sat_rolling_alpha(r_sat, r_core, cfg)
    plot_B05_sat_rolling_beta(beta_series, cfg)
    plot_B06_sat_perf_annual(annual, cfg)
    plot_B07_sat_fees_bar(sat_weights, sat_info, cfg)

    # ── Section C — Core vs Satellite ────────────────────────────────────────────
    print("\n[4] Section C – Core vs Satellite (4 figures)...")
    plot_C01_core_vs_sat_cum(r_core, r_sat, r_port, cfg)
    plot_C02_excess_cum(r_port, r_core, cfg)
    plot_C04_annual_grouped_bar(annual, cfg)
    plot_C05_excess_annual_bar(annual, cfg)

    # ── Section D — Portefeuille total ───────────────────────────────────────────
    print("\n[5] Section D – Portefeuille total (6 figures)...")
    plot_D01_portfolio_cum(r_port, r_core, cfg)
    plot_D02_portfolio_drawdown(r_port, r_core, cfg)
    plot_D03_portfolio_annual_bar(annual, cfg)
    plot_D04_portfolio_vol_target(r_port, vol_min, vol_max, cfg)
    plot_D05_beta_sat_rolling(beta_series, cfg)
    plot_D06_portfolio_dist(r_port, r_core, cfg)

    print(f"\n  ✓  24 figures générées dans {cfg.fig_dir}")


if __name__ == "__main__":
    main()
