"""
Analyse de robustesse par sous-périodes du portefeuille Core/Satellite.

Objectif :
- Vérifier que la performance n'est pas concentrée sur un seul régime
- Comparer avant/après 2020 (changement de régime : Covid, inflation, or)
- Identifier les périodes de stress et les contributions Core vs Satellite

Entrées (même dossier) :
- core_returns_monthly.csv
- satellite_returns_monthly.csv
- portfolio_returns_monthly.csv
- portfolio_weights_monthly.csv

Sorties :
- outputs/robustness_subperiods.csv
- outputs/robustness_yearly.csv
- outputs/robustness_regimes.csv
- outputs/figures/robustness_*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.portfolio_engine import (
    lire_serie_returns,
    lire_core_policy,
    aligner_series,
    metriques_base,
)


@dataclass(frozen=True)
class RobustnessConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    input_dir: Path = project_root / "outputs"
    output_dir: Path = Path(__file__).resolve().parent / "outputs" / "robustness"
    fig_dir: Path = output_dir / "figures"

    core_csv: Path = input_dir / "core3_etf_returns_monthly.csv"
    sat_csv: Path = input_dir / "satellite_returns_monthly.csv"
    portfolio_csv: Path = input_dir / "portfolio_returns_monthly.csv"
    weights_csv: Path = input_dir / "portfolio_weights_monthly.csv"

    # Policy weights Core (cohérent avec portfolio_engine.py)
    policy_w_equity: float = 0.60
    policy_w_rates: float = 0.20
    policy_w_credit: float = 0.20

    split_date: str = "2020-01-01"

    stress_periods: Tuple[Tuple[str, str, str], ...] = (
        ("Taper Tantrum", "2013-05-01", "2013-09-30"),
        ("Chine / Pétrole 2015-16", "2015-08-01", "2016-02-29"),
        ("Vol Spike Q4 2018", "2018-10-01", "2018-12-31"),
        ("Covid Crash", "2020-02-01", "2020-04-30"),
        ("Hausse taux 2022", "2022-01-01", "2022-10-31"),
    )

    w_sat: float = 0.30
    dpi: int = 160


def metriques_periode(returns: pd.Series, label: str) -> Dict[str, float]:
    """Calcule les métriques sur une sous-période donnée."""
    if len(returns) < 2:
        return {"Période": label, "Début": "", "Fin": "", "Nb mois": 0,
                "Rendement annualisé": np.nan, "Volatilité annualisée": np.nan,
                "Sharpe (rf=0)": np.nan, "Max Drawdown": np.nan, "Calmar": np.nan}

    mets = metriques_base(returns)
    mets["Période"] = label
    mets["Début"] = str(returns.index.min().date())
    mets["Fin"] = str(returns.index.max().date())
    mets["Nb mois"] = len(returns)
    return mets


# ═══════════════════════════════════════════════════════════════════════════════
# 1) Sous-périodes avant / après
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_avant_apres(port: pd.Series, core: pd.Series, sat: pd.Series, split_date: str) -> pd.DataFrame:
    print(f"\n[ROBUSTESSE 1/4] Sous-périodes (avant / après {split_date})")
    split = pd.Timestamp(split_date)
    rows = []
    for series, name in [(port, "Portefeuille"), (core, "Core"), (sat, "Satellite")]:
        s = series.dropna()
        rows.append(metriques_periode(s, f"{name} — Pleine période"))
        rows.append(metriques_periode(s[s.index < split], f"{name} — Avant {split_date[:7]}"))
        rows.append(metriques_periode(s[s.index >= split], f"{name} — Après {split_date[:7]}"))

    df = pd.DataFrame(rows)
    cols_first = ["Période", "Début", "Fin", "Nb mois"]
    df = df[cols_first + [c for c in df.columns if c not in cols_first]]

    for _, row in df.iterrows():
        print(f"  {row['Période']:45s}  Sharpe={row['Sharpe (rf=0)']:+.3f}  "
              f"Rdt={row['Rendement annualisé']:+.2%}  MaxDD={row['Max Drawdown']:+.2%}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2) Rendements annuels
# ═══════════════════════════════════════════════════════════════════════════════

def rendements_annuels(port: pd.Series, core: pd.Series, sat: pd.Series) -> pd.DataFrame:
    print("\n[ROBUSTESSE 2/4] Rendements annuels (calendar year)")
    rows = []
    for year in sorted(port.index.year.unique()):
        p = port[port.index.year == year]
        c = core[core.index.year == year]
        s = sat[sat.index.year == year]
        rdt_port = float((1 + p).prod() - 1) if len(p) > 0 else np.nan
        rdt_core = float((1 + c).prod() - 1) if len(c) > 0 else np.nan
        rdt_sat = float((1 + s).prod() - 1) if len(s) > 0 else np.nan
        exc = rdt_port - rdt_core if not (np.isnan(rdt_port) or np.isnan(rdt_core)) else np.nan
        rows.append({"Année": year, "Portefeuille": rdt_port, "Core": rdt_core,
                      "Satellite": rdt_sat, "Excès (Port - Core)": exc})
        if not np.isnan(exc):
            print(f"  {year}  Port={rdt_port:+.2%}  Core={rdt_core:+.2%}  "
                  f"Sat={rdt_sat:+.2%}  Excès={exc:+.2%}")
    return pd.DataFrame(rows).set_index("Année")


# ═══════════════════════════════════════════════════════════════════════════════
# 3) Périodes de stress
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_stress(port: pd.Series, core: pd.Series, sat: pd.Series,
                   stress_periods: Tuple[Tuple[str, str, str], ...]) -> pd.DataFrame:
    print("\n[ROBUSTESSE 3/4] Périodes de stress")
    rows = []
    for label, start, end in stress_periods:
        rdt_p = float((1 + port[(port.index >= start) & (port.index <= end)]).prod() - 1)
        rdt_c = float((1 + core[(core.index >= start) & (core.index <= end)]).prod() - 1)
        rdt_s = float((1 + sat[(sat.index >= start) & (sat.index <= end)]).prod() - 1)
        prot = rdt_p - rdt_c
        rows.append({"Stress": label, "Début": start, "Fin": end,
                      "Rdt Portefeuille": rdt_p, "Rdt Core": rdt_c, "Rdt Satellite": rdt_s,
                      "Protection (Port - Core)": prot})
        print(f"  {label:30s}  Port={rdt_p:+.2%}  Core={rdt_c:+.2%}  "
              f"Sat={rdt_s:+.2%}  Protection={prot:+.2%}")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4) Rolling Sharpe
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_sharpe(returns: pd.Series, window: int = 24) -> pd.Series:
    roll_mean = returns.rolling(window).mean() * 12
    roll_vol = returns.rolling(window).std() * np.sqrt(12)
    return (roll_mean / roll_vol).dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# Graphiques
# ═══════════════════════════════════════════════════════════════════════════════

def plot_subperiods_bar(df_sub: pd.DataFrame, cfg: RobustnessConfig) -> None:
    mask = df_sub["Période"].str.contains("Portefeuille")
    sub = df_sub[mask].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = sub["Période"].str.replace("Portefeuille — ", "", regex=False).tolist()
    x = range(len(labels))
    colors = ["#3B82F6", "#60A5FA", "#93C5FD"][:len(labels)]

    vals = sub["Sharpe (rf=0)"].values
    axes[0].bar(x, vals, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("Sharpe (rf=0)"); axes[0].set_title("Sharpe par sous-période")
    for i, v in enumerate(vals): axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    vals_r = sub["Rendement annualisé"].values * 100
    axes[1].bar(x, vals_r, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("Rendement annualisé (%)"); axes[1].set_title("Rendement par sous-période")
    for i, v in enumerate(vals_r): axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

    fig.suptitle("Robustesse : sous-périodes (Portefeuille)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "robustness_subperiods_bar.png", dpi=cfg.dpi); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'robustness_subperiods_bar.png'}")


def plot_yearly(df_yearly: pd.DataFrame, cfg: RobustnessConfig) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    years = df_yearly.index.values; x = np.arange(len(years)); width = 0.35
    ax.bar(x - width / 2, df_yearly["Portefeuille"].values * 100, width, label="Portefeuille", color="#2563EB", edgecolor="white")
    ax.bar(x + width / 2, df_yearly["Core"].values * 100, width, label="Core", color="#93C5FD", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Rendement annuel (%)"); ax.set_title("Rendements annuels : Portefeuille vs Core", fontsize=13, fontweight="bold")
    ax.legend(); ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "robustness_yearly.png", dpi=cfg.dpi); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'robustness_yearly.png'}")


def plot_rolling_sharpe(port: pd.Series, core: pd.Series, cfg: RobustnessConfig) -> None:
    rs_port = rolling_sharpe(port, 24); rs_core = rolling_sharpe(core, 24)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rs_port.index, rs_port.values, label="Portefeuille", color="#2563EB", linewidth=1.5)
    ax.plot(rs_core.index, rs_core.values, label="Core", color="#93C5FD", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(pd.Timestamp(cfg.split_date), linestyle="--", color="red", alpha=0.5, label=f"Coupure ({cfg.split_date[:7]})")
    ax.set_xlabel("Date"); ax.set_ylabel("Sharpe rolling (24 mois)")
    ax.set_title("Sharpe rolling 24 mois : Portefeuille vs Core", fontsize=13, fontweight="bold"); ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "robustness_rolling_sharpe.png", dpi=cfg.dpi); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'robustness_rolling_sharpe.png'}")


def plot_contribution(core: pd.Series, sat: pd.Series, cfg: RobustnessConfig) -> None:
    """
    Décomposition de la performance cumulée : Core vs Satellite.

    Méthode : contributions composées (Menchero).
    À chaque date T, la contribution composée du composant i vaut :
        CC_i(T) = CC_i(T-1) * (1 + r_port_T) + w_i * r_i_T
    Propriété : CC_core(T) + CC_sat(T) = (1+r_port)_cumprod - 1 (exact).
    """
    w_core = 1.0 - cfg.w_sat
    r_port = w_core * core + cfg.w_sat * sat

    cc_core = np.zeros(len(core))
    cc_sat = np.zeros(len(sat))

    for t in range(len(core)):
        if t == 0:
            cc_core[t] = w_core * core.iloc[t]
            cc_sat[t] = cfg.w_sat * sat.iloc[t]
        else:
            cc_core[t] = cc_core[t - 1] * (1.0 + r_port.iloc[t]) + w_core * core.iloc[t]
            cc_sat[t] = cc_sat[t - 1] * (1.0 + r_port.iloc[t]) + cfg.w_sat * sat.iloc[t]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(core.index, 0, cc_core * 100, alpha=0.5, label="Core (70%)", color="#3B82F6")
    ax.fill_between(core.index, cc_core * 100, (cc_core + cc_sat) * 100, alpha=0.5, label="Satellite (30%)", color="#F59E0B")
    ax.axvline(pd.Timestamp(cfg.split_date), linestyle="--", color="red", alpha=0.5, label=f"Coupure ({cfg.split_date[:7]})")
    ax.set_xlabel("Date"); ax.set_ylabel("Contribution cumulée (%)")
    ax.set_title("Décomposition de la performance : Core vs Satellite", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "robustness_contribution.png", dpi=cfg.dpi); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'robustness_contribution.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = RobustnessConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANALYSE DE ROBUSTESSE PAR SOUS-PÉRIODES")
    print("=" * 60)

    core = lire_core_policy(cfg.core_csv, cfg.policy_w_equity, cfg.policy_w_rates, cfg.policy_w_credit)
    sat = lire_serie_returns(cfg.sat_csv, "satellite_portfolio_return")
    port = lire_serie_returns(cfg.portfolio_csv, "portfolio_return")
    print(f"Core mode: policy ({cfg.policy_w_equity:.0%}/{cfg.policy_w_rates:.0%}/{cfg.policy_w_credit:.0%})")

    core_a, sat_a = aligner_series(core, sat)
    idx_common = core_a.index.intersection(sat_a.index).intersection(port.index)
    core_c = core_a.reindex(idx_common).dropna()
    sat_c = sat_a.reindex(idx_common).dropna()
    port_c = port.reindex(idx_common).dropna()
    print(f"Période commune: {idx_common.min().date()} -> {idx_common.max().date()} ({len(idx_common)} mois)")

    df_sub = analyse_avant_apres(port_c, core_c, sat_c, cfg.split_date)
    df_sub.to_csv(cfg.output_dir / "robustness_subperiods.csv", index=False)

    df_yearly = rendements_annuels(port_c, core_c, sat_c)
    df_yearly.to_csv(cfg.output_dir / "robustness_yearly.csv")

    df_stress = analyse_stress(port_c, core_c, sat_c, cfg.stress_periods)
    df_stress.to_csv(cfg.output_dir / "robustness_regimes.csv", index=False)

    print("\n[ROBUSTESSE 4/4] Génération des graphiques...")
    plot_subperiods_bar(df_sub, cfg)
    plot_yearly(df_yearly, cfg)
    plot_rolling_sharpe(port_c, core_c, cfg)
    plot_contribution(core_c, sat_c, cfg)

    print("\n" + "=" * 60)
    print("RÉSUMÉ ROBUSTESSE")
    print("=" * 60)
    mask_port = df_sub["Période"].str.contains("Portefeuille")
    for _, row in df_sub[mask_port].iterrows():
        print(f"  {row['Période']:45s}  Sharpe={row['Sharpe (rf=0)']:.3f}  Rdt={row['Rendement annualisé']:.2%}")

    neg_years = df_yearly[df_yearly["Portefeuille"] < 0]
    print(f"\nAnnées négatives: {len(neg_years)}/{len(df_yearly)}")
    for year, row in neg_years.iterrows():
        print(f"  {year}: {row['Portefeuille']:+.2%}")

    print(f"\nProtection moyenne en stress (Port - Core): {df_stress['Protection (Port - Core)'].mean():+.2%}")
    print("\nTerminé.")


if __name__ == "__main__":
    main()