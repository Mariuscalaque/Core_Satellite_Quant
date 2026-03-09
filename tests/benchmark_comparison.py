"""
Comparaison du portefeuille Core/Satellite avec des benchmarks classiques.

Benchmarks construits :
1. Core seul (passif) : sans Satellite ni vol targeting
2. Core/Sat sans VolTarget : 70/30 sans scaling
3. Core seul + VolTarget : isole l'effet Satellite
4. Equal Weight 50/50 : 50% Core + 50% Satellite sans vol targeting

Entrées (même dossier) :
- core_returns_monthly.csv
- satellite_returns_monthly.csv
- portfolio_returns_monthly.csv

Sorties :
- outputs/benchmark_comparison.csv
- outputs/benchmark_yearly.csv
- outputs/figures/benchmark_*.png
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
    appliquer_vol_targeting,
    metriques_base,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    project_root: Path = Path(__file__).resolve().parent.parent   # Core_Satellite_Quant/
    input_dir: Path = project_root / "outputs"                    # lecture CSV
    output_dir: Path = Path(__file__).resolve().parent / "outputs" / "benchmark"  # Tests/outputs/benchmark/
    fig_dir: Path = output_dir / "figures"                        # Tests/outputs/benchmark/figures/

    core_csv: Path = input_dir / "core3_etf_returns_monthly.csv"
    sat_csv: Path = input_dir / "satellite_returns_monthly.csv"
    portfolio_csv: Path = input_dir / "portfolio_returns_monthly.csv"

    # Policy weights Core (cohérent avec portfolio_engine.py)
    policy_w_equity: float = 0.60
    policy_w_rates: float = 0.20
    policy_w_credit: float = 0.20

    dpi: int = 160

# ═══════════════════════════════════════════════════════════════════════════════
# Construction des benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def construire_benchmark_core_passif(core: pd.Series) -> pd.Series:
    """Core seul, sans Satellite ni vol targeting (baseline passive)."""
    return core.copy().rename("benchmark_core_passif")


def construire_benchmark_core_sat_sans_voltarget(core: pd.Series, sat: pd.Series, w_sat: float = 0.30) -> pd.Series:
    """Core/Satellite SANS vol targeting. Isole la contribution du vol targeting."""
    return ((1.0 - w_sat) * core + w_sat * sat).rename("benchmark_sans_voltarget")


def construire_benchmark_core_only_scaled(core: pd.Series, target_vol: float = 0.10) -> pd.Series:
    """Core seul AVEC vol targeting. Isole la contribution du Satellite."""
    port, _ = appliquer_vol_targeting(
        returns_brut=core, target_vol=target_vol,
        lookback_months=12, scale_min=0.50, scale_max=2.0,
    )
    return port.rename("benchmark_core_voltarget")


def construire_benchmark_equal_weight(core: pd.Series, sat: pd.Series) -> pd.Series:
    """50% Core + 50% Satellite, sans vol targeting."""
    return (0.50 * core + 0.50 * sat).rename("benchmark_equal_weight")


# ═══════════════════════════════════════════════════════════════════════════════
# Métriques comparatives
# ═══════════════════════════════════════════════════════════════════════════════

def comparer_strategies(strategies: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    common_idx = None
    for s in strategies.values():
        idx = s.dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    rows = []
    for name, s in strategies.items():
        r = s.reindex(common_idx).dropna()
        mets = metriques_base(r)
        mets["Stratégie"] = name
        mets["Rendement cumulé total"] = float((1 + r).prod() - 1)
        mets["% mois positifs"] = float((r > 0).mean())
        mets["Skewness"] = float(r.skew())
        rows.append(mets)

    return pd.DataFrame(rows).set_index("Stratégie"), common_idx


def rendements_annuels_multi(strategies: Dict[str, pd.Series], common_idx: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for year in sorted(common_idx.year.unique()):
        row = {"Année": year}
        for name, s in strategies.items():
            r = s.reindex(common_idx)
            ry = r[r.index.year == year]
            row[name] = float((1 + ry).prod() - 1) if len(ry) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Année")


# ═══════════════════════════════════════════════════════════════════════════════
# Graphiques
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Portefeuille (Core/Sat + VolTarget)": "#2563EB",
    "Core seul (passif)": "#93C5FD",
    "Core/Sat sans VolTarget": "#F59E0B",
    "Core seul + VolTarget": "#10B981",
    "Equal Weight 50/50": "#8B5CF6",
}


def plot_cumulative(strategies: Dict[str, pd.Series], common_idx, cfg: BenchmarkConfig) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, s in strategies.items():
        r = s.reindex(common_idx).dropna()
        cum = 100 * (1 + r).cumprod()
        color = COLORS.get(name, "#666666")
        lw = 2.5 if "Portefeuille" in name else 1.5
        ls = "-" if "Portefeuille" in name or "VolTarget" in name else "--"
        ax.plot(cum.index, cum.values, label=name, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlabel("Date"); ax.set_ylabel("Valeur (base 100)")
    ax.set_title("Performance cumulée : Portefeuille vs Benchmarks", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "benchmark_cumulative.png", dpi=cfg.dpi); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'benchmark_cumulative.png'}")


def plot_drawdown_compare(strategies: Dict[str, pd.Series], common_idx, cfg: BenchmarkConfig) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, s in strategies.items():
        r = s.reindex(common_idx).dropna()
        cum = (1 + r).cumprod(); dd = cum / cum.cummax() - 1
        color = COLORS.get(name, "#666666")
        lw = 2 if "Portefeuille" in name else 1.2
        ax.plot(dd.index, dd.values * 100, label=name, color=color, linewidth=lw)
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown comparé : Portefeuille vs Benchmarks", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "benchmark_drawdown.png", dpi=cfg.dpi); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'benchmark_drawdown.png'}")


def plot_yearly_bar(df_yearly: pd.DataFrame, cfg: BenchmarkConfig) -> None:
    cols = [c for c in ["Portefeuille (Core/Sat + VolTarget)", "Core seul (passif)", "Core/Sat sans VolTarget"]
            if c in df_yearly.columns]
    if not cols:
        cols = df_yearly.columns[:3].tolist()

    fig, ax = plt.subplots(figsize=(14, 5))
    years = df_yearly.index.values; x = np.arange(len(years)); width = 0.8 / len(cols)
    for i, col in enumerate(cols):
        offset = (i - len(cols) / 2 + 0.5) * width
        color = COLORS.get(col, f"C{i}")
        ax.bar(x + offset, df_yearly[col].values * 100, width, label=col, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Rendement annuel (%)"); ax.set_title("Rendements annuels comparés", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "benchmark_yearly_bar.png", dpi=cfg.dpi); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'benchmark_yearly_bar.png'}")


def plot_radar(df_comp: pd.DataFrame, cfg: BenchmarkConfig) -> None:
    metrics_to_plot = ["Rendement annualisé", "Sharpe (rf=0)", "Calmar", "% mois positifs"]
    df_plot = df_comp[metrics_to_plot].copy()
    df_plot["Contrôle DD"] = -df_comp["Max Drawdown"]
    labels = list(df_plot.columns)
    df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for strat in [s for s in df_norm.index if s in COLORS][:4]:
        vals = df_norm.loc[strat].values.tolist() + [df_norm.loc[strat].values[0]]
        color = COLORS.get(strat, "#666666")
        ax.plot(angles, vals, "o-", linewidth=2, label=strat, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Comparaison multi-critères (normalisé)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "benchmark_radar.png", dpi=cfg.dpi, bbox_inches="tight"); plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'benchmark_radar.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = BenchmarkConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPARAISON AVEC BENCHMARKS")
    print("=" * 60)

    core = lire_core_policy(cfg.core_csv, cfg.policy_w_equity, cfg.policy_w_rates, cfg.policy_w_credit)
    sat = lire_serie_returns(cfg.sat_csv, "satellite_portfolio_return")
    port = lire_serie_returns(cfg.portfolio_csv, "portfolio_return")
    core_a, sat_a = aligner_series(core, sat)
    print(f"Core mode: policy ({cfg.policy_w_equity:.0%}/{cfg.policy_w_rates:.0%}/{cfg.policy_w_credit:.0%})")

    print("\n[1/4] Construction des benchmarks...")
    strategies = {
        "Portefeuille (Core/Sat + VolTarget)": port,
        "Core seul (passif)": construire_benchmark_core_passif(core_a),
        "Core/Sat sans VolTarget": construire_benchmark_core_sat_sans_voltarget(core_a, sat_a, w_sat=0.30),
        "Core seul + VolTarget": construire_benchmark_core_only_scaled(core_a, target_vol=0.10),
        "Equal Weight 50/50": construire_benchmark_equal_weight(core_a, sat_a),
    }
    for name, s in strategies.items():
        print(f"    {name}: {s.index.min().date()} -> {s.index.max().date()} ({len(s)} mois)")

    print("\n[2/4] Calcul des métriques comparatives...")
    df_comp, common_idx = comparer_strategies(strategies)
    print(f"\nPériode commune: {common_idx.min().date()} -> {common_idx.max().date()} ({len(common_idx)} mois)\n")
    for strat in df_comp.index:
        row = df_comp.loc[strat]
        print(f"  {strat}")
        print(f"    Rdt={row['Rendement annualisé']:.2%}  Vol={row['Volatilité annualisée']:.2%}  "
              f"Sharpe={row['Sharpe (rf=0)']:.3f}  MaxDD={row['Max Drawdown']:.2%}  Calmar={row['Calmar']:.3f}")
    df_comp.to_csv(cfg.output_dir / "benchmark_comparison.csv")

    print("\n[3/4] Rendements annuels...")
    df_yearly = rendements_annuels_multi(strategies, common_idx)
    df_yearly.to_csv(cfg.output_dir / "benchmark_yearly.csv")

    print("\n[4/4] Génération des graphiques...")
    plot_cumulative(strategies, common_idx, cfg)
    plot_drawdown_compare(strategies, common_idx, cfg)
    plot_yearly_bar(df_yearly, cfg)
    plot_radar(df_comp, cfg)

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ : VALEUR AJOUTÉE DU PORTEFEUILLE")
    print("=" * 60)

    port_sharpe = df_comp.loc["Portefeuille (Core/Sat + VolTarget)", "Sharpe (rf=0)"]
    core_sharpe = df_comp.loc["Core seul (passif)", "Sharpe (rf=0)"]
    brut_sharpe = df_comp.loc["Core/Sat sans VolTarget", "Sharpe (rf=0)"]
    core_vt_sharpe = df_comp.loc["Core seul + VolTarget", "Sharpe (rf=0)"]

    print(f"\n  Sharpe portefeuille final          : {port_sharpe:.3f}")
    print(f"  Sharpe Core seul (passif)          : {core_sharpe:.3f}")
    print(f"  Sharpe Core/Sat sans vol targeting  : {brut_sharpe:.3f}")
    print(f"  Sharpe Core seul + vol targeting    : {core_vt_sharpe:.3f}")

    delta_sat = brut_sharpe - core_sharpe
    delta_vt = port_sharpe - brut_sharpe
    delta_total = port_sharpe - core_sharpe

    print(f"\n  Contribution Satellite (brut)      : {delta_sat:+.3f} Sharpe")
    print(f"  Contribution Vol Targeting          : {delta_vt:+.3f} Sharpe")
    print(f"  Amélioration totale vs Core passif  : {delta_total:+.3f} Sharpe")
    print("\nTerminé.")


if __name__ == "__main__":
    main()