"""
Analyse de sensibilité du portefeuille Core/Satellite.

Teste l'impact de 3 paramètres clés sur les métriques du portefeuille :
1. Bornes de poids Satellite (w_sat)
2. Plafond du scale (vol targeting)
3. Fenêtre d'estimation de la volatilité (lookback)

Réutilise les fonctions de portfolio_engine.py pour garantir la cohérence.

Entrées (même dossier) :
- core_returns_monthly.csv
- satellite_returns_monthly.csv

Sorties :
- outputs/sensitivity_w_sat.csv
- outputs/sensitivity_scale_cap.csv
- outputs/sensitivity_lookback.csv
- outputs/figures/sensitivity_*.png
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
class SensitivityConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    input_dir: Path = project_root / "outputs"
    output_dir: Path = Path(__file__).resolve().parent / "outputs" / "sensitivity"
    fig_dir: Path = output_dir / "figures"

    core_csv: Path = input_dir / "core3_etf_returns_monthly.csv"
    sat_csv: Path = input_dir / "satellite_returns_monthly.csv"

    # Policy weights Core (cohérent avec portfolio_engine.py)
    policy_w_equity: float = 0.60
    policy_w_rates: float = 0.20
    policy_w_credit: float = 0.20

    # Grilles de test
    w_sat_grid: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    scale_max_grid: Tuple[float, ...] = (1.0, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0)
    lookback_grid: Tuple[int, ...] = (6, 9, 12, 18, 24, 36)

    # Paramètres de référence (baseline)
    baseline_w_sat: float = 0.30
    baseline_target_vol: float = 0.10
    baseline_lookback: int = 12
    baseline_scale_min: float = 0.50
    baseline_scale_max: float = 2.0

    dpi: int = 160


def run_portfolio(
    core: pd.Series,
    sat: pd.Series,
    w_sat: float,
    target_vol: float,
    lookback: int,
    scale_min: float,
    scale_max: float,
) -> Dict[str, float]:
    """Construit le portefeuille et renvoie les métriques.

    Note : on construit le portefeuille brut directement ici (sans passer
    par construire_portefeuille_brut qui impose un cap à 30%) afin de
    tester librement l'impact de w_sat au-delà de la contrainte de production.
    """
    w_core = 1.0 - w_sat
    brut = w_core * core + w_sat * sat
    port, scale_lag = appliquer_vol_targeting(
        returns_brut=brut,
        target_vol=target_vol,
        lookback_months=lookback,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    if len(port) < 24:
        return {
            "Rendement annualisé": np.nan,
            "Volatilité annualisée": np.nan,
            "Sharpe (rf=0)": np.nan,
            "Max Drawdown": np.nan,
            "Calmar": np.nan,
            "Nombre de mois": float(len(port)),
            "Scale moyen": np.nan,
        }
    mets = metriques_base(port)
    mets["Scale moyen"] = float(scale_lag.mean())
    return mets


# ═══════════════════════════════════════════════════════════════════════════════
# 1) Sensibilité w_sat
# ═══════════════════════════════════════════════════════════════════════════════

def sensibilite_w_sat(core: pd.Series, sat: pd.Series, cfg: SensitivityConfig) -> pd.DataFrame:
    print("\n[SENSIBILITÉ 1/3] Poids Satellite (w_sat)")
    print(f"  Grille: {cfg.w_sat_grid}")

    rows = []
    for w in cfg.w_sat_grid:
        mets = run_portfolio(core, sat, w_sat=w, target_vol=cfg.baseline_target_vol,
                             lookback=cfg.baseline_lookback, scale_min=cfg.baseline_scale_min,
                             scale_max=cfg.baseline_scale_max)
        mets["w_sat"] = w
        rows.append(mets)
        print(f"  w_sat={w:.0%} -> Sharpe={mets['Sharpe (rf=0)']:.3f}  "
              f"Vol={mets['Volatilité annualisée']:.2%}  MaxDD={mets['Max Drawdown']:.2%}")

    df = pd.DataFrame(rows).set_index("w_sat")
    df.to_csv(cfg.output_dir / "sensitivity_w_sat.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2) Sensibilité scale_max
# ═══════════════════════════════════════════════════════════════════════════════

def sensibilite_scale_cap(core: pd.Series, sat: pd.Series, cfg: SensitivityConfig) -> pd.DataFrame:
    print("\n[SENSIBILITÉ 2/3] Plafond de scale (scale_max)")
    print(f"  Grille: {cfg.scale_max_grid}")

    rows = []
    for s_max in cfg.scale_max_grid:
        mets = run_portfolio(core, sat, w_sat=cfg.baseline_w_sat, target_vol=cfg.baseline_target_vol,
                             lookback=cfg.baseline_lookback, scale_min=cfg.baseline_scale_min,
                             scale_max=s_max)
        mets["scale_max"] = s_max
        rows.append(mets)
        print(f"  scale_max={s_max:.2f} -> Sharpe={mets['Sharpe (rf=0)']:.3f}  "
              f"Vol={mets['Volatilité annualisée']:.2%}  Scale moy={mets['Scale moyen']:.2f}")

    df = pd.DataFrame(rows).set_index("scale_max")
    df.to_csv(cfg.output_dir / "sensitivity_scale_cap.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3) Sensibilité lookback
# ═══════════════════════════════════════════════════════════════════════════════

def sensibilite_lookback(core: pd.Series, sat: pd.Series, cfg: SensitivityConfig) -> pd.DataFrame:
    print("\n[SENSIBILITÉ 3/3] Fenêtre lookback (vol targeting)")
    print(f"  Grille: {cfg.lookback_grid}")

    rows = []
    for lb in cfg.lookback_grid:
        mets = run_portfolio(core, sat, w_sat=cfg.baseline_w_sat, target_vol=cfg.baseline_target_vol,
                             lookback=lb, scale_min=cfg.baseline_scale_min,
                             scale_max=cfg.baseline_scale_max)
        mets["lookback_months"] = lb
        rows.append(mets)
        print(f"  lookback={lb} mois -> Sharpe={mets['Sharpe (rf=0)']:.3f}  "
              f"Vol={mets['Volatilité annualisée']:.2%}  MaxDD={mets['Max Drawdown']:.2%}")

    df = pd.DataFrame(rows).set_index("lookback_months")
    df.to_csv(cfg.output_dir / "sensitivity_lookback.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Graphiques
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_w_sat(df: pd.DataFrame, cfg: SensitivityConfig) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df.index * 100, df["Sharpe (rf=0)"], "o-", color="#2563EB", linewidth=2, label="Sharpe")
    ax1.set_xlabel("Poids Satellite (%)")
    ax1.set_ylabel("Sharpe (rf=0)", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.axvline(cfg.baseline_w_sat * 100, linestyle="--", color="grey", alpha=0.6, label="Baseline (30%)")

    ax2 = ax1.twinx()
    ax2.plot(df.index * 100, df["Max Drawdown"] * 100, "s--", color="#DC2626", linewidth=2, label="Max Drawdown")
    ax2.set_ylabel("Max Drawdown (%)", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")

    fig.suptitle("Sensibilité au poids Satellite", fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "sensitivity_w_sat.png", dpi=cfg.dpi)
    plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'sensitivity_w_sat.png'}")


def plot_sensitivity_scale_cap(df: pd.DataFrame, cfg: SensitivityConfig) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(df.index, df["Sharpe (rf=0)"], "o-", color="#2563EB", linewidth=2)
    axes[0].axvline(cfg.baseline_scale_max, linestyle="--", color="grey", alpha=0.6)
    axes[0].set_xlabel("Scale max"); axes[0].set_ylabel("Sharpe"); axes[0].set_title("Sharpe vs Scale max")

    axes[1].plot(df.index, df["Volatilité annualisée"] * 100, "o-", color="#059669", linewidth=2)
    axes[1].axhline(10, linestyle=":", color="red", alpha=0.5, label="Cible 10%")
    axes[1].axvline(cfg.baseline_scale_max, linestyle="--", color="grey", alpha=0.6)
    axes[1].set_xlabel("Scale max"); axes[1].set_ylabel("Vol ann. (%)"); axes[1].set_title("Volatilité vs Scale max")
    axes[1].legend()

    axes[2].plot(df.index, df["Rendement annualisé"] * 100, "o-", color="#D97706", linewidth=2)
    axes[2].axvline(cfg.baseline_scale_max, linestyle="--", color="grey", alpha=0.6)
    axes[2].set_xlabel("Scale max"); axes[2].set_ylabel("Rdt ann. (%)"); axes[2].set_title("Rendement vs Scale max")

    fig.suptitle("Sensibilité au plafond de scale (vol targeting)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "sensitivity_scale_cap.png", dpi=cfg.dpi)
    plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'sensitivity_scale_cap.png'}")


def plot_sensitivity_lookback(df: pd.DataFrame, cfg: SensitivityConfig) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df.index, df["Sharpe (rf=0)"], "o-", color="#2563EB", linewidth=2, label="Sharpe")
    ax1.set_xlabel("Lookback (mois)")
    ax1.set_ylabel("Sharpe (rf=0)", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.axvline(cfg.baseline_lookback, linestyle="--", color="grey", alpha=0.6, label="Baseline (12m)")

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["Max Drawdown"] * 100, "s--", color="#DC2626", linewidth=2, label="Max Drawdown")
    ax2.set_ylabel("Max Drawdown (%)", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")

    fig.suptitle("Sensibilité à la fenêtre lookback (vol targeting)", fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "sensitivity_lookback.png", dpi=cfg.dpi)
    plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'sensitivity_lookback.png'}")


def plot_heatmap_w_sat_vs_scale(core: pd.Series, sat: pd.Series, cfg: SensitivityConfig) -> None:
    print("\n[BONUS] Heatmap Sharpe(w_sat, scale_max)...")
    w_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    s_grid = [1.0, 1.25, 1.50, 1.75, 2.0, 2.5]
    sharpe_matrix = np.full((len(s_grid), len(w_grid)), np.nan)

    for i, s_max in enumerate(s_grid):
        for j, w in enumerate(w_grid):
            mets = run_portfolio(core, sat, w_sat=w, target_vol=cfg.baseline_target_vol,
                                 lookback=cfg.baseline_lookback, scale_min=cfg.baseline_scale_min,
                                 scale_max=s_max)
            sharpe_matrix[i, j] = mets["Sharpe (rf=0)"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(sharpe_matrix, cmap="RdYlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(w_grid))); ax.set_xticklabels([f"{w:.0%}" for w in w_grid])
    ax.set_yticks(range(len(s_grid))); ax.set_yticklabels([f"{s:.2f}" for s in s_grid])
    ax.set_xlabel("Poids Satellite"); ax.set_ylabel("Scale max")

    for i in range(len(s_grid)):
        for j in range(len(w_grid)):
            val = sharpe_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if val < 0.9 or val > 1.4 else "black")
    try:
        bj = w_grid.index(cfg.baseline_w_sat); bi = s_grid.index(cfg.baseline_scale_max)
        ax.plot(bj, bi, "s", markersize=18, markerfacecolor="none", markeredgecolor="black", markeredgewidth=2.5)
    except ValueError:
        pass

    plt.colorbar(im, label="Sharpe (rf=0)")
    ax.set_title("Sharpe : poids Satellite × Scale max", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.fig_dir / "sensitivity_sharpe_heatmap.png", dpi=cfg.dpi)
    plt.close()
    print(f"  -> Figure: {cfg.fig_dir / 'sensitivity_sharpe_heatmap.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = SensitivityConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANALYSE DE SENSIBILITÉ")
    print("=" * 60)

    core = lire_core_policy(cfg.core_csv, cfg.policy_w_equity, cfg.policy_w_rates, cfg.policy_w_credit)
    sat = lire_serie_returns(cfg.sat_csv, "satellite_portfolio_return")
    core, sat = aligner_series(core, sat)
    print(f"Core mode: policy ({cfg.policy_w_equity:.0%}/{cfg.policy_w_rates:.0%}/{cfg.policy_w_credit:.0%})")
    print(f"Période commune: {core.index.min().date()} -> {core.index.max().date()} ({len(core)} mois)")

    df_w = sensibilite_w_sat(core, sat, cfg)
    plot_sensitivity_w_sat(df_w, cfg)

    df_s = sensibilite_scale_cap(core, sat, cfg)
    plot_sensitivity_scale_cap(df_s, cfg)

    df_lb = sensibilite_lookback(core, sat, cfg)
    plot_sensitivity_lookback(df_lb, cfg)

    plot_heatmap_w_sat_vs_scale(core, sat, cfg)

    print("\n" + "=" * 60)
    print("RÉSUMÉ DE LA SENSIBILITÉ")
    print("=" * 60)
    best_w = df_w["Sharpe (rf=0)"].idxmax()
    print(f"\n1) w_sat optimal = {best_w:.0%} (Sharpe = {df_w.loc[best_w, 'Sharpe (rf=0)']:.3f})")
    print(f"   Baseline (30%) : Sharpe = {df_w.loc[0.30, 'Sharpe (rf=0)']:.3f}")
    best_s = df_s["Sharpe (rf=0)"].idxmax()
    print(f"\n2) scale_max optimal = {best_s:.2f} (Sharpe = {df_s.loc[best_s, 'Sharpe (rf=0)']:.3f})")
    best_lb = df_lb["Sharpe (rf=0)"].idxmax()
    print(f"\n3) lookback optimal = {best_lb} mois (Sharpe = {df_lb.loc[best_lb, 'Sharpe (rf=0)']:.3f})")
    print("\nTerminé.")


if __name__ == "__main__":
    main()