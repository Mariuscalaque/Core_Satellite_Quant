"""
Pipeline Core – v4 (Excel unifié)

Sources de données :
  - 'ETF EUR Bloomberg Cross Asset.xlsx'
      * EQUITY : prix daily (paires date/prix)
      * RATES  : prix daily (paires date/prix)
      * CREDIT : prix daily (paires date/prix)
      * Worksheet (optionnel) : metadata TER

Pipeline :
  1) Lecture prix Equity/Rates/Credit depuis le même fichier
  2) Filtrage structurel (date de démarrage, fréquence ; frais si metadata dispo)
  3) Sélection 1 ETF « best » par thème :
       - Equity  : pick_best parmi EQUITY
       - Rates   : pick_best parmi RATES
       - Credit  : pick_best parmi CREDIT
  4) Backtest rolling OOS trimestriel (Max Sharpe, poids ∈ [5%, 50%])
  5) Export rendements journaliers du Core (3 ETFs sélectionnés)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path


@dataclass(frozen=True)
class CoreConfig:
    """Paramètres du pipeline Core."""
    project_root: Path = Path(__file__).resolve().parent.parent

    # ── Fichier Excel unifié ───────────────────────
    core_excel: str = str(project_root / "data" / "ETF EUR Bloomberg Cross Asset.xlsx")
    sheet_equity: str = "EQUITY"
    sheet_rates: str = "RATES"
    sheet_credit: str = "CREDIT"
    # Metadata TER optionnel
    sheet_meta: str = "Worksheet"
    require_expense_info: bool = False

    # ── Filtres Equity ────────────────────────────
    max_start_date: str = "2019-01-01"
    max_avg_gap_days: float = 2.0
    total_fee_budget_bps: float = 80.0
    w_core_mid: float = 0.725
    satellite_expense_bps: float = 60.0

    # ── pick_best ─────────────────────────────────
    min_obs_pick_best: int = 750
    # Fenêtre de calibration pour le scoring (évite le look-ahead)
    score_start: str = "2019-01-01"
    score_end: str = "2020-12-31"

    # ── Backtest ──────────────────────────────────
    lookback: int = 252
    rebal_freq: int = 63
    w_min: float = 0.05
    w_max: float = 0.50

    # ── Sorties ───────────────────────────────────

    output_core_daily_csv: str = str(project_root / "outputs" / "core_returns_daily_oos.csv")
    output_selected_core_csv: str = str(project_root / "outputs" / "core_selected_etfs.csv")
    output_core_finaux_csv: str = str(project_root / "outputs" / "Core_finaux.csv")

    @property
    def max_core_expense_pct(self) -> float:
        """Expense ratio max (%) pour un ETF core, dérivé du budget total."""
        w_sat = 1.0 - self.w_core_mid
        max_bps = (self.total_fee_budget_bps - w_sat * self.satellite_expense_bps) / self.w_core_mid
        return max_bps / 100.0

def _parse_dates_any(x: pd.Series) -> pd.Series:
    """Parse dates robuste : datetime / string dayfirst / serial Excel."""
    dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
    if dt.notna().sum() < max(5, int(0.20 * len(x))):
        num = pd.to_numeric(x, errors="coerce")
        mask_valid = num.notna() & (num >= 1) & (num <= 80_000)
        dt2 = pd.Series(pd.NaT, index=x.index)
        if mask_valid.any():
            dt2.loc[mask_valid] = pd.to_datetime(
                num.loc[mask_valid].astype(int),
                errors="coerce", unit="D", origin="1899-12-30"
            )
        dt = dt.fillna(dt2)
    # Évite les dates aberrantes (ex: 1970) dues aux cellules mal parsées.
    year = dt.dt.year
    dt = dt.where(year.between(1990, 2100))
    return dt


# ══════════════════════════════════════════════════
#  Lecture des données
# ══════════════════════════════════════════════════

def _lire_wide_paires(path: str, sheet, skiprows: int = 0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lecture générique d'un Excel avec paires (date, prix).
    Retourne (wide DataFrame, liste ordonnée des tickers).
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=skiprows)
    tickers_row = df.iloc[0, :].tolist()
    data = df.iloc[1:]

    seen: set = set()
    series_dict: Dict[str, pd.Series] = {}
    ordered_tickers: List[str] = []

    for j in range(0, df.shape[1], 2):
        ticker = tickers_row[j] if j < len(tickers_row) else None
        if not isinstance(ticker, str):
            continue
        ticker = ticker.strip()
        if ticker in seen:
            continue
        seen.add(ticker)

        price_col = j + 1
        if price_col >= df.shape[1]:
            break

        dates = _parse_dates_any(data.iloc[:, j])
        prices = pd.to_numeric(data.iloc[:, price_col], errors="coerce")
        valid = dates.notna() & prices.notna()
        if valid.sum() == 0:
            continue

        s = pd.Series(prices[valid].values,
                      index=pd.DatetimeIndex(dates[valid].values),
                      name=ticker)
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="first")]
        series_dict[ticker] = s
        ordered_tickers.append(ticker)

    wide = pd.DataFrame(series_dict).sort_index()
    wide.index.name = "Date"
    return wide, ordered_tickers


def lire_equity(cfg: CoreConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lit les ETFs Equity depuis le nouveau fichier + metadata,
    applique les filtres (date, fréquence, expense ratio).
    Retourne (wide_filtré, summary).
    """
    print("[1/7] Lecture Equity (fichier unifié)...")
    wide, _ = _lire_wide_paires(cfg.core_excel, cfg.sheet_equity)
    print(f"  -> {wide.shape[1]} ETFs uniques | {wide.index.min().date()} à {wide.index.max().date()}")

    # Metadata TER (optionnel)
    expense_map: Dict[str, float] = {}
    try:
        xls = pd.ExcelFile(cfg.core_excel)
        if cfg.sheet_meta in xls.sheet_names:
            df_meta = pd.read_excel(xls, sheet_name=cfg.sheet_meta)
            if {"Ticker", "Expense Ratio"}.issubset(set(df_meta.columns)):
                df_meta["expense_ratio_pct"] = (
                    df_meta["Expense Ratio"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )
                df_meta["ticker_full"] = df_meta["Ticker"].astype(str).str.strip() + " Equity"
                expense_map = dict(zip(df_meta["ticker_full"], df_meta["expense_ratio_pct"]))
    except Exception as exc:
        print(f"  ⚠  Metadata TER indisponible ({exc}). Filtre frais assoupli.")

    # Filtrage
    max_date = pd.Timestamp(cfg.max_start_date)
    max_expense = cfg.max_core_expense_pct
    print(f"[2/7] Filtrage Equity : début ≤ {cfg.max_start_date} | daily (gap ≤ {cfg.max_avg_gap_days}) "
          f"| expense ≤ {max_expense:.2f} %")

    kept: List[str] = []
    rows: List[dict] = []

    for ticker in wide.columns:
        prices = wide[ticker].dropna()
        if len(prices) < 10:
            continue
        first_date = prices.index.min()
        n_obs = len(prices)
        avg_gap = prices.index.to_series().diff().dropna().dt.days.mean()
        expense = expense_map.get(ticker, np.nan)

        pass_date = first_date <= max_date
        pass_freq = avg_gap <= cfg.max_avg_gap_days
        if cfg.require_expense_info:
            pass_expense = (not np.isnan(expense)) and (expense <= max_expense)
        else:
            pass_expense = np.isnan(expense) or (expense <= max_expense)
        selected = pass_date and pass_freq and pass_expense

        rows.append(dict(
            ticker=ticker, first_date=first_date.strftime("%Y-%m-%d"),
            n_obs=n_obs, avg_gap=round(avg_gap, 2), expense_pct=expense,
            pass_date=pass_date, pass_freq=pass_freq, pass_expense=pass_expense,
            selected=selected,
        ))
        if selected:
            kept.append(ticker)

    summary = pd.DataFrame(rows)
    print(f"  -> {len(wide.columns)} total | "
          f"excl. date {(~summary['pass_date']).sum()} | "
          f"excl. fréq {(~summary['pass_freq']).sum()} | "
          f"excl. frais {(~summary['pass_expense']).sum()}")
    print(f"  -> ✓ {len(kept)} ETFs Equity retenus")

    wide_filtered = wide[kept].dropna(how="all").ffill()
    return wide_filtered, summary


def lire_rates_credit(cfg: CoreConfig) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Lit les ETFs Rates & Credit depuis le fichier Excel unifié,
    reconstruit le mapping ticker → bucket.
    """
    print("[3/7] Lecture Rates / Credit (fichier unifié)...")
    wide_rates, _ = _lire_wide_paires(cfg.core_excel, cfg.sheet_rates)
    wide_credit, _ = _lire_wide_paires(cfg.core_excel, cfg.sheet_credit)

    # Filtrage structurel commun (date/frequence)
    max_date = pd.Timestamp(cfg.max_start_date)
    kept_rates: List[str] = []
    for t in wide_rates.columns:
        s = wide_rates[t].dropna()
        if len(s) < 10:
            continue
        avg_gap = s.index.to_series().diff().dropna().dt.days.mean()
        if s.index.min() <= max_date and avg_gap <= cfg.max_avg_gap_days:
            kept_rates.append(t)

    kept_credit: List[str] = []
    for t in wide_credit.columns:
        s = wide_credit[t].dropna()
        if len(s) < 10:
            continue
        avg_gap = s.index.to_series().diff().dropna().dt.days.mean()
        if s.index.min() <= max_date and avg_gap <= cfg.max_avg_gap_days:
            kept_credit.append(t)

    wide = pd.concat([wide_rates[kept_rates], wide_credit[kept_credit]], axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated(keep="first")]
    wide = wide.sort_index().ffill()
    wide.index.name = "Date"

    # Mapping thème pour la sélection pick_best
    ticker_to_bucket: Dict[str, str] = {}
    for t in kept_rates:
        ticker_to_bucket[t] = "Rates EMU Govies (Core)"
    for t in kept_credit:
        ticker_to_bucket[t] = "Credit EMU IG (Core)"

    print(f"  -> {wide.shape[1]} ETFs Rates/Credit | {wide.index.min().date()} à {wide.index.max().date()}")
    for t in wide.columns:
        print(f"     {t:25s}  bucket = {ticker_to_bucket.get(t, '?')}")

    return wide, ticker_to_bucket


# ══════════════════════════════════════════════════
#  Thèmes & pick_best
# ══════════════════════════════════════════════════

def definir_themes() -> Dict[str, Set[str]]:
    return {
        "Rates": {"Rates EMU Govies (Core)", "Rates EMU Govies (Bucket)", "Rates EMU Govies (Linkers)"},
        "Credit": {
            "Credit EMU IG (Core)", "Credit EMU IG (Bucket)",
            "Credit EMU IG (Large Cap)", "Credit EMU IG (Covered)",
            "EMU Aggregate (IG)",
        },
    }


def pick_best(
    theme_name: str,
    wide: pd.DataFrame,
    rets_log: pd.DataFrame,
    ticker_to_bucket: Dict[str, str],
    themes: Dict[str, Set[str]],
    min_obs: int,
) -> str:
    """
    Sélection du meilleur ETF par thème.
    rets_log doit être pré-restreint à la fenêtre de calibration (pas de look-ahead).
    score = corrélation moyenne intra-thème + bonus obs − pénalité vol.
    min_obs est appliqué sur la fenêtre de calibration passée.
    """
    cands = [t for t in wide.columns if ticker_to_bucket.get(t, "") in themes[theme_name]]
    # Filtre : ETF présent dans la fenêtre de calibration avec assez d'obs
    cands = [t for t in cands if t in rets_log.columns and rets_log[t].dropna().shape[0] >= min(min_obs, len(rets_log) // 2)]

    if len(cands) == 0:
        raise ValueError(f"Aucun candidat pour {theme_name} dans la fenêtre de calibration")

    R = rets_log[cands].dropna(how="all")
    corr = R.corr()
    avg_corr = corr.mean(axis=1).values
    obs = np.array([rets_log[t].dropna().shape[0] for t in cands], dtype=float)
    vol = R.std().values

    obs_z = (obs - obs.mean()) / (obs.std() + 1e-12)
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-12)

    score = avg_corr + 0.10 * obs_z - 0.10 * vol_z
    return cands[int(np.argmax(score))]


def pick_best_equity(
    wide_equity: pd.DataFrame,
    rets_log: pd.DataFrame,
    min_obs: int,
) -> str:
    """
    Sélection du meilleur ETF Equity parmi les candidats filtrés.
    rets_log doit être pré-restreint à la fenêtre de calibration (pas de look-ahead).
    """
    cands = [t for t in wide_equity.columns
             if t in rets_log.columns and rets_log[t].dropna().shape[0] >= min(min_obs, len(rets_log) // 2)]
    if len(cands) == 0:
        raise ValueError(f"Aucun candidat Equity dans la fenêtre de calibration")

    R = rets_log[cands].dropna(how="all")
    corr = R.corr()
    avg_corr = corr.mean(axis=1).values
    obs = np.array([rets_log[t].dropna().shape[0] for t in cands], dtype=float)
    vol = R.std().values

    obs_z = (obs - obs.mean()) / (obs.std() + 1e-12)
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-12)

    score = avg_corr + 0.10 * obs_z - 0.10 * vol_z
    return cands[int(np.argmax(score))]


def optimiser_max_sharpe_contraint(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """
    Optimisation Max Sharpe (rf=0) sous contraintes long-only et bornes [w_min, w_max].
    Fallback : équipondéré si échec.
    """
    n = len(mu)
    x0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov @ w))
        if vol < 1e-12:
            return 1e10
        ret = float(w @ mu)
        return -(ret / vol)

    bounds = [(w_min, w_max)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return x0
    return res.x


def backtest_rolling_oos(
    prices: pd.DataFrame,
    lookback: int,
    rebal_freq: int,
    w_min: float,
    w_max: float,
) -> pd.Series:
    """
    Backtest rolling trimestriel OOS :
    - On estime mu/cov sur une fenêtre lookback (log-rendements)
    - On optimise max sharpe sous contraintes
    - On applique les poids sur la période suivante (OOS) en log-rendements

    Sortie :
    - série de log-rendements OOS quotidiens du portefeuille Core
    """
    print("[6/7] Backtest rolling OOS (Max Sharpe contraint)...")

    rets_log = np.log(prices).diff().dropna()
    dates = rets_log.index

    port_log_rets: List[float] = []
    port_dates: List[pd.Timestamp] = []

    for start in range(lookback, len(dates) - rebal_freq, rebal_freq):
        window = rets_log.iloc[start - lookback:start]
        oos = rets_log.iloc[start:start + rebal_freq]

        mu = window.mean().values * 252
        cov = window.cov().values * 252

        w = optimiser_max_sharpe_contraint(mu, cov, w_min, w_max)

        oos_port = oos.values @ w
        port_log_rets.extend(oos_port.tolist())
        port_dates.extend(oos.index.tolist())

    s = pd.Series(port_log_rets, index=pd.DatetimeIndex(port_dates), name="core_log_return_oos")
    s = s.sort_index()
    print(f"  -> OOS: {s.index.min().date()} à {s.index.max().date()} | Nb jours: {len(s)}")
    return s


def log_daily_to_monthly_simple(log_daily: pd.Series) -> pd.Series:
    """
    Convertit des log-rendements journaliers en rendements simples mensuels :
    r_m = exp( somme(log_r_jour) ) - 1
    """
    print("[7/7] Conversion OOS journalier -> mensuel (r_m = exp(sum) - 1)...")
    monthly = np.exp(log_daily.resample("ME").sum()) - 1.0
    monthly = monthly.dropna()
    monthly.name = "core_portfolio_return"
    print(f"  -> Nb mois: {len(monthly)} | Début: {monthly.index.min().date()} | Fin: {monthly.index.max().date()}")
    return monthly


def main() -> None:
    cfg = CoreConfig()

    # 1-2) Equity : lecture + filtrage
    wide_eq, summary_eq = lire_equity(cfg)

    # 3) Rates / Credit : lecture depuis le fichier unifié
    wide_rc, ticker_to_bucket = lire_rates_credit(cfg)

    # 4) Sélection 1 ETF par thème (scoring sur 2019-2020 uniquement : pas de look-ahead)
    print("[4/7] Sélection pick_best (1 ETF / thème)...")
    print(f"  Fenêtre de scoring : {cfg.score_start} → {cfg.score_end}")

    rets_eq_full = np.log(wide_eq).diff()
    rets_eq_calib = rets_eq_full.loc[cfg.score_start:cfg.score_end]
    best_eq = pick_best_equity(wide_eq, rets_eq_calib, cfg.min_obs_pick_best)
    print(f"  -> Equity : {best_eq}")

    rets_rc_full = np.log(wide_rc).diff()
    rets_rc_calib = rets_rc_full.loc[cfg.score_start:cfg.score_end]
    themes = definir_themes()
    best_rt = pick_best("Rates", wide_rc, rets_rc_calib, ticker_to_bucket, themes, cfg.min_obs_pick_best)
    best_cr = pick_best("Credit", wide_rc, rets_rc_calib, ticker_to_bucket, themes, cfg.min_obs_pick_best)
    print(f"  -> Rates  : {best_rt}")
    print(f"  -> Credit : {best_cr}")

    core_etfs = [best_eq, best_rt, best_cr]
    core_themes = ["Equity", "Rates", "Credit"]

    # Sauvegarde Core_finaux.csv (avec Ticker et Theme)
    print("[5/7] Export Core_finaux.csv...")
    df_core_finaux = pd.DataFrame({
        "Ticker": core_etfs,
        "Theme": core_themes,
    })
    df_core_finaux.to_csv(cfg.output_core_finaux_csv, index=False)
    print(f"  -> {cfg.output_core_finaux_csv}")

    # Sauvegarde core_selected_etfs.csv (ancien format)
    pd.DataFrame({"core_etfs": core_etfs, "theme": core_themes})\
        .to_csv(cfg.output_selected_core_csv, index=False)
    print(f"  -> {cfg.output_selected_core_csv}")

    # Construire le wide combiné des 3 ETFs sélectionnés
    wide_combined = pd.concat([
        wide_eq[[best_eq]],
        wide_rc[[best_rt, best_cr]],
    ], axis=1).sort_index().dropna(how="all").ffill()

    # Export log-rendements journaliers des 3 ETFs (pour frontière efficiente)
    core_etf_log_daily = np.log(wide_combined).diff().dropna()
    etf_log_path = str(cfg.project_root / "outputs" / "core3_etf_daily_log_returns.csv")
    core_etf_log_daily.to_csv(etf_log_path)
    print(f"  -> {etf_log_path}")

    # 6) Backtest rolling OOS
    core_log_daily_oos = backtest_rolling_oos(
        wide_combined,
        lookback=cfg.lookback,
        rebal_freq=cfg.rebal_freq,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
    )
    core_log_daily_oos.to_frame().to_csv(cfg.output_core_daily_csv, index=True)
    print(f"  -> {cfg.output_core_daily_csv}")

    print(f"Export OK : {cfg.output_core_daily_csv}")


if __name__ == "__main__":
    main()
