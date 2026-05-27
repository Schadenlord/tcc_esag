"""tables.py — agrega todos os resultados e salva CSVs em outputs/tables/csv/."""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ECON_COL, ESPECTRO_COL, ALPHA, BLOCS, SINGULAR_DVS,
    CSV_DIR, KEY_DVS,
)
from data_loader import DataBundle
from model_base import ModelResult, fit_ordered_logit, fit_firth_ridge, fit_lpm
from diagnostics import brant_test, goodness_of_fit, compute_vif, power_analysis
from multiple_testing import apply_corrections
from effects import compute_ame_all, compute_odds_ratios, compute_counterfactual, compute_interaction
from permutation_tests import run_permutation_tests

log = logging.getLogger(__name__)

CSV_DIR.mkdir(parents=True, exist_ok=True)


def _bloc_of(dv: str) -> str:
    for bloc, dvs in BLOCS.items():
        if dv in dvs:
            return bloc
    return "Outro"


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ─── Tabela descritiva ────────────────────────────────────────────────────────

def build_descritiva(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    for dv in bundle.dependentes_cols:
        df_dv = bundle.df_all[[dv, ECON_COL]].dropna()
        y = pd.to_numeric(df_dv[dv], errors="coerce")
        econ = df_dv[ECON_COL]
        y0 = y[econ == 0]
        y1 = y[econ == 1]
        rows.append({
            "DV":       dv,
            "Bloc":     _bloc_of(dv),
            "Pergunta": bundle.acronimos.get(dv, ""),
            "N_total":  len(y),
            "N_econ0":  len(y0),
            "N_econ1":  len(y1),
            "media_e0": round(y0.mean(), 3) if len(y0) > 0 else np.nan,
            "media_e1": round(y1.mean(), 3) if len(y1) > 0 else np.nan,
            "sd_e0":    round(y0.std(), 3)  if len(y0) > 0 else np.nan,
            "sd_e1":    round(y1.std(), 3)  if len(y1) > 0 else np.nan,
            "mediana_e0": y0.median() if len(y0) > 0 else np.nan,
            "mediana_e1": y1.median() if len(y1) > 0 else np.nan,
            "diff_media": round(y1.mean() - y0.mean(), 3)
                          if len(y0) > 0 and len(y1) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ─── Tabela síntese v2 (modelo PO) ───────────────────────────────────────────

def build_sintese(bundle: DataBundle) -> pd.DataFrame:
    """Roda ordered logit para todos os 53 DVs; inclui Brant, GOF, ORs, BH, Holm."""
    rows = []
    econ_pvals  = []
    esp_pvals   = []
    dv_order    = []

    for dv in bundle.dependentes_cols:
        log.info("Sintese: %s", dv)
        df_m = bundle.df_all[[dv] + bundle.control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
        X = df_m[bundle.control_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.loc[y.index]
        nuniq = X.nunique(); X = X.drop(columns=nuniq[nuniq <= 1].index)

        row = {
            "DV":    dv,
            "Bloc":  _bloc_of(dv),
            "N":     len(y),
            "Singular": dv in SINGULAR_DVS,
        }

        # Modelo PO ou Firth-Ridge para singulares
        if dv in SINGULAR_DVS:
            mr = fit_firth_ridge(y, X, dv=dv)
            if mr: row["model_type"] = "firth_ridge"
        else:
            mr = fit_ordered_logit(y, X, dv=dv)
            if mr: row["model_type"] = "ordered_logit"

        if mr is None:
            row.update({k: np.nan for k in [
                "beta_econ","se_econ","p_econ","OR","OR_CI_lo","OR_CI_hi",
                "beta_esp","se_esp","p_esp","mcfadden","aic","bic",
                "brant_chi2","brant_p","po_ok",
                "media_e0","media_e1","media_cf","delta_Y",
            ]})
            econ_pvals.append(np.nan); esp_pvals.append(np.nan)
            dv_order.append(dv); rows.append(row); continue

        # Coeficientes
        row["beta_econ"] = round(float(mr.params.get(ECON_COL, np.nan)), 4)
        row["se_econ"]   = round(float(mr.se.get(ECON_COL, np.nan)), 4)
        row["p_econ"]    = round(float(mr.pvalues.get(ECON_COL, np.nan)), 4)
        row["Singular"]  = mr.is_singular

        esp_col_real = next((c for c in mr.params.index if "espectro" in c.lower()
                             or "Com qual" in c), ESPECTRO_COL)
        row["beta_esp"]  = round(float(mr.params.get(esp_col_real, np.nan)), 4)
        row["se_esp"]    = round(float(mr.se.get(esp_col_real, np.nan)), 4)
        row["p_esp"]     = round(float(mr.pvalues.get(esp_col_real, np.nan)), 4)

        # ORs
        or_d = compute_odds_ratios(mr, col=ECON_COL)
        row.update(or_d)

        # GOF
        gof = goodness_of_fit(mr)
        row["mcfadden"] = gof["mcfadden_r2"]
        row["aic"]      = gof["aic"]
        row["bic"]      = gof["bic"]
        row["lr_p"]     = gof["lr_p"]

        # Brant (só para PO)
        if mr.model_type == "ordered_logit" and y.nunique() >= 3:
            try:
                br = brant_test(y, X)
                row["brant_chi2"] = round(br["chi2_stat"], 3)
                row["brant_p"]    = round(br["p_value"], 4)
                row["po_ok"]      = br["po_ok"]
            except Exception:
                row["brant_chi2"] = row["brant_p"] = np.nan
                row["po_ok"] = None
        else:
            row["brant_chi2"] = row["brant_p"] = np.nan
            row["po_ok"] = None

        # Contrafactual e AME
        cf = compute_counterfactual(y, X, mr)
        row.update(cf)
        ame_d = compute_ame_all(mr, X)
        row["delta_Y"] = round(ame_d.get("delta_Y", np.nan), 4)

        econ_pvals.append(row["p_econ"])
        esp_pvals.append(row["p_esp"])
        dv_order.append(dv)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Correções de multiplicidade
    corrections_econ = apply_corrections(np.array(econ_pvals, dtype=float))
    corrections_esp  = apply_corrections(np.array(esp_pvals,  dtype=float))

    df["p_econ_bonf"]    = corrections_econ["p_bonferroni"].values
    df["p_econ_holm"]    = corrections_econ["p_holm"].values
    df["p_econ_bh"]      = corrections_econ["p_bh"].values
    df["p_esp_bh"]       = corrections_esp["p_bh"].values
    df["p_esp_holm"]     = corrections_esp["p_holm"].values

    # Stars
    df["stars_econ"]     = df["p_econ"].apply(_stars)
    df["stars_econ_bh"]  = df["p_econ_bh"].apply(_stars)
    df["stars_esp"]      = df["p_esp"].apply(_stars)

    return df


# ─── Tabela AME (todos os DVs e 4 itens chave) ───────────────────────────────

def build_ame_todos(bundle: DataBundle, df_sintese: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for dv in bundle.dependentes_cols:
        df_m = bundle.df_all[[dv] + bundle.control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
        X = df_m[bundle.control_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.loc[y.index]
        nuniq = X.nunique(); X = X.drop(columns=nuniq[nuniq <= 1].index)
        mr = fit_ordered_logit(y, X, dv=dv)
        if mr is None:
            rows.append({"DV": dv, "Bloc": _bloc_of(dv)})
            continue
        ame = compute_ame_all(mr, X)
        row = {"DV": dv, "Bloc": _bloc_of(dv)}
        row.update({k: round(v, 4) for k, v in ame.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Tabela GOF ──────────────────────────────────────────────────────────────

def build_gof(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    for dv in bundle.dependentes_cols:
        df_m = bundle.df_all[[dv] + bundle.control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
        X = df_m[bundle.control_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.loc[y.index]; nuniq = X.nunique(); X = X.drop(columns=nuniq[nuniq <= 1].index)
        mr = fit_ordered_logit(y, X, dv=dv)
        if mr is None:
            rows.append({"DV": dv, "Bloc": _bloc_of(dv)}); continue
        gof = goodness_of_fit(mr)
        row = {"DV": dv, "Bloc": _bloc_of(dv), "N": mr.n_obs}
        row.update(gof)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Tabela Brant ─────────────────────────────────────────────────────────────

def build_brant(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    for dv in bundle.dependentes_cols:
        df_m = bundle.df_all[[dv] + bundle.control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
        X = df_m[bundle.control_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.loc[y.index]; nuniq = X.nunique(); X = X.drop(columns=nuniq[nuniq <= 1].index)
        row = {"DV": dv, "Bloc": _bloc_of(dv), "N_cats": int(y.nunique())}
        if y.nunique() >= 3:
            try:
                br = brant_test(y, X)
                row["brant_chi2"] = round(br["chi2_stat"], 3)
                row["brant_df"]   = br["df"]
                row["brant_p"]    = round(br["p_value"], 4)
                row["po_ok"]      = br["po_ok"]
            except Exception as e:
                row["brant_chi2"] = row["brant_p"] = np.nan
                row["brant_df"] = np.nan; row["po_ok"] = None
        else:
            row["brant_chi2"] = row["brant_p"] = row["brant_df"] = np.nan
            row["po_ok"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Tabela LPM ───────────────────────────────────────────────────────────────

def build_lpm(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    for dv in bundle.dependentes_cols:
        df_m = bundle.df_all[[dv] + bundle.control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
        X = df_m[bundle.control_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.loc[y.index]; nuniq = X.nunique(); X = X.drop(columns=nuniq[nuniq <= 1].index)
        mr = fit_lpm(y, X, dv=dv)
        row = {"DV": dv, "Bloc": _bloc_of(dv)}
        if mr and ECON_COL in mr.params.index:
            row["coef_econ_lpm"] = round(float(mr.params[ECON_COL]), 4)
            row["se_lpm"]        = round(float(mr.se[ECON_COL]), 4)
            row["p_lpm"]         = round(float(mr.pvalues[ECON_COL]), 4)
            row["r2_lpm"]        = round(mr.mcfadden_r2, 4)
        else:
            row.update({"coef_econ_lpm": np.nan, "se_lpm": np.nan,
                        "p_lpm": np.nan, "r2_lpm": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Tabela Firth para os 5 singulares ───────────────────────────────────────

def build_firth(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    for dv in SINGULAR_DVS:
        df_m = bundle.df_all[[dv] + bundle.control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
        X = df_m[bundle.control_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.loc[y.index]; nuniq = X.nunique(); X = X.drop(columns=nuniq[nuniq <= 1].index)
        mr = fit_firth_ridge(y, X, dv=dv)
        row = {"DV": dv}
        if mr and ECON_COL in mr.params.index:
            row["beta_econ_ridge"] = round(float(mr.params[ECON_COL]), 4)
            row["se_ridge"]        = round(float(mr.se[ECON_COL]), 4)
            row["p_ridge"]         = round(float(mr.pvalues[ECON_COL]), 4)
            row["mcfadden_ridge"]  = round(mr.mcfadden_r2, 4)
            row["singular"]        = mr.is_singular
            row["converged"]       = mr.converged
        else:
            row.update({"beta_econ_ridge": np.nan, "se_ridge": np.nan,
                        "p_ridge": np.nan, "mcfadden_ridge": np.nan,
                        "singular": True, "converged": False})
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Tabela Interação ────────────────────────────────────────────────────────

def build_interacao(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    esp_col = "espectro"   # nome na matriz de controles (data_loader)
    for dv in bundle.dependentes_cols:
        df_m = bundle.df_all[[dv] + bundle.control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
        X = df_m[bundle.control_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.loc[y.index]; nuniq = X.nunique(); X = X.drop(columns=nuniq[nuniq <= 1].index)
        row = {"DV": dv, "Bloc": _bloc_of(dv)}
        result = compute_interaction(y, X, dv=dv, espectro_col=esp_col)
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Tabela Resumo por Bloc ───────────────────────────────────────────────────

def build_resumo_blocs(df_sintese: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bloc in BLOCS:
        sub = df_sintese[df_sintese["Bloc"] == bloc]
        n_total  = len(sub)
        n_sing   = int(sub["Singular"].sum()) if "Singular" in sub.columns else 0
        n_valid  = n_total - n_sing
        n_esp_nom = int((sub["p_esp"] < ALPHA).sum()) if "p_esp" in sub.columns else 0
        n_esp_bh  = int((sub["p_esp_bh"] < ALPHA).sum()) if "p_esp_bh" in sub.columns else 0
        n_econ_nom = int((sub["p_econ"] < ALPHA).sum()) if "p_econ" in sub.columns else 0
        n_econ_bh  = int((sub["p_econ_bh"] < ALPHA).sum()) if "p_econ_bh" in sub.columns else 0
        rows.append({
            "Bloc": bloc, "N_itens": n_total, "N_singulares": n_sing,
            "N_estimaveis": n_valid,
            "Esp_nom": f"{n_esp_nom}/{n_total}",   # p_esp existe para todos os DVs (incl. Firth-Ridge)
            "Esp_BH":  f"{n_esp_bh}/{n_total}",
            "Econ_nom": f"{n_econ_nom}/{n_valid}",  # p_econ padrão só nos não-singulares
            "Econ_BH":  f"{n_econ_bh}/{n_valid}",
        })
    return pd.DataFrame(rows)


# ─── Carregamento rápido a partir de CSVs já gerados ─────────────────────────

def load_tables_from_csv() -> dict[str, pd.DataFrame]:
    """Carrega todas as tabelas dos CSVs em CSV_DIR (sem re-executar modelos)."""
    tables = {}
    for f in sorted(CSV_DIR.glob("tabela_*.csv")):
        key = f.stem.replace("tabela_", "")
        try:
            tables[key] = pd.read_csv(f, sep=";")
        except Exception:
            tables[key] = pd.read_csv(f)
    if not tables:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {CSV_DIR}. Rode sem --only primeiro."
        )
    return tables


# ─── Pipeline completo de tabelas ────────────────────────────────────────────

def build_all_tables(
    bundle: DataBundle,
    df_bootstrap: pd.DataFrame | None = None,
    df_permutation: pd.DataFrame | None = None,
    skip_slow: bool = False,
) -> dict[str, pd.DataFrame]:
    """Constrói todas as tabelas e salva CSVs. Retorna dict de DataFrames."""
    log.info("=== Construindo tabelas ===")

    tables = {}

    log.info("Descritiva...")
    tables["descritiva"] = build_descritiva(bundle)

    log.info("Síntese (modelos PO + Firth)...")
    tables["sintese_v2"] = build_sintese(bundle)

    log.info("AME todos DVs...")
    tables["ame_todos"] = build_ame_todos(bundle)

    # AME chave (4 itens com mais detalhe — mesmo que ame_todos mas filtrado)
    key_dvs = [dv for dv in KEY_DVS if dv in bundle.dependentes_cols]
    tables["ame_chave"] = tables["ame_todos"][tables["ame_todos"]["DV"].isin(key_dvs)]

    log.info("GOF...")
    tables["gof"] = build_gof(bundle)

    log.info("Brant test...")
    tables["brant"] = build_brant(bundle)

    log.info("LPM...")
    tables["lpm"] = build_lpm(bundle)

    log.info("Firth/Ridge singulares...")
    tables["firth"] = build_firth(bundle)

    if not skip_slow:
        log.info("Interações econ×espectro...")
        tables["interacao"] = build_interacao(bundle)

    log.info("Resumo por Bloc...")
    tables["resumo_blocs"] = build_resumo_blocs(tables["sintese_v2"])

    # Power analysis
    tables["power"] = power_analysis(
        n_econ_values=[17, 30, 50, 100],
        n_total_values=[183, 300, 500, 1000],
    )

    # Adiciona bootstrap e permutation se fornecidos
    if df_bootstrap is not None:
        tables["bootstrap_v2"] = df_bootstrap
    if df_permutation is not None:
        tables["permutation"] = df_permutation

    # Salva CSVs
    for name, df in tables.items():
        path = CSV_DIR / f"tabela_{name}.csv"
        df.to_csv(path, index=False, sep=";")
        log.info("Salvo: %s (%d linhas)", path.name, len(df))

    return tables
