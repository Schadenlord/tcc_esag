"""permutation_tests.py — Testes não-paramétricos marginais por DV.

┌─ REFERÊNCIA — visão geral ──────────────────────────────────────────────────
│ Davidson & MacKinnon (2004) Econometric Theory and Methods, Cap.4 p.154-162:
│   Permutation test: H₀: F_{Y|econ=0} = F_{Y|econ=1} (igualdade de
│   distribuições marginais). Válido sem suposição de normalidade; exato
│   para permutações finitas da amostra observada.
│
│ Phipson & Smyth (2010) Permutation P-values Should Never Be Zero,
│   Statistical Applications in Genetics and Molecular Biology, §2.2:
│   N_PERMUTAÇÕES mínimo para estabilidade de p-valor: B ≥ (α⁻¹ − 1)×10.
│   Para α=0.05: B ≥ 190. N_PERMUTATIONS=5000 é conservadoramente maior.
│
│ Mann-Whitney U (MWU): testa H₀: P(Y_econ1 > Y_econ0) = 0.5 (stochastic
│   dominance), não igualdade de médias. Mais apropriado que t-test para DVs
│   ordinais (Agresti 2002, Categorical Data Analysis, Cap.8 p.278).
│
│ LIMITAÇÃO FUNDAMENTAL: nenhum desses testes controla por covariates.
│   Testam H₀ marginal P(Y|econ=0) = P(Y|econ=1), não H₀ condicional
│   P(Y|econ=0, X) = P(Y|econ=1, X). Devem ser reportados como evidência
│   descritiva não-ajustada, complementar ao ordered logit (blocos 3–6).
│   A diferença entre p-valores marginais (bloco 8) e condicionais (bloco 3)
│   reflete o papel dos controles — economistas diferem no espectro, escolaridade
│   e engajamento, que também predizem as DVs.
│
│ DESEQUILÍBRIO n₀:n₁≈166:17≈10:1: MWU e permutation são robustos a
│   desequilíbrio amostral. KS pode perder poder com n₁=17 (distribuição
│   empírica com apenas 17 pontos de suporte é grosseira).
└─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
from statsmodels.stats.multitest import multipletests

from config import ECON_COL, N_PERMUTATIONS, BOOTSTRAP_SEED, ALPHA

log = logging.getLogger(__name__)


def permutation_test_median_diff(
    y0: np.ndarray,
    y1: np.ndarray,
    n_perm: int = N_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """
    Permutation test: H₀ grupos econ=0 e econ=1 têm mesma mediana.
    Retorna p-value empírico (two-tailed).

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Davidson & MacKinnon (2004) Cap.4 p.156: p-valor permutacional =
    │   #{|T*_b| ≥ |T_obs|} / B, onde T = diferença de medianas (two-tailed).
    │   Válido sem suposição distribucional; exato sob H₀ de permutabilidade.
    │
    │ Phipson & Smyth (2010) §2.2: usar #{|T*_b| ≥ |T_obs|} / B (sem +1
    │   no numerador/denominador) é válido quando B é suficientemente grande
    │   — aqui n_perm=5000 excede largamente o mínimo B≥190 para α=0.05.
    │
    │ NOTA: estatística = |mediana(y1) − mediana(y0)| — diferença de medianas,
    │   não de distribuições completas (KS). Captura deslocamento de localização
    │   ordinal; menos poderoso que MWU para detectar deslocamentos monotônicos.
    └─────────────────────────────────────────────────────────────────────────
    """
    rng = np.random.default_rng(seed)
    combined = np.concatenate([y0, y1])
    n0, n1   = len(y0), len(y1)
    obs_stat = abs(np.median(y1) - np.median(y0))

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        count += abs(np.median(perm[n0:]) - np.median(perm[:n0])) >= obs_stat

    return float(count / n_perm)


def run_permutation_tests(
    df_all: pd.DataFrame,
    dependentes_cols: list[str],
    control_cols: list[str],
    econ_col: str = ECON_COL,
    n_perm: int = N_PERMUTATIONS,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Para cada DV, roda:
      1. Mann-Whitney U (scipy) — stochastic dominance
      2. Kolmogorov-Smirnov (scipy) — igualdade de distribuições
      3. Permutation test (diferença de medianas)

    Aplica correção BH sobre as 3 famílias de p-valores.

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ MWU como stochastic dominance: Agresti (2002) Cap.8 p.278 — para DVs
    │   ordinais, MWU é preferível ao t-test pois não assume intervalaridade
    │   das categorias. H₀: P(Y_econ1 > Y_econ0) = P(Y_econ1 < Y_econ0).
    │
    │ KS: testa H₀ de igualdade da função de distribuição acumulada completa.
    │   Com n₁=17, o suporte empírico F̂(y|econ=1) tem resolução grosseira
    │   → menor poder que MWU para detectar deslocamentos ordenados.
    │
    │ COVARIATES: nenhum desses testes condiciona em controles. A comparação
    │   entre p-valores marginais (este bloco) e p-valores condicionais do
    │   ordered logit (bloco 3) revela o quanto das diferenças observadas é
    │   explicado por diferenças nas distribuições de espectro, escolaridade,
    │   engajamento entre economistas e não-economistas.
    │
    │ SEED por DV: seed=BOOTSTRAP_SEED+i garante sequências RNG independentes
    │   entre os 53 DVs, evitando correlações espúrias nas estatísticas
    │   permutacionais.
    └─────────────────────────────────────────────────────────────────────────
    """
    rows = []
    for i, dv in enumerate(dependentes_cols):
        df_model = df_all[[dv, econ_col]].dropna()
        y = pd.to_numeric(df_model[dv], errors="coerce").dropna()
        if y.empty:
            rows.append({"DV": dv})
            continue

        econ = df_model[econ_col].loc[y.index]
        y0 = y[econ == 0].values
        y1 = y[econ == 1].values

        row = {"DV": dv, "n_e0": len(y0), "n_e1": len(y1)}

        if len(y0) > 0 and len(y1) > 0:
            mwu_stat, mwu_p = mannwhitneyu(y0, y1, alternative="two-sided")
            row["mwu_stat"] = round(float(mwu_stat), 2)
            row["mwu_p"]    = round(float(mwu_p), 4)

            ks_stat, ks_p = ks_2samp(y0, y1)
            row["ks_stat"] = round(float(ks_stat), 4)
            row["ks_p"]    = round(float(ks_p), 4)
        else:
            row["mwu_stat"] = np.nan
            row["mwu_p"]    = np.nan
            row["ks_stat"]  = np.nan
            row["ks_p"]     = np.nan

        if len(y0) > 1 and len(y1) > 1:
            perm_p = permutation_test_median_diff(y0, y1, n_perm=n_perm,
                                                   seed=BOOTSTRAP_SEED + i)
            row["perm_p"] = round(perm_p, 4)
        else:
            row["perm_p"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for p_col in ("mwu_p", "ks_p", "perm_p"):
        if p_col not in df.columns:
            continue
        valid_mask = df[p_col].notna()
        p_valid    = df.loc[valid_mask, p_col].values
        if len(p_valid) > 0:
            _, p_bh, _, _ = multipletests(p_valid, alpha=alpha, method="fdr_bh")
            df[f"{p_col}_bh"] = np.nan
            df.loc[valid_mask, f"{p_col}_bh"] = p_bh.round(4)

    return df
