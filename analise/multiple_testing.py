"""multiple_testing.py — correções de multiplicidade para as famílias de p-valores."""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)


# ┌─ REFERÊNCIA — apply_corrections ───────────────────────────────────────────
# │ Bonferroni, C.E. (1936). Teoria Statistica delle Classi e Calcolo delle
# │ Probabilità. Sezione di Matematica. Firenze.
# │ α_corrigido = α / m — controla FWER (Pr de qualquer falsa rejeição).
# │ Excessivamente conservador para m > 10; apresentado como referência.
# │
# │ Holm, S. (1979). "A simple sequentially rejective multiple test procedure."
# │ Scandinavian Journal of Statistics, 6(2), 65–70.
# │ Stepdown FWER: ordena p_(1) ≤ … ≤ p_(m); rejeita p_(i) se p_(i) ≤ α/(m−i+1).
# │ Domina Bonferroni uniformemente (mais poderoso, mesmo controle de FWER).
# │
# │ Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery rate:
# │ A practical and powerful approach to multiple testing."
# │ Journal of the Royal Statistical Society B, 57(1), 289–300.
# │ FDR = E[V/R]: taxa esperada de falsas descobertas entre as rejeitadas.
# │ Limiar: k* = max{k : p_(k) ≤ (k/m)α}. Válido sob independência ou PRDS
# │ (dependência positiva por resampling). Para DVs Likert de blocos conceituais
# │ correlacionados positivamente, PRDS é plausível → BH é o método principal.
# │
# │ Benjamini, Y. & Yekutieli, D. (2001). "The control of the false discovery
# │ rate in multiple testing under dependency."
# │ Annals of Statistics, 29(4), 1165–1188.
# │ FDR sob dependência arbitrária: usa fator c_m = Σ_{i=1}^{m} 1/i (harmônico).
# │ Mais conservador que BH; incluído como análise de sensibilidade.
# │
# │ DECISÕES:
# │  (a) BH aplicado por família separada (53 p-valores econ + 53 espectro).
# │      Famílias pré-especificadas por distinção teórica — dois preditores,
# │      duas perguntas de pesquisa. BH poolado (106 testes) seria mais
# │      conservador sem justificativa conceitual.
# │  (b) BY incluído como sensibilidade: se os 53 testes dentro de cada família
# │      forem positivamente dependentes (itens do mesmo bloco conceitual), BH
# │      é suficiente; BY é a alternativa conservadora robusta.
# │  (c) NaN tratados antes de passar para multipletests: a função exige vetor
# │      sem NaN. Posições originais de NaN são restauradas após a correção.
# └───────────────────────────────────────────────────────────────────────────────

def apply_corrections(
    pvalues: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Aplica Bonferroni, Holm (FWER), BH e BY (FDR) sobre um vetor de p-valores.

    Ignora NaN: corrige apenas os p-valores válidos e repõe NaN nas posições
    originais para manter alinhamento com os nomes das DVs.

    Retorna DataFrame com colunas: p_nominal, p_bonferroni, p_holm, p_bh, p_by.
    """
    pvals = np.asarray(pvalues, dtype=float)
    n     = len(pvals)

    df = pd.DataFrame({"p_nominal": pvals})
    valid_mask = ~np.isnan(pvals)
    p_valid    = pvals[valid_mask]

    if len(p_valid) == 0:
        for col in ("p_bonferroni", "p_holm", "p_bh", "p_by"):
            df[col] = np.nan
        return df

    log.debug("apply_corrections: %d p-valores válidos de %d", int(valid_mask.sum()), n)

    methods = [
        ("p_bonferroni", "bonferroni"),
        ("p_holm",       "holm"),
        ("p_bh",         "fdr_bh"),
        ("p_by",         "fdr_by"),
    ]
    for col_name, method in methods:
        _, p_adj, _, _ = multipletests(p_valid, alpha=alpha, method=method)
        col_arr = np.full(n, np.nan)
        col_arr[valid_mask] = p_adj
        df[col_name] = col_arr

    return df


def summarize_corrections(
    df_corrections: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Conta quantos testes passam cada limiar de correção.
    Entrada: saída de apply_corrections.
    """
    cols = [c for c in df_corrections.columns if c.startswith("p_")]
    rows = []
    for col in cols:
        sig = int(pd.to_numeric(df_corrections[col], errors="coerce").lt(alpha).sum())
        rows.append({"metodo": col.replace("p_", ""), "n_significativo": sig})
    return pd.DataFrame(rows)


def get_significant_dvs(
    df_corrections: pd.DataFrame,
    dv_names: list[str],
    alpha: float = 0.05,
) -> dict[str, list[str]]:
    """
    Retorna dict {método: [lista de DVs significativas]} para cada correção.

    Complementa summarize_corrections (que só conta) com os nomes das DVs
    que passam cada limiar — útil para tabelas de resultados e figuras.
    """
    result: dict[str, list[str]] = {}
    for col in [c for c in df_corrections.columns if c.startswith("p_")]:
        method = col.replace("p_", "")
        mask   = pd.to_numeric(df_corrections[col], errors="coerce") < alpha
        result[method] = [dv for dv, ok in zip(dv_names, mask) if ok]
    return result
