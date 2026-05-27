"""effects.py — AME, Odds Ratios, Contrafactual e Interação econ×espectro.

┌─ REFERÊNCIA — visão geral ──────────────────────────────────────────────────
│ Cameron & Trivedi (2005) Microeconometrics, Cambridge University Press,
│   Cap.14 p.470-477: AME via diferenças finitas para variável binária é
│   preferível ao derivativo contínuo (garante validade para variáveis
│   discretas e produz interpretação direta em probabilidade).
│
│ Wooldridge (2016) Introductory Econometrics, 6ª ed., Cap.17 p.576-582:
│   AME = E_X[∂E[Y|X]/∂x_j] — média sobre a distribuição dos dados.
│
│ NOTA sobre DVs Firth-Ridge (5 singulares):
│   AME e contrafactual calculados APENAS para ordered_logit (MLE padrão).
│   Coeficientes penalizados (L2-ridge) não geram efeitos marginais
│   interpretáveis via MLE — requerem abordagem bayesiana.
│   OR para Firth-Ridge = exp(beta_penalizado): limite conservador do efeito
│   (encolhimento L2 em direção a 0 → OR mais próximo de 1 que o verdadeiro).
└───────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

from model_base import ModelResult, fit_ordered_logit
from config import ECON_COL

log = logging.getLogger(__name__)


# ─── Average Marginal Effects (diferenças finitas) ───────────────────────────

def compute_ame(
    mr: ModelResult,
    X: pd.DataFrame,
    focal_col: str = ECON_COL,
) -> Optional[pd.Series]:
    """
    AME via diferenças finitas para uma variável binária (padrão: econ).

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Cameron & Trivedi (2005) Microeconometrics, Cap.14 p.470:
    │   Para variável binária focal, AME = E_X[P(Y=j|X, focal=1) − P(Y=j|X, focal=0)]
    │   (diferença finita, não derivativo) — interpretação direta em probabilidade.
    │
    │ HLS (2013) Applied Logistic Regression, Cap.8 p.290:
    │   Probabilidades previstas do ordered logit como base dos efeitos marginais.
    │
    │ NOTA sobre Firth-Ridge: AME requer predict() do MLE padrão. Para os 5 DVs
    │   ajustados via Firth-Ridge, o predict() usa coeficientes penalizados (L2),
    │   produzindo efeitos marginais distorcidos. Esses DVs são excluídos (retorna None).
    └─────────────────────────────────────────────────────────────────────────

    Para cada observação i:
        AME_j = P(Y_i=j | X_i, focal=1) - P(Y_i=j | X_i, focal=0)

    Média sobre i → AME_j por categoria.

    Retorna pd.Series indexada por categoria {0, 1, ..., K-1} + 'delta_Y'.
    """
    if mr.result_obj is None or mr.model_type not in ("ordered_logit",):
        log.debug(
            "compute_ame: pulando (model_type=%s) — AME requer MLE padrão, "
            "não disponível para modelos penalizados (Firth-Ridge)", mr.model_type
        )
        return None
    if focal_col not in X.columns:
        return None

    res = mr.result_obj
    classes = mr.classes

    X1 = X.copy(); X1[focal_col] = 1.0
    X0 = X.copy(); X0[focal_col] = 0.0

    try:
        p1 = res.predict(X1)   # shape (n_obs, K)
        p0 = res.predict(X0)
    except Exception:
        return None

    diff = p1 - p0   # shape (n_obs, K)
    ame  = diff.mean(axis=0)  # shape (K,)

    # Variação na resposta esperada: Δ_Y = Σ_j j * AME_j
    delta_y = float(np.sum(classes.astype(float) * ame))

    result = pd.Series(
        list(ame) + [delta_y],
        index=[f"AME_cat{int(c)}" for c in classes] + ["delta_Y"],
    )
    return result


def compute_ame_all(
    mr: ModelResult,
    X: pd.DataFrame,
    focal_col: str = ECON_COL,
) -> dict:
    """
    Retorna dict com:
      E_Y_econ0, E_Y_econ1, delta_Y, AME_cat{j} para cada categoria j.

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Wooldridge (2016) Cap.17 p.576: E[Y|econ=1] e E[Y|econ=0] são médias
    │   das respostas esperadas preditas pelo modelo (não médias observadas raw)
    │   — condicionais aos controles, avaliam o efeito ceteris paribus.
    │
    │ NOTA Firth-Ridge: mesmo filtro de compute_ame() — retorna {} para os
    │   5 DVs singulares (coefs penalizados não geram AME MLE interpretável).
    └─────────────────────────────────────────────────────────────────────────
    """
    if mr.result_obj is None or mr.model_type not in ("ordered_logit",) or focal_col not in X.columns:
        log.debug(
            "compute_ame_all: pulando (model_type=%s) — consistente com compute_ame()", mr.model_type
        )
        return {}

    res = mr.result_obj
    classes = mr.classes
    X1 = X.copy(); X1[focal_col] = 1.0
    X0 = X.copy(); X0[focal_col] = 0.0

    try:
        p1 = res.predict(X1)
        p0 = res.predict(X0)
    except Exception:
        return {}

    E1 = float(np.mean(p1 @ classes.astype(float)))
    E0 = float(np.mean(p0 @ classes.astype(float)))

    diff = p1 - p0
    ame  = diff.mean(axis=0)

    out = {"E_Y_econ1": E1, "E_Y_econ0": E0, "delta_Y": E1 - E0}
    for k, c in enumerate(classes):
        out[f"AME_cat{int(c)}"] = float(ame[k])
    return out


# ─── Odds Ratios ──────────────────────────────────────────────────────────────

def compute_odds_ratios(mr: ModelResult, col: str = ECON_COL) -> dict:
    """
    OR = exp(β), IC95 via delta method: exp(β ± 1.96·SE).

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ HLS (2013) Applied Logistic Regression, Cap.8 p.290-295:
    │   No ordered logit (convenção statsmodels β > 0 → ↑P(Y > k)):
    │   OR = exp(β); OR > 1 → maior probabilidade de categorias superiores.
    │
    │ IC 95% via delta method: exp(β ± z_{α/2}·SE)
    │   — Aproximação normal assintótica; válida para n grande.
    │   Com n_econ=17, pode subestimar incerteza nas caudas (IC liberal).
    │   Bootstrap BC₀ (bloco 7) fornece alternativa mais robusta.
    │
    │ NOTA Firth-Ridge: compute_odds_ratios() inclui todos os model_types.
    │   Para os 5 DVs singulares (Firth-Ridge), OR = exp(beta_penalizado):
    │   encolhimento L2 tende a atenuar β em direção a 0 → OR tende a
    │   subestimar a magnitude do efeito real. Não é lower bound formal
    │   para modelos não-lineares; interpretar como heurística direcional.
    └─────────────────────────────────────────────────────────────────────────
    """
    if col not in mr.params.index:
        return {"OR": np.nan, "OR_CI_lo": np.nan, "OR_CI_hi": np.nan}

    beta = float(mr.params[col])
    se   = float(mr.se[col]) if col in mr.se.index else np.nan

    OR     = np.exp(beta)
    CI_lo  = np.exp(beta - 1.96 * se) if not np.isnan(se) else np.nan
    CI_hi  = np.exp(beta + 1.96 * se) if not np.isnan(se) else np.nan

    return {"OR": round(OR, 3), "OR_CI_lo": round(CI_lo, 3), "OR_CI_hi": round(CI_hi, 3)}


# ─── Contrafactual (médias observadas e contrafactual para economistas) ───────

def compute_counterfactual(
    y: pd.Series,
    X: pd.DataFrame,
    mr: ModelResult,
    econ_col: str = ECON_COL,
) -> dict:
    """
    média_e0: média observada y[econ=0]
    média_e1: média observada y[econ=1]
    media_cf: E[Y | X, econ=0] para as observações com econ=1
              (como economistas responderiam sem formação)

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Imbens & Rubin (2015) Causal Inference for Statistics, Social, and
    │   Biomedical Sciences, Cap.12 §12.2, p.258-260: CIA/unconfoundedness
    │   W_i ⊥ (Y_i(0), Y_i(1)) | X_i — exige que todos os confunderes
    │   estejam em X. §12.2.3 p.261: "Unconfoundedness Is Not Testable".
    │   ATT = E[Y(0)|W=1] = media_cf (Cap.12 §12.4, p.268-269).
    │
    │ Angrist & Pischke (2009) Mostly Harmless Econometrics, Cap.2 §2.2 p.28:
    │   CIA (Conditional Independence Assumption): Y(0),Y(1) ⊥ Z | X
    │   — exige que TODOS os confunderes estejam em X.
    │
    │ LIMITAÇÃO (CIA): com n_econ=17 e 6 controles observáveis (espectro,
    │   escolaridade, idade, gênero, trabalha_econ, engajado), há risco
    │   substancial de viés de seleção não-observada (ex: habilidade analítica
    │   inata, motivação intrínseca que induz tanto a formação quanto as
    │   opiniões). O contrafactual é interpretação ASSOCIACIONAL, não causal
    │   pura. Deve ser reportado com essa ressalva nas limitações do TCC.
    │
    │ NOTA Firth-Ridge: contrafactual calculado apenas para ordered_logit
    │   (linha 138 original) — coeficientes penalizados distorcem predict().
    └─────────────────────────────────────────────────────────────────────────
    """
    out = {"media_e0": np.nan, "media_e1": np.nan, "media_cf": np.nan}
    if econ_col not in X.columns:
        return out

    mask0 = X[econ_col] == 0
    mask1 = X[econ_col] == 1
    out["media_e0"] = float(y[mask0].mean()) if mask0.any() else np.nan
    out["media_e1"] = float(y[mask1].mean()) if mask1.any() else np.nan

    if mask1.any() and mr.result_obj is not None and mr.model_type == "ordered_logit":
        X_cf = X.loc[mask1].copy()
        X_cf[econ_col] = 0.0
        try:
            probs    = mr.result_obj.predict(X_cf)
            exp_vals = probs @ mr.classes.astype(float)
            out["media_cf"] = float(np.mean(exp_vals))
        except Exception:
            pass

    return out


# ─── Interaction econ × espectro ─────────────────────────────────────────────

def compute_interaction(
    y: pd.Series,
    X: pd.DataFrame,
    dv: str = "",
    econ_col: str = ECON_COL,
    espectro_col: str = "espectro",   # nome na matriz de controles (data_loader)
) -> dict:
    """
    Ajusta ordered logit com interação econ×espectro.
    Reporta beta_interacao, p_interacao, e AME de econ no Q1 e Q3 do espectro.

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Wooldridge (2016) Introductory Econometrics, Cap.6 §6-3 p.175-178:
    │   Interação β_3·x1·x2 → efeito de x1 sobre Y depende do valor de x2.
    │   β_3 > 0: efeito de econ AUMENTA com espectro (mais à direita → maior
    │   efeito de formação); β_3 < 0: efeito DIMINUI com espectro.
    │
    │ NOTA collinearidade: criar econ_x_espectro introduz correlação entre
    │   os três regressores (econ, espectro, produto). Contudo, espectro já
    │   está centrado em 0 (Centro=0 no IDEOL_MAP — ver config.py), o que
    │   minimiza a colinearidade entre espectro e econ_x_espectro. Centering
    │   adicional não é necessário neste caso.
    │
    │ NOTA exogeneidade: espectro pode ser endógeno em relação a econ —
    │   economistas tendem a se posicionar mais à direita (auto-seleção
    │   ideológica da profissão). Nesse caso, o coeficiente de interação
    │   β_3 capta moderação real + viés de seleção conjunta. Reportar como
    │   associação, não moderação causal pura.
    │
    │ AME de econ no Q1/Q3 do espectro (ame_at): substitui espectro pelo valor
    │   Q1 ou Q3 para TODA a amostra — exercício hipotético, não AME condicional
    │   em respondentes com espectro no Q1/Q3 (Wooldridge Cap.6 §6-2d p.179:
    │   "plug in interesting values of x_j — such as lower and upper quartiles").
    │   Angrist & Pischke (2009) Cap.3 pp.64-68: heterogeneidade de efeito.
    └─────────────────────────────────────────────────────────────────────────
    """
    if econ_col not in X.columns or espectro_col not in X.columns:
        return {
            "beta_interacao": np.nan,
            "se_interacao":   np.nan,
            "p_interacao":    np.nan,
            "AME_econ_Q1":    np.nan,
            "AME_econ_Q3":    np.nan,
        }

    X_int = X.copy()
    X_int["econ_x_espectro"] = X_int[econ_col] * X_int[espectro_col]

    mr_int = fit_ordered_logit(y, X_int, dv=dv + "_int")
    if mr_int is None:
        return {
            "beta_interacao": np.nan,
            "se_interacao":   np.nan,
            "p_interacao":    np.nan,
            "AME_econ_Q1":    np.nan,
            "AME_econ_Q3":    np.nan,
        }

    beta_int = float(mr_int.params.get("econ_x_espectro", np.nan))
    se_int   = float(mr_int.se.get("econ_x_espectro", np.nan))
    p_int    = float(mr_int.pvalues.get("econ_x_espectro", np.nan))

    # AME de econ avaliado no Q1 e Q3 do espectro
    esp_vals = X[espectro_col].dropna()
    q1 = float(esp_vals.quantile(0.25))
    q3 = float(esp_vals.quantile(0.75))

    def ame_at(esp_val: float) -> float:
        X_sub = X_int.copy()
        X_sub[espectro_col] = esp_val
        X_sub["econ_x_espectro"] = X_sub[econ_col] * esp_val
        ame = compute_ame(mr_int, X_sub, focal_col=econ_col)
        return float(ame["delta_Y"]) if ame is not None else np.nan

    return {
        "beta_interacao": round(beta_int, 4),
        "se_interacao":   round(se_int, 4),
        "p_interacao":    round(p_int, 4),
        "AME_econ_Q1":    round(ame_at(q1), 4),
        "AME_econ_Q3":    round(ame_at(q3), 4),
    }
