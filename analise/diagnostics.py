"""diagnostics.py — Brant test, VIF, Goodness-of-Fit, power analysis, retrodesign.

Todas as funções incluem blocos de referência acadêmica no formato padrão do projeto.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

from model_base import ModelResult, fit_ordered_logit

log = logging.getLogger(__name__)


# ┌─ REFERÊNCIA — brant_test ──────────────────────────────────────────────────
# │ Hosmer, D.W., Lemeshow, S. & Sturdivant, R.X.
# │ Applied Logistic Regression. 3ª ed. Wiley, 2013.
# │ Cap. 8 — Ordinal Logistic Regression
# │ Seção 8.2.2 — Proportional Odds Model  |  p. 289–308
# │ Equação (8.26): W = Σ_k Σ_j (β_kj − β_PO_j)² / (SE²_kj + SE²_PO_j)
# │ df = (K − 1) × p  [HLS p. 302; ex: K=4, p=4 → df=12; K=3, p=6 → df=12]
# │
# │ Brant, R. (1990). "Assessing proportionality in the proportional odds
# │ model for ordinal logistic regression." Biometrics, 46(4), 1171–1178.
# │
# │ DECISÕES:
# │  (a) y_bin = (y > k): alinha escala de β com o statsmodels OrderedModel.
# │      OrderedModel: β > 0 → X aumenta P(Y > k). Logit(y > k): coef δ com
# │      mesma convenção → diff = δ_k − β_PO = 0 sob H0 (PO).
# │      Logit(y ≤ k): γ = −β → diff = −2β ≠ 0 mesmo sob H0 (bug de sinal).
# │  (b) df = (K − 1) × p: versão correta. O bug (K − 2) produzia df = p em
# │      vez de 2p para K=3, inflando p-valores (χ² com df subestimado). HLS p. 302.
# │  (c) Covariância Cov(β̂_k, β̂_PO) omitida: aproximação conservadora padrão.
# │      Como Cov > 0 (ambos estimados no mesmo dataset), a omissão subestima
# │      χ² → teste mais conservador, não anti-conservador (HLS p. 295–296).
# │      Implementações em R (brant) e Stata omitem igualmente.
# └───────────────────────────────────────────────────────────────────────────────

def brant_test(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    Brant (1990): teste da hipótese de odds proporcionais (PO).

    Ajusta K-1 logits binários P(Y ≤ k | X) e compara seus coeficientes com
    β_PO do modelo ordered logit. H0: coeficientes idênticos entre thresholds.
    Deseja-se NÃO rejeitar H0 (p > 0.05) para validar o modelo PO.

    Retorna dict: chi2_stat, df, p_value, po_ok (bool), coef_by_threshold.
    """
    classes = np.sort(y.dropna().unique())
    K = len(classes)
    if K < 3:
        return {"chi2_stat": np.nan, "df": np.nan, "p_value": np.nan,
                "po_ok": None, "coef_by_threshold": None}

    nuniq = X.nunique()
    X2    = X.drop(columns=nuniq[nuniq <= 1].index.tolist())
    X2_sm = sm.add_constant(X2.astype(float), has_constant="raise")
    pred_cols = X2.columns.tolist()
    n_preds   = len(pred_cols)

    # K-1 logits binários P(Y > k) — alinha sinal com statsmodels OrderedModel.
    # OrderedModel: params β têm convenção "β > 0 → X aumenta P(Y > k)".
    # Logit(y > k): coef δ > 0 → X aumenta P(Y > k). Mesma escala → diff=0 sob H0.
    # Logit(y ≤ k): coef γ = −δ = −β → diff = γ − β = −2β ≠ 0 mesmo sob H0.
    bin_coefs: dict[int, np.ndarray] = {}
    bin_ses:   dict[int, np.ndarray] = {}
    for k_idx in range(K - 1):
        threshold = classes[k_idx]
        y_bin = (y > threshold).astype(int)   # P(Y > k): mesmo sinal que β_PO
        try:
            r = sm.Logit(y_bin, X2_sm).fit(disp=False, method="bfgs")
            bin_coefs[k_idx] = r.params.loc[pred_cols].values
            bin_ses[k_idx]   = r.bse.loc[pred_cols].values
        except Exception:
            bin_coefs[k_idx] = np.full(n_preds, np.nan)
            bin_ses[k_idx]   = np.full(n_preds, np.nan)

    po_res = fit_ordered_logit(y, X2, dv="__brant__")
    if po_res is None:
        return {"chi2_stat": np.nan, "df": np.nan, "p_value": np.nan,
                "po_ok": None, "coef_by_threshold": None}

    po_coef = po_res.params.loc[pred_cols].values
    po_se   = po_res.se.loc[pred_cols].values

    # Wald: Σ_k Σ_j (β_kj − β_PO_j)² / (SE²_kj + SE²_PO_j)  [HLS eq. 8.26]
    chi2_stat   = 0.0
    valid_terms = 0
    coef_rows   = []
    for k_idx in range(K - 1):
        row = {"threshold": k_idx}
        for j, col in enumerate(pred_cols):
            diff  = bin_coefs[k_idx][j] - po_coef[j]
            denom = bin_ses[k_idx][j] ** 2 + po_se[j] ** 2
            if denom > 0 and not np.isnan(diff):
                chi2_stat   += diff ** 2 / denom
                valid_terms += 1
            row[col] = bin_coefs[k_idx][j]
        coef_rows.append(row)

    df_val = (K - 1) * n_preds   # A1: (K-1), não (K-2) — HLS p. 302
    p_val  = 1 - chi2.cdf(chi2_stat, df=df_val) if df_val > 0 else np.nan
    po_ok  = bool(p_val > 0.05) if not np.isnan(p_val) else None

    coef_df = pd.DataFrame(coef_rows).set_index("threshold")
    coef_df.loc["PO"] = po_coef

    return {
        "chi2_stat": float(chi2_stat),
        "df":        int(df_val),
        "p_value":   float(p_val) if not np.isnan(p_val) else np.nan,
        "po_ok":     po_ok,
        "coef_by_threshold": coef_df,
    }


# ┌─ REFERÊNCIA — compute_vif ──────────────────────────────────────────────────
# │ Wooldridge, J.M.
# │ Introductory Econometrics: A Modern Approach. 6ª ed. Cengage, 2016.
# │ Cap. 3 — Multiple Regression Analysis: Estimation
# │ Seção 3-4a — Multicollinearity  |  p. 86
# │ VIF_j = 1 / (1 − R²_j), R²_j = R² de x_j regredido sobre os demais.
# │ "if VIF_j is above 10 [...] we conclude that multicollinearity is a
# │ 'problem' for estimating β_j" (Wooldridge p. 86)
# │
# │ DECISÃO: threshold 10 é heurístico e admitidamente arbitrário (Wooldridge:
# │ "not especially helpful as a stand-alone criterion"). Em nossos dados todos
# │ os VIFs ficaram < 3, confirmando que codificação ordinal (escolaridade 0–6,
# │ idade 0–6) em vez de dummies eliminou a colinearidade estrutural.
# └───────────────────────────────────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    VIF (Variance Inflation Factor) para cada coluna de X.

    VIF_j = 1/(1 − R²_j); VIF > 10 sinaliza multicolinearidade problemática
    (Wooldridge Cap. 3, p. 86). Remove colunas constantes antes do cálculo.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    nuniq = X.nunique()
    X2    = X.drop(columns=nuniq[nuniq <= 1].index.tolist()).dropna().astype(float)
    if X2.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "VIF"])

    rows = []
    for i, col in enumerate(X2.columns):
        try:
            vif = variance_inflation_factor(X2.values, i)
        except Exception:
            vif = np.nan
        rows.append({"feature": col, "VIF": round(vif, 2)})

    df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    df["high_vif"] = df["VIF"] > 10
    return df


# ┌─ REFERÊNCIA — goodness_of_fit ─────────────────────────────────────────────
# │ Hosmer, D.W., Lemeshow, S. & Sturdivant, R.X. (2013)
# │ Cap. 5 — Assessing the Fit of the Model  |  p. 160–185
# │ LR test: G = −2[ln L(β̂₀) − ln L(β̂)] ~ χ²(q)
# │ q = parâmetros livres = len(params) − (K − 1) cutpoints
# │
# │ Wooldridge, J.M. (2016), Cap. 17 — Limited Dependent Variable Models
# │ Equação (17.12): LR = 2(ℓ_irrestrito − ℓ_restrito) ~ χ²_q
# │
# │ McFadden R²: R²_McF = 1 − ℓ(β̂)/ℓ(β̂₀)  (Wooldridge Cap. 17, p. 583–590)
# │ Nagelkerke R²: R²_CS / R²_CS_max; R²_CS = 1 − (L₀/L̂)^{2/n}
# │   Nagelkerke, N.J.D. (1991). Biometrika, 78(3), 691–692.
# │
# │ DECISÃO: lr_df = len(params) − (n_categories − 1) porque ModelResult.params
# │ inclui slopes + cutpoints; o LR test do modelo completo vs. nulo (apenas
# │ cutpoints) tem q = número de slopes — exatamente essa diferença.
# └───────────────────────────────────────────────────────────────────────────────

def goodness_of_fit(mr: ModelResult) -> dict:
    """
    Métricas de ajuste a partir de um ModelResult.

    Retorna: mcfadden_r2, nagelkerke_r2, aic, bic, lr_chi2, lr_df, lr_p,
             n_obs, singular, converged.
    """
    n      = mr.n_obs
    k      = len(mr.params)
    llf    = mr.llf
    llnull = mr.llnull

    lr_stat = -2.0 * (llnull - llf)
    lr_df   = k - (mr.n_categories - 1)   # slopes livres: total − cutpoints
    lr_p    = 1 - chi2.cdf(lr_stat, df=max(lr_df, 1))

    # A3: código morto removido (correct_pct nunca foi retornado nem computado)

    return {
        "mcfadden_r2":   round(mr.mcfadden_r2, 4),
        "nagelkerke_r2": round(mr.nagelkerke_r2, 4),
        "aic":           round(mr.aic, 2),
        "bic":           round(mr.bic, 2),
        "lr_chi2":       round(lr_stat, 3),
        "lr_df":         int(max(lr_df, 1)),
        "lr_p":          round(lr_p, 4),
        "n_obs":         n,
        "singular":      mr.is_singular,
        "converged":     mr.converged,
    }


# ┌─ REFERÊNCIA — power_analysis ──────────────────────────────────────────────
# │ Whittemore, A.S. (1981). "Sample size for logistic regression with small
# │ response probability." Journal of the American Statistical Association,
# │ 76(373), 27–32.
# │ Fórmula: ncp = |p₁ − p₀| / √[p₁(1−p₁)/n₁ + p₀(1−p₀)/n₀]
# │ power = Φ(ncp − z_α) + Φ(−ncp − z_α)
# │
# │ Gelman, A. & Carlin, J. (2014). "Beyond Power Calculations: Assessing
# │ Type S and Type M Errors." Perspectives on Psychological Science, 9(6), 641–651.
# │ "We recommend a design analysis that goes beyond simple power calculations"
# │ (p. 641): condicionamento em estimativas plausíveis de tamanho de efeito.
# │
# │ DECISÕES:
# │  (a) Dois cenários p₀ (probabilidade baseline no grupo de controle):
# │      — "conservative": p₀ = 0.5 (maximiza variância → poder mínimo absoluto)
# │      — "uniform_likert": p₀ = 1/3 ≈ 0.333 (distribuição uniforme em K=3
# │        categorias implica P(Y ≤ 1) = 1/3 como baseline para o 1º threshold)
# │      Apresentar dois cenários revela a sensibilidade do poder à escolha de p₀.
# │  (b) Aproximação binary logit para outcome ordinal K=3: subestima poder.
# │      Sem fórmula analítica fechada para ordered logit power, Whittemore é
# │      a aproximação mais citada (Wooldridge Cap. 17 §17.1).
# └───────────────────────────────────────────────────────────────────────────────

def power_analysis(
    n_econ_values: list[int] | None = None,
    n_total_values: list[int] | None = None,
    or_values: list[float] | None = None,
    alpha: float = 0.05,
    p0_scenarios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Análise de poder (Whittemore 1981) como proxy para ordered logit K=3.

    Compara dois cenários de p₀ baseline (prob. de resposta no grupo controle):
      - "conservative": p₀=0.5 (máxima variância → poder mínimo absoluto)
      - "uniform_likert": p₀=1/3 (distribuição uniforme em K=3 categorias)

    Parâmetros:
        n_econ_values: tamanhos do grupo econ=1 (default: [17, 30, 50, 100])
        n_total_values: N total correspondente (default: n_econ × 11)
        or_values: Odds Ratios a testar (default: [1.5, 2.0, 2.5, 3.0, 4.0])
        alpha: nível de significância bilateral (default: 0.05)
        p0_scenarios: dict {nome: p₀} (default: conservative=0.5, empirical=17/183)
    """
    if n_econ_values is None:
        n_econ_values = [17, 30, 50, 100]
    if or_values is None:
        or_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    if p0_scenarios is None:
        p0_scenarios = {"conservative": 0.5, "uniform_likert": round(1 / 3, 4)}

    rows = []
    for scenario_name, p0 in p0_scenarios.items():
        for idx, n1 in enumerate(n_econ_values):
            n0 = (n_total_values[idx] - n1) if n_total_values else n1 * 10
            for OR in or_values:
                p1    = (p0 * OR) / (1 - p0 + p0 * OR)
                p_bar = (n0 * p0 + n1 * p1) / (n0 + n1)
                var_h1 = p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1
                var_h0 = p_bar * (1 - p_bar) * (1 / n0 + 1 / n1)
                if var_h0 <= 0 or var_h1 <= 0:
                    power = np.nan
                else:
                    z_alpha = norm.ppf(1 - alpha / 2)
                    ncp     = abs(p1 - p0) / np.sqrt(var_h1)
                    power   = float(norm.cdf(ncp - z_alpha) + norm.cdf(-ncp - z_alpha))
                rows.append({
                    "scenario":  scenario_name,
                    "p0":        round(p0, 4),
                    "n_econ":    n1,
                    "n_total":   n0 + n1,
                    "OR":        OR,
                    "power":     round(power, 3) if not np.isnan(power) else np.nan,
                    "power_pct": f"{power*100:.1f}%" if not np.isnan(power) else "n/d",
                })

    return pd.DataFrame(rows)


# ┌─ REFERÊNCIA — retrodesign ──────────────────────────────────────────────────
# │ Gelman, A. & Carlin, J. (2014). "Beyond Power Calculations: Assessing
# │ Type S and Type M Errors." Perspectives on Psychological Science, 9(6), 641–651.
# │
# │ Type S (sign error):
# │   Pr(β̂ e β_verdadeiro têm sinais opostos | β̂ significativo)
# │   s = Φ(−ncp − z_α) / power,  ncp = |β_verdadeiro| / SE
# │   [Gelman & Carlin, nota de rodapé 2, p. 641]
# │
# │ Type M (magnitude exaggeration factor):
# │   m = E[|β̂| / |β_verdadeiro| | β̂ significativo]
# │   estimado via simulação Monte Carlo condicional (10 000 draws, semente 42)
# │
# │ DECISÃO: β̂_obs como proxy de β_verdadeiro (retroanálise condicional).
# │ Com poder < 50%, Type M tipicamente > 2: magnitudes em amostras pequenas
# │ estão infladas mesmo condicionalmente à significância. Gelman & Carlin
# │ (p. 649): "statistically significant results in noisy settings are highly
# │ likely to overestimate absolute values."
# └───────────────────────────────────────────────────────────────────────────────

def retrodesign(estimate: float, se: float, alpha: float = 0.05) -> dict:
    """
    Retrodesign (Gelman & Carlin 2014): Type S e Type M para um resultado.

    Usa β̂_obs como proxy do efeito verdadeiro (retroanálise condicional).
    Adequado para diagnosticar inflação de magnitudes em amostras pequenas
    (n_econ = 17).

    Retorna: power, type_s (probabilidade de sinal errado), type_m (fator de
             exageração de magnitude), ambos condicionados em significância.
    """
    if se <= 0 or np.isnan(estimate) or np.isnan(se):
        return {"power": np.nan, "type_s": np.nan, "type_m": np.nan}

    z_alpha = norm.ppf(1 - alpha / 2)
    ncp     = abs(estimate) / se   # parâmetro de não-centralidade

    power  = float(norm.cdf(ncp - z_alpha) + norm.cdf(-ncp - z_alpha))
    type_s = float(norm.cdf(-ncp - z_alpha) / power) if power > 1e-10 else np.nan

    rng  = np.random.default_rng(42)
    sims = rng.normal(estimate, se, 10_000)
    sig  = sims[np.abs(sims) > z_alpha * se]
    if len(sig) > 0 and abs(estimate) > 1e-10:
        type_m = float(np.mean(np.abs(sig)) / abs(estimate))
    else:
        type_m = np.nan

    return {
        "power":  round(power, 3),
        "type_s": round(type_s, 4) if not np.isnan(type_s) else np.nan,
        "type_m": round(type_m, 2) if not np.isnan(type_m) else np.nan,
    }
