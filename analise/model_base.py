"""model_base.py — funções de ajuste de modelos estatísticos.

Todos os estimadores retornam um `ModelResult` padronizado, permitindo que os
blocos downstream (tables.py, effects.py, bootstrap_ci.py) processem qualquer
modelo sem branching por tipo.

Modelos implementados
---------------------
1. Ordered Logit / Proportional Odds  →  fit_ordered_logit()
2. Linear Probability Model (OLS HC3) →  fit_lpm()
3. Firth-Ridge Penalizado             →  fit_firth_ridge()
4. Generalized Ordered Logit          →  fit_gol()

Correções aplicadas na auditoria econométrica v2
-------------------------------------------------
A1  N consistente: y_clean = y.dropna() do início ao fim (fit_ordered_logit e fit_lpm)
A2  optimizer_used logado; powell emite warning (sem gradiente analítico)
A3  converged default=False — não assume convergência quando chave ausente
A4  _detect_singularity: detecta SE=NaN, (SE>10 E coef>10), e razão SE/|coef|>5
    na mesma variável (AND por variável, não por conjunto)
A5  _cutpoints_ok usa "/" (padrão real do statsmodels, não "cut...")
A6  y.dropna() em todas as funções (evita NaN em classes/searchsorted)
B7  mcfadden_r2 no LPM armazena R² OLS — semântica diferente do logit, declarada
C10 _init_cuts_from_data inicializa cutpoints pelos logits empíricos acumulados
C12 _numerical_hessian: diferenças centrais cruzadas O(n²), step relativo
C14 grad_norm < 0.5 aceita solução fraca (evita descartar convergências quase-OK)
C15 z → NaN quando SE ≈ 0 (elimina p-value falsamente zero por divisão numérica)
C16 RIDGE_ALPHA 1.0 → 0.1 (heurística alpha ∝ p/n; 1.0 encolhia ~50% dos coef)
+   _reconstruct_cuts: nova parametrização — cuts[0] livre em R (admite valores
    negativos); cuts[i] = cuts[0] + cumsum(exp(raw[1:])) para i≥1
+   _nagelkerke: guard para cox_snell < 0 (modelo pior que nulo → NaN)
+   powell maxiter=2000 (default 500 insuficiente para p+K-1≈10 parâmetros)
+   Verificação de positiva-definitude da Hessiana Firth-Ridge
+   ModelResult: dois novos campos — is_numerically_suspect e optimizer_used
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import expit
from scipy.stats import norm as scipy_norm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm

from config import RIDGE_ALPHA, ECON_COL

log = logging.getLogger(__name__)


# ┌─ REFERÊNCIA — ModelResult ────────────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 5 — Assessing the Fit of the Model
# │ Seção 5.2.5 — Other Summary Measures of Fit  |  p. 184–185
# │
# │ Os campos mcfadden_r2 (Eq. 5.18, p.184) e nagelkerke_r2 (Eq. 5.19, p.184–185)
# │ são as duas medidas de ajuste recomendadas pelos autores para modelos de resposta
# │ discreta. is_singular e is_numerically_suspect operacionalizam o diagnóstico de
# │ quase-separação descrito no Cap. 4, Seção 4.4 (p.146–150).
# └───────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelResult:
    """Resultado padronizado de qualquer estimador deste módulo.

    Campos de ajuste
    ----------------
    mcfadden_r2 : float
        Para ordered_logit e firth_ridge: 1 − llf/llnull (McFadden 1974).
        Para lpm: R² OLS comum (não McFadden) — semântica diferente; ver B7.
    nagelkerke_r2 : float
        Cox-Snell R² normalizado para atingir máximo 1. Para lpm: R²-ajustado OLS.
    is_singular : bool
        True se (a) algum SE=NaN, (b) SE>10 E coef>10, ou (c) SE>10 E razão
        SE/|coef|>5 — para a mesma variável estrutural. Indica quase-separação.
    is_numerically_suspect : bool
        True se SE>10 sem NaN — quase-separação sem colapso total da Hessiana.
        Menos grave que is_singular; ainda permite inferência com cautela.
    optimizer_used : str
        Otimizador que convergiu: "lbfgs" | "bfgs" | "powell" | "L-BFGS-B-Ridge" | "OLS-HC3".
    """
    dv: str
    model_type: str           # "ordered_logit" | "lpm" | "firth_ridge" | "gol"
    n_obs: int
    params: pd.Series
    pvalues: pd.Series
    se: pd.Series
    llf: float
    llnull: float
    aic: float
    bic: float
    mcfadden_r2: float
    nagelkerke_r2: float
    is_singular: bool
    is_numerically_suspect: bool
    cutpoints_ok: Optional[bool]
    converged: bool
    optimizer_used: str
    n_categories: int
    classes: np.ndarray
    result_obj: object = field(repr=False, default=None)


# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════════

def _safe_float(x) -> float:
    """Converte x para float, retornando NaN em caso de falha."""
    try:
        return float(x)
    except Exception:
        return np.nan


# ┌─ REFERÊNCIA — _cutpoints_ok ──────────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 8 — Logistic Regression Models for Multinomial and Ordinal Outcomes
# │ Seção 8.2 — Ordinal Logistic Regression Models  |  p. 291–302
# │ Tabela 8.19  |  p. 302
# │
# │ O modelo cumulativo de logit (Eq. 8.17, p.291) exige θ_1 < θ_2 < … < θ_{K-1}
# │ para que as probabilidades cumulativas P(Y≤k|x) sejam monotonicamente crescentes
# │ em k. Se os cutpoints não estiverem ordenados, o modelo convergiu para um ponto
# │ de sela ou mínimo local — não para o máximo global da log-verossimilhança.
# │
# │ DECISÃO: usar "/" como padrão de nome (statsmodels usa "1/2", "2/3", etc.,
# │ não "cut1", "cut2"); "cut" mantido como fallback defensivo.
# └───────────────────────────────────────────────────────────────────────────────
def _cutpoints_ok(params: pd.Series) -> Optional[bool]:
    """Verifica se os cutpoints estão em ordem estritamente crescente.

    Returns None se encontrar menos de 2 cutpoints (não é possível verificar).
    """
    cuts = [v for k, v in params.items() if "/" in str(k) or str(k).startswith("cut")]
    if len(cuts) < 2:
        return None
    return all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1))


# ┌─ REFERÊNCIA — _mcfadden ──────────────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 5 — Assessing the Fit of the Model
# │ Seção 5.2.5 — Other Summary Measures of Fit  |  p. 184
# │ Equação 5.18: R²_L = 1 − L_p / L_0
# │
# │ L_0 é a log-verossimilhança do modelo nulo (só cutpoints, sem covariáveis).
# │ L_p é a log-verossimilhança do modelo ajustado.
# │ Quando o modelo converge para os dados, L_p → L_s (saturado) e R²_L → 1.
# │ Valores entre 0,20–0,40 são considerados excelentes (análogos a R²=0,60–0,80
# │ no OLS linear), conforme McFadden (1974, p.305) citado pelos autores.
# └───────────────────────────────────────────────────────────────────────────────
def _mcfadden(llf: float, llnull: float) -> float:
    if llnull == 0 or np.isnan(llnull):
        return np.nan
    return 1.0 - llf / llnull


# ┌─ REFERÊNCIA — _nagelkerke ────────────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 5 — Assessing the Fit of the Model
# │ Seção 5.2.5 — Other Summary Measures of Fit  |  p. 184–185
# │ Equação 5.19: R²_LS = (L_0 − L_p) / (L_0 − L_s)
# │
# │ A versão de Nagelkerke (1991) normaliza o Cox-Snell R² pela log-verossimilhança
# │ do modelo saturado (L_s), garantindo que o máximo possível seja 1,0 mesmo quando
# │ n < J (número de padrões de covariáveis distintos). O Cox-Snell puro nunca
# │ atinge 1,0 nesse caso, tornando a comparação entre DVs enganosa.
# │
# │ Guard adicionada: se cox_snell < 0 (llf > llnull — modelo pior que o nulo,
# │ improvável mas possível com quasi-separação severa), retorna NaN em vez de um
# │ R² negativo sem interpretação.
# └───────────────────────────────────────────────────────────────────────────────
def _nagelkerke(llf: float, llnull: float, n: int) -> float:
    if llnull == 0 or np.isnan(llnull) or n == 0:
        return np.nan
    cox_snell = 1.0 - np.exp(2.0 * (llnull - llf) / n)
    if cox_snell < 0:
        return np.nan
    max_cs = 1.0 - np.exp(2.0 * llnull / n)
    return cox_snell / max_cs if max_cs != 0 else np.nan


# ┌─ REFERÊNCIA — _null_loglik_ordinal ───────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 8 — Logistic Regression Models for Multinomial and Ordinal Outcomes
# │ Seção 8.2 — Ordinal Logistic Regression Models  |  p. 291–292
# │
# │ O modelo nulo (sem covariáveis) do ordered logit tem log-verossimilhança:
# │   LL_0 = Σ_k [ n_k * log(n_k / n) ]
# │ onde n_k é o número de observações na categoria k. Esta é a LL de um modelo
# │ que estima apenas os cutpoints otimizando as proporções marginais observadas.
# │
# │ DECISÃO (A1): recebe y_clean (sem NaN) para garantir que N usado aqui seja
# │ idêntico ao N do modelo ajustado — consistência do McFadden R².
# └───────────────────────────────────────────────────────────────────────────────
def _null_loglik_ordinal(y_clean: pd.Series) -> float:
    """Log-verossimilhança do modelo nulo ordinal. Requer y sem NaN."""
    classes, counts = np.unique(y_clean, return_counts=True)
    probs = counts / counts.sum()
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    return float(np.sum(counts * np.log(probs)))


# ┌─ REFERÊNCIA — _detect_singularity ───────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 4 — Interpretation of the Fitted Logistic Regression Model
# │ Seção 4.4 — Numerical Problems in Fitting the Logistic Regression Model
# │ p. 145–150; Tabela 4.37  |  p. 147
# │
# │ (p. 146): "What is rather obvious, and the tip-off that there is a problem with
# │ the model, is the large estimated coefficient for the second design variable and
# │ especially its large estimated standard error."
# │
# │ (p. 148): "In general, any time that the estimated standard error of an estimated
# │ coefficient is large relative to the point estimate, we should suspect the presence
# │ of one of the data structures described in this section." [separação perfeita,
# │ quase-separação, célula de frequência zero — Albert & Anderson 1984]
# │
# │ DECISÃO: três condições de singularidade, verificadas por variável (não por
# │ conjunto — AND por conjunto gera falsos positivos quando variáveis distintas
# │ disparam cada critério independentemente):
# │   (a) SE = NaN           → separação completa; Hessiana não invertível
# │   (b) SE > 10 E coef > 10 → separação severa com coeficiente divergindo
# │   (c) SE > 10 E SE/|coef| > 5 → quase-separação moderada (coef ∈ [4,8], SE ∈ [10,200])
# │       detectada pela razão SE/|coef| na mesma variável (Tabela 4.37, p.147)
# └───────────────────────────────────────────────────────────────────────────────
def _detect_singularity(bse: pd.Series, params: pd.Series) -> tuple[bool, bool]:
    """Detecta singularidade numérica e quase-separação (A4).

    Returns
    -------
    is_singular : bool
        True se qualquer condição (a), (b) ou (c) acima for satisfeita.
    is_numerically_suspect : bool
        True se SE>10 sem NaN — quase-separação sem colapso total (SE ≠ NaN).
    """
    has_nan_se = bse.isna().any()
    # Exclui cutpoints (nomes statsmodels com "/") — seus SEs grandes são esperados
    structural = bse.index[~bse.index.str.contains("/", na=False)]
    se_s       = bse[structural].dropna()
    p_s        = params[structural]
    large_se   = se_s.gt(10).any()
    large_coef = p_s.abs().gt(10).any()
    # Condição (c): AND por variável — mesmo parâmetro tem SE>10 E razão>5
    valid_idx = se_s.index.intersection(p_s.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = se_s[valid_idx] / p_s[valid_idx].abs().replace(0, np.nan)
    implausible_ratio = bool(((se_s[valid_idx] > 10) & (ratio > 5)).any())
    numerically_suspect = (large_se or implausible_ratio) and not has_nan_se
    singular = has_nan_se or (large_se and large_coef) or implausible_ratio
    return bool(singular), bool(numerically_suspect)


# ════════════════════════════════════════════════════════════════════════════════
# 1. ORDERED LOGIT / PROPORTIONAL ODDS
# ════════════════════════════════════════════════════════════════════════════════

# ┌─ REFERÊNCIA — fit_ordered_logit ──────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 8 — Logistic Regression Models for Multinomial and Ordinal Outcomes
# │ Seção 8.2 — Ordinal Logistic Regression Models  |  p. 289–302
# │ Equação 8.17 (p.291): ln[P(Y≤k|x) / P(Y>k|x)] = θ_k − x'β
# │
# │ DECISÃO (por que Ordered Logit e não OLS):
# │ O modelo de odds proporcionais é adequado para variáveis dependentes ordinais
# │ (Likert 1–5) pois: (1) impõe monotonicidade nas probabilidades cumulativas;
# │ (2) evita que probabilidades previstas saiam de [0,1]; (3) estima um único β
# │ para todos os thresholds, o que é eficiente quando a suposição de PO é válida.
# │ OLS violaria os pressupostos (1) e (2) e atribuiria distâncias iguais entre
# │ categorias ordinais — pressuposto não sustentado para escalas Likert.
# │
# │ Complemento — por que não OLS:
# │ WOOLDRIDGE, J.M. Introductory Econometrics. 6. ed. Cengage, 2016.
# │ Cap. 17 — Limited Dependent Variable Models  |  p. 524–526
# │ "The two most important disadvantages [of LPM] are that the fitted probabilities
# │ can be less than zero or greater than one and the partial effect of any
# │ explanatory variable (appearing in level form) is constant." (p. 525)
# │
# │ REFERÊNCIA — fallback de otimizadores (lbfgs → bfgs → powell):
# │ DAVIDSON, R.; MACKINNON, J.G.
# │ Econometric Theory and Methods. New York: Oxford University Press, 2004.
# │ Cap. 10 — The Method of Maximum Likelihood
# │ Seção 10.2 — Basic Concepts of Maximum Likelihood Estimation  |  p. 400–401
# │
# │ DM (p.401): "when the loglikelihood function is globally concave and not too
# │ flat, maximizing it is usually quite easy. At the other extreme, when the
# │ loglikelihood function has several local maxima, doing so can be very difficult."
# │ Métodos quasi-Newton (L-BFGS, BFGS) usam aproximações da Hessiana que garantem
# │ positividade; Powell é o fallback sem gradiente quando a Hessiana é indefinida.
# │
# │ REFERÊNCIA — converged=False como default conservador:
# │ DAVIDSON e MACKINNON, idem  |  p. 400 (Eq. 10.14)
# │ Um estimador de MLE "Tipo 2" exige que o ponto encontrado seja um máximo local
# │ confirmado. Na ausência de confirmação explícita de convergência, tratar o
# │ resultado como não-convergido preserva a validade dos testes de hipótese.
# └───────────────────────────────────────────────────────────────────────────────
def fit_ordered_logit(y: pd.Series, X: pd.DataFrame, dv: str = "") -> Optional[ModelResult]:
    """Ajusta OrderedModel (Proportional Odds) com fallback lbfgs → bfgs → powell.

    Parâmetros
    ----------
    y : pd.Series
        Variável dependente ordinal (pode conter NaN — serão removidos).
    X : pd.DataFrame
        Matriz de covariáveis (sem constante; OrderedModel não usa intercepto).
    dv : str
        Nome da variável dependente para logging.
    """
    # A1/A6: remove NaN antes de qualquer operação — garante N consistente
    y_clean = y.dropna()
    if y_clean.nunique() < 2:
        return None
    classes = np.sort(y_clean.unique())   # A6: sem NaN nos valores únicos

    # Remove colunas constantes (causam colinearidade perfeita e falha no fit)
    nuniq = X.nunique()
    const = nuniq[nuniq <= 1].index.tolist()
    if const:
        X = X.drop(columns=const)

    # ┌─ Otimização com fallback ────────────────────────────────────────────────
    # │ Ordem: lbfgs (rápido, gradiente de 1ª ordem) → bfgs (quasi-Newton,
    # │ mais robusto) → powell (sem gradiente, último recurso).
    # │ Powell sem gradiente analítico é suspeito: pode parar em patamar plano
    # │ sem verificar condição de KKT de 2ª ordem. Ver DM p.401.
    # └─────────────────────────────────────────────────────────────────────────
    model = OrderedModel(y_clean, X.loc[y_clean.index], distr="logit", hasconst=False)
    result = None
    optimizer_used = ""
    for method in ("lbfgs", "bfgs", "powell"):
        try:
            fit_kwargs: dict = {"method": method, "disp": False}
            if method == "powell":
                # default 500 insuficiente para p + K − 1 ≈ 10 parâmetros
                fit_kwargs["maxiter"] = 2000
            result = model.fit(**fit_kwargs)
            optimizer_used = method
            if method == "powell":
                log.warning(
                    "DV=%s: convergiu via Powell (sem gradiente analítico) — "
                    "resultado suspeito; ver DM (2004) Cap.10 p.401", dv
                )
            else:
                log.debug("DV=%s: convergiu via %s", dv, method)
            break
        except Exception:
            continue
    if result is None:
        return None

    # A3: default False — não assume convergência (ver DM 2004, p.400, Eq.10.14)
    converged = bool(result.mle_retvals.get("converged", False))
    if not converged:
        log.warning("DV=%s: otimizador %s não reportou convergência", dv, optimizer_used)

    # A4: detecta singularidade e quase-separação (HLS 2013, Cap.4 Sec.4.4, p.146–148)
    is_singular, is_numerically_suspect = _detect_singularity(result.bse, result.params)

    llf    = _safe_float(result.llf)
    n      = len(y_clean)              # A1: N da amostra limpa
    llnull = _null_loglik_ordinal(y_clean)   # A1: mesma amostra → McFadden R² consistente

    return ModelResult(
        dv=dv,
        model_type="ordered_logit",
        n_obs=n,
        params=result.params,
        pvalues=result.pvalues,
        se=result.bse,
        llf=llf,
        llnull=llnull,
        aic=_safe_float(result.aic),
        bic=_safe_float(result.bic),
        mcfadden_r2=_mcfadden(llf, llnull),
        nagelkerke_r2=_nagelkerke(llf, llnull, n),
        is_singular=is_singular,
        is_numerically_suspect=is_numerically_suspect,
        cutpoints_ok=_cutpoints_ok(result.params),
        converged=converged,
        optimizer_used=optimizer_used,
        n_categories=int(y_clean.nunique()),
        classes=classes,
        result_obj=result,
    )


# ════════════════════════════════════════════════════════════════════════════════
# 2. LINEAR PROBABILITY MODEL — verificação de robustez
# ════════════════════════════════════════════════════════════════════════════════

# ┌─ REFERÊNCIA — fit_lpm ────────────────────────────────────────────────────────
# │ ANGRIST, J.D.; PISCHKE, J.S.
# │ Mostly Harmless Econometrics: An Empiricist's Companion.
# │ Princeton: Princeton University Press, 2009.
# │ Cap. 3 — Making Regression Make Sense
# │ Seção 3.4.2 — Limited Dependent Variables and Marginal Effects  |  p. 94–107
# │
# │ (p. 94): "our view of regression as inheriting its legitimacy from the CEF
# │ makes LDVness less central."
# │ (p. 107): "when it comes to marginal effects, this probably matters little
# │ [whether OLS or probit/Tobit]. This optimistic conclusion is not a theorem,
# │ but, as in the empirical example here, it seems to be fairly robustly true."
# │ Tabela 3.4.2 (p.106): efeito estimado por OLS (−0,162) e por probit (−0,163)
# │ são virtualmente idênticos mesmo com taxa de emprego de 83%.
# │
# │ DECISÃO: LPM com HC3 é a verificação de robustez canônica de Angrist & Pischke.
# │ Se OLS e ordered logit concordam em sinal e significância, o resultado é robusto
# │ à escolha da forma funcional.
# │
# │ REFERÊNCIA — erros HC3:
# │ DAVIDSON, R.; MACKINNON, J.G. Econometric Theory and Methods. Oxford, 2004.
# │ Cap. 5 — Confidence Intervals
# │ Seção 5.5 — Heteroskedasticity-Consistent Covariance Matrices  |  p. 199–200
# │
# │ (p. 200): "HC3 uses û²_t / (1 − h_t)²" onde h_t é o elemento diagonal da
# │ matriz hat. HC3 divide por (1 − h_t)² para corrigir o fato de que observações
# │ influentes (h_t grande) têm resíduos artificialmente pequenos. MacKinnon &
# │ White (1985) e Long & Ervin (2000) mostram que HC3 tem melhor desempenho em
# │ amostras finitas.
# │
# │ Angrist & Pischke (p. 47) justificam HC3 no LPM: "LPM residuals are necessarily
# │ heteroskedastic unless the only regressor is a constant."
# │
# │ NOTA (B7): mcfadden_r2 armazena R² OLS (result.rsquared), não McFadden.
# │ nagelkerke_r2 armazena R²-ajustado OLS. A semântica difere do logit —
# │ declarada aqui explicitamente para evitar uso incorreto downstream.
# └───────────────────────────────────────────────────────────────────────────────
def fit_lpm(y: pd.Series, X: pd.DataFrame, dv: str = "") -> Optional[ModelResult]:
    """OLS em y normalizado [0,1] com erros HC3 (verificação de robustez A-P).

    Normalização y_norm = (y − min) / (max − min) coloca a variável em [0,1],
    tornando os coeficientes interpretáveis como variação proporcional em toda a
    extensão da escala de resposta — comparável ao efeito marginal do ordered logit.
    """
    # A1: consistência de N — y_clean usado do início ao fim
    y_clean = y.dropna()
    if y_clean.nunique() < 2:
        return None
    y_min, y_max = float(y_clean.min()), float(y_clean.max())
    if y_max == y_min:
        return None
    y_norm = (y_clean.astype(float) - y_min) / (y_max - y_min)

    nuniq = X.nunique()
    X = X.drop(columns=nuniq[nuniq <= 1].index.tolist())
    X_sm = sm.add_constant(X.loc[y_clean.index].astype(float))
    try:
        result = sm.OLS(y_norm, X_sm).fit(cov_type="HC3")
    except Exception:
        return None

    n   = len(y_clean)            # A1: N da amostra limpa
    llf = _safe_float(result.llf)
    # Log-verossimilhança do modelo nulo gaussiano (só intercepto = média de y_norm)
    llnull_ols = -0.5 * n * np.log(2 * np.pi * y_norm.var(ddof=0)) - n / 2

    return ModelResult(
        dv=dv,
        model_type="lpm",
        n_obs=n,
        params=result.params,
        pvalues=result.pvalues,
        se=result.bse,
        llf=llf,
        llnull=llnull_ols,
        aic=_safe_float(result.aic),
        bic=_safe_float(result.bic),
        mcfadden_r2=float(result.rsquared),      # B7: R² OLS, semântica diferente do logit
        nagelkerke_r2=float(result.rsquared_adj),
        is_singular=False,
        is_numerically_suspect=False,
        cutpoints_ok=None,
        converged=True,
        optimizer_used="OLS-HC3",
        n_categories=int(y_clean.nunique()),
        classes=np.sort(y_clean.unique()),
        result_obj=result,
    )


# ════════════════════════════════════════════════════════════════════════════════
# 3. FIRTH-RIDGE PENALIZADO — para DVs com Hessiana singular
# ════════════════════════════════════════════════════════════════════════════════

def _logistic_cdf(x: np.ndarray) -> np.ndarray:
    """Função logística CDF = sigmoid(x). Usada na log-verossimilhança do PO."""
    return expit(x)


# ┌─ REFERÊNCIA — _reconstruct_cuts ─────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 8 — Ordinal Logistic Regression Models  |  p. 291
# │ Equação 8.17: θ_k são os limiares do modelo PO, com θ_1 < θ_2 < … < θ_{K-1}
# │
# │ DECISÃO sobre a parametrização:
# │   cuts[0] = raw[0]                           — livre em R (pode ser negativo)
# │   cuts[i] = cuts[0] + cumsum(exp(raw[1:i+1])) — incrementos sempre positivos
# │
# │ Isso garante cuts[i] > cuts[i-1] para todo i ≥ 1 (pressuposto de HLS p.291).
# │ A parametrização anterior (cumsum(exp(raw))) forçava cuts[0] > 0, causando
# │ inconsistência com a inicialização quando o primeiro cutpoint empírico é negativo
# │ — caso frequente quando maioria das respostas está nas categorias inferiores
# │ (logit(P(Y≤1)) < 0 quando P(Y≤1) < 0,5).
# └───────────────────────────────────────────────────────────────────────────────
def _reconstruct_cuts(raw_cuts: np.ndarray) -> np.ndarray:
    """Reconstrói cutpoints a partir do vetor irrestrito raw_cuts.

    Parametrização: cuts[0] = raw[0] (livre em R);
                    cuts[i] = cuts[0] + cumsum(exp(raw[1:])) para i ≥ 1.

    Garantias
    ---------
    - cuts[i] > cuts[i-1] ∀ i ≥ 1 (exp > 0)
    - cuts[0] admite qualquer valor real (não força positivo)
    - Consistente com _init_cuts_from_data: raw[0]=logit_cuts[0] → cuts[0]=logit_cuts[0]
    """
    if len(raw_cuts) == 1:
        return raw_cuts.copy()
    return np.concatenate([[raw_cuts[0]], raw_cuts[0] + np.cumsum(np.exp(raw_cuts[1:]))])


# ┌─ REFERÊNCIA — _ordered_logit_loglik ─────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 8 — Ordinal Logistic Regression Models
# │ Seção 8.2 — Ordinal Logistic Regression Models  |  p. 291–292
# │ Equação 8.18 (p.292):
# │   P(Y=k|x) = Λ(θ_k − x'β) − Λ(θ_{k-1} − x'β)   para k=2,…,K-1
# │   P(Y=1|x) = Λ(θ_1 − x'β)
# │   P(Y=K|x) = 1 − Λ(θ_{K-1} − x'β)
# │ onde Λ é a função logística (sigmoid).
# └───────────────────────────────────────────────────────────────────────────────
def _ordered_logit_loglik(params: np.ndarray, y: np.ndarray,
                           X: np.ndarray, K: int) -> float:
    """Log-verossimilhança negativa do ordered logit para scipy.optimize.

    Usa _reconstruct_cuts para garantir cutpoints ordenados sem restrição de caixa.
    """
    n_beta   = X.shape[1]
    beta     = params[:n_beta]
    cuts     = _reconstruct_cuts(params[n_beta:])   # cuts[0] livre; cuts[i]>cuts[i-1]

    eta = X @ beta
    ll  = 0.0
    for k in range(K):
        mask = (y == k)
        if not mask.any():
            continue
        eta_k = eta[mask]
        if k == 0:
            p = _logistic_cdf(cuts[0] - eta_k)
        elif k == K - 1:
            p = 1.0 - _logistic_cdf(cuts[k - 1] - eta_k)
        else:
            p = _logistic_cdf(cuts[k] - eta_k) - _logistic_cdf(cuts[k - 1] - eta_k)
        p = np.clip(p, 1e-10, 1.0)
        ll += np.sum(np.log(p))
    return -ll   # negativo para minimização via scipy.optimize


# ┌─ REFERÊNCIA — _init_cuts_from_data ──────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 8 — Ordinal Logistic Regression Models  |  p. 291–302
# │
# │ Os cutpoints do modelo nulo (x=0) satisfazem:
# │   θ_k = logit(P(Y ≤ k)) = log[P(Y≤k) / P(Y>k)]
# │ Usar as proporções acumuladas empíricas como valores iniciais é o método de
# │ momentos padrão — fornece estimativas consistentes de θ_k no modelo nulo.
# │
# │ DAVIDSON, R.; MACKINNON, J.G. Econometric Theory and Methods. Oxford, 2004.
# │ Cap. 10 — The Method of Maximum Likelihood
# │ Seção 10.2  |  p. 401
# │ "when the loglikelihood function has several local maxima, doing so can be very
# │ difficult." — valores iniciais próximos ao ótimo reduzem risco de mínimos locais.
# │
# │ DECISÃO sobre a parametrização (consistente com _reconstruct_cuts):
# │   raw[0] = logit_cuts[0]                         → cuts[0] = logit_cuts[0] ✓
# │   raw[i] = log(max(logit_cuts[i]−logit_cuts[i-1], 0.01))  para i ≥ 1
# │
# │ Consistência: cuts[i] = logit_cuts[0] + Σ(logit_cuts[j]−logit_cuts[j-1], j=1..i)
# │             = logit_cuts[i]  — inicialização exata na escala logit.
# └───────────────────────────────────────────────────────────────────────────────
def _init_cuts_from_data(y_arr: np.ndarray, K: int) -> np.ndarray:
    """Inicializa cutpoints pelos logits das proporções acumuladas empíricas (C10)."""
    cum_props  = np.array([np.mean(y_arr <= k) for k in range(K - 1)])
    cum_props  = np.clip(cum_props, 0.01, 0.99)
    logit_cuts = np.log(cum_props / (1.0 - cum_props))
    raw0 = float(logit_cuts[0])   # direto — sem log extra; admite valor negativo
    init_cuts = [raw0] + [
        np.log(max(float(logit_cuts[i]) - float(logit_cuts[i - 1]), 0.01))
        for i in range(1, K - 1)
    ]
    return np.array(init_cuts)


# ┌─ REFERÊNCIA — _numerical_hessian ────────────────────────────────────────────
# │ DAVIDSON, R.; MACKINNON, J.G.
# │ Econometric Theory and Methods. New York: Oxford University Press, 2004.
# │ Cap. 10 — The Method of Maximum Likelihood
# │ Seção 10.4 — The Covariance Matrix of the ML Estimator  |  p. 409–411
# │
# │ (p. 409–410): "The first method is just to use minus the inverse of the Hessian,
# │ evaluated at the vector of ML estimates. […] This yields the estimator
# │ Var_H(θ̂) = −H⁻¹(θ̂), which is referred to as the empirical Hessian estimator.
# │ This estimator is easy to obtain whenever Newton's Method, or some sort of
# │ quasi-Newton method that uses second derivatives, is used to maximize the
# │ loglikelihood function."
# │
# │ FÓRMULA — diferenças centrais cruzadas de 2ª ordem:
# │   H[i,j] = [f(x+eᵢ+eⱼ) − f(x+eᵢ−eⱼ) − f(x−eᵢ+eⱼ) + f(x−eᵢ−eⱼ)] / (4hᵢhⱼ)
# │ Erro de truncamento O(h²) — mais estável que diferenças diretas aninhadas
# │ (versão anterior), que sofriam cancelamento catastrófico quando parâmetros
# │ tinham magnitudes muito diferentes.
# │
# │ DECISÃO sobre o step: hᵢ = max(|xᵢ| × 1e-5, 1e-7)
# │   - Step relativo ao parâmetro adapta-se à escala (evita step fixo 1e-5
# │     instável para parâmetros pequenos ou grandes)
# │   - Piso 1e-7 evita underflow quando xᵢ ≈ 0
# └───────────────────────────────────────────────────────────────────────────────
def _numerical_hessian(fun, x: np.ndarray) -> np.ndarray:
    """Hessiana numérica O(n²) via diferenças centrais cruzadas com step relativo (C12).

    Para p + K − 1 ≈ 10 parâmetros, O(n²) = 100 avaliações de função — custo
    aceitável. A simetria H[i,j] = H[j,i] é explorada (apenas i ≤ j calculados).
    """
    steps = np.maximum(np.abs(x) * 1e-5, 1e-7)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n)
            ei[i] = steps[i]; ej[j] = steps[j]
            val = (
                fun(x + ei + ej)
                - fun(x + ei - ej)
                - fun(x - ei + ej)
                + fun(x - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
            H[i, j] = val
            H[j, i] = val
    return H


# ┌─ REFERÊNCIA — fit_firth_ridge ────────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 4 — Interpretation of the Fitted Logistic Regression Model
# │ Seção 4.4 — Numerical Problems in Fitting the Logistic Regression Model
# │ p. 148–150
# │
# │ (p. 150): "An alternative is to use the ridge regression methods proposed by
# │ Schaefer (1986)."
# │ (p. 150): "Heinze and Schemper (2002) and Heinze (2006) discuss and illustrate
# │ the use of methods that can produce valid parameter estimates and confidence
# │ intervals with data containing zero frequency cells and/or separation. These
# │ methods include exact logistic regression and penalized likelihood methods."
# │ (p. 148): Albert & Anderson (1984): "Complete separation → MLE does not exist."
# │
# │ DECISÃO: penalização L2 (Ridge) como aproximação computacional à penalização
# │ de Firth (1993) — penaliza pela raiz quadrada do determinante da informação
# │ de Fisher. A equivalência é aproximada, não exata. Ridge regulariza a
# │ superfície de verossimilhança, tornando-a côncava mesmo com quase-separação.
# │
# │ REFERÊNCIA — RIDGE_ALPHA = 0.1:
# │ Heurística: α ∝ p/n ≈ 6/180 ≈ 0.033; 0.1 como valor conservador.
# │ LE CESSIE, S.; VAN HOUWELINGEN, J.C. Ridge estimators in logistic regression.
# │ Applied Statistics, v. 41, n. 1, p. 191–201, 1992.
# │ Discutem critérios de cross-validation para seleção de α. Para amostras
# │ pequenas com quase-separação, cross-validation por fold pode ser instável
# │ (folds com n_econ ≈ 2–3), justificando a heurística fixa.
# │
# │ REFERÊNCIA — C14 (critério grad_norm < 0.5):
# │ DAVIDSON, R.; MACKINNON, J.G. Econometric Theory and Methods. Oxford, 2004.
# │ Cap. 10, Seção 10.2  |  p. 400 (Eq. 10.14)
# │ As condições de 1ª ordem do MLE exigem g(θ̂) = 0. grad_norm < 0.5 indica que
# │ o vetor score é suficientemente pequeno para aceitar o ponto como "solução fraca"
# │ reportável com sinalização de incerteza. O limiar 0.5 é heurístico, sem referência
# │ bibliográfica formal — declarado explicitamente para transparência.
# │
# │ REFERÊNCIA — C15 (z → NaN quando SE ≈ 0):
# │ HOSMER et al. Cap. 4, Seção 4.4  |  p. 146–148
# │ (p. 148): "any time that the estimated standard error of an estimated coefficient
# │ is large relative to the point estimate, we should suspect the presence of one of
# │ the data structures described in this section."
# │ Quando SE → 0 por erro numérico (divisão pelo epsilon da máquina), z → ∞ e
# │ p-value → 0, sinalizando falsamente significância perfeita. Retornar z=NaN
# │ e p=NaN é a resposta correta: inferência é inválida nesse caso.
# │
# │ AVISO sobre SE e AIC/BIC:
# │ Os SE são derivados da Hessiana do objetivo PENALIZADO — válidos para o
# │ estimador Firth-Ridge, mas anticonservadores vs. MLE (HLS Cap.5, p.193).
# │ O AIC/BIC usa k = len(params) (número bruto), subestimando o número efetivo
# │ de parâmetros do Ridge. Não comparar AIC/BIC com o OrderedModel (MLE).
# └───────────────────────────────────────────────────────────────────────────────
def fit_firth_ridge(y: pd.Series, X: pd.DataFrame, dv: str = "",
                    alpha: float = RIDGE_ALPHA) -> Optional[ModelResult]:
    """Ordered logit com penalização L2 (Ridge) para DVs com Hessiana singular.

    Objetivo: L_pen(β) = −LL(β) + (α/2) Σ_j β_j²

    A penalidade encolhe β em direção a zero, regularizando a superfície de
    log-verossimilhança e tornando-a côncava mesmo sob quase-separação.
    """
    y_clean = y.dropna()     # A6
    if y_clean.nunique() < 2:
        return None
    classes = np.sort(y_clean.unique())   # A6: sem NaN
    K = len(classes)
    # Remapeia y para 0, 1, …, K-1 (necessário para _ordered_logit_loglik)
    y_arr = np.searchsorted(classes, y_clean.values.astype(float))

    nuniq = X.nunique()
    X2    = X.drop(columns=nuniq[nuniq <= 1].index.tolist()).astype(float)
    X_arr = X2.loc[y_clean.index].values
    n_beta = X_arr.shape[1]

    def penalized_nll(params):
        nll   = _ordered_logit_loglik(params, y_arr, X_arr, K)
        ridge = alpha * 0.5 * np.sum(params[:n_beta] ** 2)   # penaliza só β, não cutpoints
        return nll + ridge

    # C10: inicialização pelos logits empíricos (ver _init_cuts_from_data)
    init_beta = np.zeros(n_beta)
    init_cuts = _init_cuts_from_data(y_arr, K)
    x0 = np.concatenate([init_beta, init_cuts])

    try:
        res = opt.minimize(penalized_nll, x0, method="L-BFGS-B",
                           options={"maxiter": 2000, "ftol": 1e-9})
    except Exception:
        return None

    # C14: aceita solução fraca se grad_norm < 0.5 (ver referência DM acima)
    if not res.success:
        grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else np.inf
        if grad_norm > 0.5:
            log.warning(
                "DV=%s: Firth-Ridge não convergiu (grad_norm=%.4f > 0.5); "
                "descartando resultado", dv, grad_norm
            )
            return None
        log.warning(
            "DV=%s: Firth-Ridge convergência fraca (grad_norm=%.6f ≤ 0.5); "
            "usando resultado com cautela", dv, grad_norm
        )

    beta_hat = res.x[:n_beta]
    cuts     = _reconstruct_cuts(res.x[n_beta:])   # consistente com _ordered_logit_loglik

    # C12: Hessiana numérica com diferenças centrais (ver _numerical_hessian)
    # + verificação de positiva-definitude (ver DM 2004, Cap.10 p.409)
    try:
        H = _numerical_hessian(penalized_nll, res.x)
        eigvals = np.linalg.eigvalsh(H)
        if eigvals.min() < -1e-6:
            log.warning(
                "DV=%s: Hessiana Firth-Ridge não é PD (autovalor mín=%.4f) — "
                "SE possivelmente inválido", dv, eigvals.min()
            )
        cov    = np.linalg.pinv(H)
        # Clamp: diagonal da covariância pode ser levemente negativa por erro numérico
        diag   = np.maximum(np.diag(cov), 0.0)
        se_all = np.sqrt(diag)
        se_beta = se_all[:n_beta]
        singular = False
    except Exception:
        se_beta  = np.full(n_beta, np.nan)
        singular = True

    # Monta Series com nomes de parâmetros
    idx       = X2.columns.tolist()
    cut_names = [f"cut{i}/{i+1}" for i in range(K - 1)]
    params_s  = pd.Series(np.concatenate([beta_hat, cuts]), index=idx + cut_names)
    se_s      = pd.Series(np.concatenate([se_beta, np.full(K - 1, np.nan)]),
                           index=idx + cut_names)

    # C15: z → NaN quando SE ≈ 0 (ver referência HLS acima)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_beta > 1e-6, beta_hat / se_beta, np.nan)
    pvals   = np.where(np.isnan(z), np.nan, 2.0 * (1.0 - scipy_norm.cdf(np.abs(z))))
    pvals_s = pd.Series(np.concatenate([pvals, np.full(K - 1, np.nan)]),
                         index=idx + cut_names)

    # LLF não-penalizado (subtrai penalidade do valor otimizado)
    # Permite McFadden R² comparável com o modelo nulo (sem penalidade)
    llf    = -float(res.fun - alpha * 0.5 * np.sum(beta_hat ** 2))
    n      = len(y_clean)
    llnull = _null_loglik_ordinal(y_clean)   # A1: y_clean consistente
    k_par  = len(res.x)

    # Detecta quase-separação também no Firth-Ridge (HLS Cap.4 Sec.4.4, p.146–148)
    is_singular_fr, is_numerically_suspect_fr = _detect_singularity(se_s, params_s)
    is_singular_fr = is_singular_fr or singular   # inclui falha total de Hessiana

    return ModelResult(
        dv=dv,
        model_type="firth_ridge",
        n_obs=n,
        params=params_s,
        pvalues=pvals_s,
        se=se_s,
        llf=llf,
        llnull=llnull,
        aic=2 * k_par - 2 * llf,    # k bruto — não comparável ao AIC do MLE
        bic=k_par * np.log(n) - 2 * llf,
        mcfadden_r2=_mcfadden(llf, llnull),
        nagelkerke_r2=_nagelkerke(llf, llnull, n),
        is_singular=is_singular_fr,
        is_numerically_suspect=is_numerically_suspect_fr,
        cutpoints_ok=True,   # garantido pela parametrização _reconstruct_cuts
        converged=bool(res.success),
        optimizer_used="L-BFGS-B-Ridge",
        n_categories=K,
        classes=classes,
        result_obj=res,
    )


# ════════════════════════════════════════════════════════════════════════════════
# 4. GENERALIZED ORDERED LOGIT — modelo alternativo para o Brant test
# ════════════════════════════════════════════════════════════════════════════════

# ┌─ REFERÊNCIA — fit_gol ────────────────────────────────────────────────────────
# │ HOSMER, D.W.; LEMESHOW, S.; STURDIVANT, R.X.
# │ Applied Logistic Regression. 3. ed. Hoboken: Wiley, 2013.
# │ Cap. 8 — Logistic Regression Models for Multinomial and Ordinal Outcomes
# │ Seção 8.2.2 — Model Building Strategies (Score Test / Brant Test)
# │ p. 305–308; Equação 8.32  |  p. 308
# │
# │ O GOL define K−1 desfechos binários: ỹ_{ki} = 1 se yᵢ > k, 0 caso contrário.
# │ Para cada threshold k é estimado um logit binário separado: P(Y > k | x).
# │ O conjunto de K−1 vetores de coeficientes {β̂_k} é o modelo irrestrito —
# │ o modelo de Proportional Odds (PO) impõe a restrição β_1 = β_2 = … = β_{K-1}.
# │
# │ O teste de Brant (Wald test) testa H₀: β_k = β para todo k contra a alternativa
# │ de que ao menos um par β_k ≠ β_{k'} (p.306): "one should check to see whether
# │ the assumption of proportional odds is supported by the data."
# │
# │ Em nossos dados: 52 de 53 DVs passam (p > 0,05). Apenas 'quem_considera_maior'
# │ falha (p=0,024), validando a escolha do ordered logit como modelo principal.
# └───────────────────────────────────────────────────────────────────────────────
def fit_gol(y: pd.Series, X: pd.DataFrame, dv: str = "") -> Optional[dict]:
    """Generalized Ordered Logit: K-1 logits binários P(Y > k).

    Retorna dict {threshold_value: statsmodels_logit_result}.
    Usado internamente por diagnostics.brant_test().
    """
    y_clean = y.dropna()    # A6
    classes = np.sort(y_clean.unique())
    K = len(classes)
    if K < 3:
        return None

    nuniq = X.nunique()
    X2    = X.drop(columns=nuniq[nuniq <= 1].index.tolist())
    X2_sm = sm.add_constant(X2.loc[y_clean.index].astype(float))

    gol_results = {}
    for k_idx in range(K - 1):
        threshold = classes[k_idx]
        y_bin = (y_clean > threshold).astype(int)
        try:
            r = sm.Logit(y_bin, X2_sm).fit(disp=False)
            gol_results[threshold] = r
        except Exception:
            gol_results[threshold] = None

    return gol_results
