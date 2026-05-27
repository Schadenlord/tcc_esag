"""bootstrap_ci.py — Bootstrap BC₀ para beta_econ (intervalo de confiança).

┌─ REFERÊNCIA — visão geral ──────────────────────────────────────────────────
│ Efron & Tibshirani (1993) An Introduction to the Bootstrap, §12.4 (BC₀):
│   z₀ = Φ⁻¹(#{β*_b < β̂}/B) — correção de viés pela mediana bootstrap.
│   α_lo = Φ(2·z₀ + z_{α/2}),  α_hi = Φ(2·z₀ + z_{1−α/2})
│
│ Davidson & MacKinnon (2004) Econometric Theory and Methods, Cap.4 p.163-168:
│   B=999 satisfaz α(B+1) ∈ ℤ para α=0.05 (→ 0.05×1000=50): garante que o
│   p-valor bootstrap pode atingir exatamente α sem artefato de discretização.
│   Cap.4 p.165: BC₀ preferível ao percentil simples quando β̂ é enviesado.
│
│ NOTA BCa: BC₀ implementado (sem jackknife). BCa completo (Efron 1987, §14)
│   exige n=183 fits jackknife — factível, reservado para extensão futura.
│   BC₀ é boa aproximação quando o viés de aceleração (a) é pequeno, o que
│   ocorre tipicamente em modelos de regressão bem-especificados.
│
│ NOTA Firth-Ridge (5 DVs singulares): cada reamostra usa fit_firth_ridge()
│   em vez de fit_ordered_logit() — mantém consistência com o estimador do
│   modelo original (E22). Sem essa correção, reamostras com separação
│   quasi-perfeita convergem via MLE com coeficientes inflados, e reamostras
│   extremas (as que refletem o efeito mais intenso) tendem a falhar
│   convergência e são descartadas → distribuição bootstrap truncada nas caudas.
│
│ NOTA p_boot: 2·min(frac_neg, frac_pos) é um teste de sinal (sign test):
│   testa se a distribuição de β*_b é consistente com β > 0 ou β < 0.
│   Difere do p-valor bootstrap de DM eq.4.61, que centra a distribuição
│   em zero sob H₀. Válido como inferência direcional, não como p-valor
│   bootstrap clássico para H₀: β=0.
└─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import logging
from scipy.stats import norm

import numpy as np
import pandas as pd

from model_base import fit_ordered_logit, fit_firth_ridge
from config import N_BOOTSTRAP, BOOTSTRAP_SEED, ECON_COL

log = logging.getLogger(__name__)


def _beta_econ_from_sample(
    indices: np.ndarray,
    y_arr: np.ndarray,
    X_arr: np.ndarray,
    X_cols: list[str],
    econ_col: str = ECON_COL,
    use_firth: bool = False,
) -> float:
    """
    Ajusta modelo em uma reamostra bootstrap e retorna beta_econ.

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Davidson & MacKinnon (2004) Cap.4 p.155: cada reamostra bootstrap deve
    │   usar o mesmo estimador do modelo original — condição de consistência
    │   do bootstrap paramétrico/não-paramétrico.
    │
    │ NOTA use_firth: para os 5 DVs com separação quasi-perfeita, o estimador
    │   original é Firth-Ridge (fit_firth_ridge). Usar fit_ordered_logit nessas
    │   reamostras produziria coeficientes MLE potencialmente inflados e descarte
    │   seletivo de reamostras extremas → distribuição bootstrap truncada.
    └─────────────────────────────────────────────────────────────────────────
    """
    y_b = pd.Series(y_arr[indices])
    X_b = pd.DataFrame(X_arr[indices], columns=X_cols)
    if X_b[econ_col].nunique() < 2:
        return np.nan
    mr = fit_firth_ridge(y_b, X_b) if use_firth else fit_ordered_logit(y_b, X_b)
    if mr is None:
        return np.nan
    return float(mr.params.get(econ_col, np.nan))


def _bco_ci(betas_valid: np.ndarray, beta_obs: float,
            confidence: float = 0.95) -> tuple[float, float]:
    """BC₀ (bias-corrected percentil) CI para a estatística bootstrap.

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Efron & Tibshirani (1993) An Introduction to the Bootstrap, §12.4:
    │   z₀ = Φ⁻¹(#{β*_b < β̂} / B) — estima o viés via proporção de
    │   reamostras abaixo do estimador original.
    │   α_lo = Φ(2·z₀ + z_{α/2}),  α_hi = Φ(2·z₀ + z_{1−α/2})
    │   Clamp em [1e-6, 1−1e-6] evita z₀ = ±∞ quando β̂ é extremo.
    └─────────────────────────────────────────────────────────────────────────
    """
    alpha = 1 - confidence
    z_a   = norm.ppf(alpha / 2)
    z_b   = norm.ppf(1 - alpha / 2)

    frac_below = float((betas_valid < beta_obs).mean())
    frac_below = max(min(frac_below, 1 - 1e-6), 1e-6)  # evita ±inf
    z0 = float(norm.ppf(frac_below))

    p_lo = float(norm.cdf(2 * z0 + z_a))
    p_hi = float(norm.cdf(2 * z0 + z_b))

    lo = float(np.percentile(betas_valid, 100 * p_lo))
    hi = float(np.percentile(betas_valid, 100 * p_hi))
    return lo, hi


def bootstrap_beta_econ(
    y: pd.Series,
    X: pd.DataFrame,
    beta_obs: float,
    econ_col: str = ECON_COL,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    use_firth: bool = False,
) -> dict:
    """
    Bootstrap BC₀ para beta_econ em um único DV.

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Davidson & MacKinnon (2004) Cap.4 p.165: BC₀ preferível ao percentil
    │   simples quando o estimador é enviesado em amostras pequenas.
    │   Com n_econ=17 e n_total=183, viés do ordered logit em amostras
    │   bootstrap pode ser não-negligenciável.
    │
    │ p_boot = 2·min(frac_neg, frac_pos): teste de sinal sobre a distribuição
    │   bootstrap de β*_b — testa se a massa da distribuição está concentrada
    │   em um sinal consistente. NÃO equivale ao p-valor bootstrap de DM
    │   eq.4.61 (que centra a distribuição em zero sob H₀: β=0).
    │   Interpretar como p-valor direcional, não teste de hipótese clássico.
    └─────────────────────────────────────────────────────────────────────────

    Retorna dict:
        B_valid, beta_mediana, IC95_lo, IC95_hi,
        frac_neg, frac_pos, p_boot, method
    """
    y_arr  = y.values.astype(float)
    X_cols = X.columns.tolist()
    X_arr  = X.values.astype(float)
    n      = len(y_arr)
    rng    = np.random.default_rng(seed)

    betas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        b   = _beta_econ_from_sample(idx, y_arr, X_arr, X_cols, econ_col,
                                     use_firth=use_firth)
        betas.append(b)

    betas_arr   = np.array(betas, dtype=float)
    valid_mask  = ~np.isnan(betas_arr)
    betas_valid = betas_arr[valid_mask]
    B_valid     = int(valid_mask.sum())

    if B_valid < 10:
        log.warning("Bootstrap: apenas %d amostras válidas de %d", B_valid, n_resamples)
        return {
            "B_valid": B_valid, "beta_mediana": np.nan,
            "IC95_lo": np.nan, "IC95_hi": np.nan,
            "frac_neg": np.nan, "frac_pos": np.nan,
            "p_boot": np.nan, "method": "failed",
        }

    beta_med = float(np.median(betas_valid))
    frac_neg = float((betas_valid < 0).mean())
    frac_pos = float((betas_valid > 0).mean())
    p_boot   = float(2 * min(frac_neg, frac_pos))

    IC_lo, IC_hi = _bco_ci(betas_valid, beta_obs)

    return {
        "B_valid":      B_valid,
        "beta_mediana": round(beta_med, 4),
        "IC95_lo":      round(IC_lo, 4),
        "IC95_hi":      round(IC_hi, 4),
        "frac_neg":     round(frac_neg, 4),
        "frac_pos":     round(frac_pos, 4),
        "p_boot":       round(p_boot, 4),
        "method":       "BC0_percentile",
    }


def bootstrap_all_dvs(
    df_all: pd.DataFrame,
    dependentes_cols: list[str],
    control_cols: list[str],
    model_results: dict | None = None,
    econ_col: str = ECON_COL,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """
    Roda bootstrap_beta_econ para todos os DVs.

    ┌─ REFERÊNCIA ───────────────────────────────────────────────────────────
    │ Davidson & MacKinnon (2004) Cap.4 p.163: cada DV usa seed=BOOTSTRAP_SEED+i
    │   (i = índice do DV) para garantir sequências RNG independentes entre DVs.
    │   Seed idêntico para todos os DVs produziria reamostras correlacionadas,
    │   violando a premissa de independência entre estimativas bootstrap.
    │
    │ beta_obs=0.0 fallback: quando beta_obs não está disponível (model_results
    │   ausente ou DV não mapeado), usa beta_obs=0.0 → z₀ = Φ⁻¹(#{β*_b<0}/B).
    │   Com z₀≈0, BC₀ degrada para percentil puro — válido mas menos eficiente
    │   que BC₀ com β̂ correto. Matematicamente coerente (coef vs. coef),
    │   ao contrário de nanmedian(y) que mistura unidades.
    │
    │ model_results: dict {dv: ModelResult} — usado para (1) obter beta_obs
    │   para correção BC₀; (2) detectar DVs Firth-Ridge (model_type="firth_ridge")
    │   e usar fit_firth_ridge() nas reamostras (E22).
    └─────────────────────────────────────────────────────────────────────────
    """
    rows = []
    for i, dv in enumerate(dependentes_cols):
        log.info("Bootstrap [%d/%d] %s", i + 1, len(dependentes_cols), dv)
        df_model = df_all[[dv] + control_cols].dropna()
        y = pd.to_numeric(df_model[dv], errors="coerce").astype(float)
        X = df_model[control_cols].apply(pd.to_numeric, errors="coerce").astype(float)

        nuniq = X.nunique()
        X = X.drop(columns=nuniq[nuniq <= 1].index.tolist())

        if econ_col not in X.columns or y.nunique() < 2:
            row = {"DV": dv, "B_valid": 0, "beta_mediana": np.nan,
                   "IC95_lo": np.nan, "IC95_hi": np.nan,
                   "frac_neg": np.nan, "frac_pos": np.nan,
                   "p_boot": np.nan, "method": "skipped"}
        else:
            beta_obs  = np.nan
            use_firth = False
            if model_results is not None and dv in model_results:
                mr_ref = model_results[dv]
                if mr_ref is not None and hasattr(mr_ref, "params"):
                    beta_obs = float(mr_ref.params.get(econ_col, np.nan))
                use_firth = (
                    mr_ref is not None
                    and getattr(mr_ref, "model_type", "") == "firth_ridge"
                )

            if np.isnan(beta_obs):
                # z0=0 → BC0 degrada para percentil puro (sem correção de viés)
                beta_obs = 0.0
                log.debug("Bootstrap %s: beta_obs não disponível → usando 0.0 (percentil puro)", dv)

            row = bootstrap_beta_econ(y, X, beta_obs=beta_obs,
                                      econ_col=econ_col,
                                      n_resamples=n_resamples,
                                      seed=seed + i,
                                      use_firth=use_firth)
            row["DV"] = dv

        rows.append(row)

    df = pd.DataFrame(rows)
    cols_order = ["DV", "B_valid", "beta_mediana", "IC95_lo", "IC95_hi",
                  "frac_neg", "frac_pos", "p_boot", "method"]
    return df[[c for c in cols_order if c in df.columns]]
