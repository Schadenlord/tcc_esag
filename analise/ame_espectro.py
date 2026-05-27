"""ame_espectro.py — Calcula AME para `espectro` nos 10 itens de maior magnitude.

Método: diferença finita discreta.
  Para cada obs i: delta_P_j(s) = P(Y=j|X_i, esp=s+1) - P(Y=j|X_i, esp=s)
  AME_j(s) = média sobre i
  Reporta AME avaliado na transição de espectro médio (s → s+1), ou seja,
  avaliado no valor modal/mediana da amostra (s=0, centro).

Referência:
  Cameron & Trivedi (2005) Cap.15, p.501: AME via diferenças finitas para
  variável discreta; "∂p/∂x avaliado em valores representativos de x".
  Professor recomendou: primeira diferença discreta em s=0→s+1 e s=-1→s=0.
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np
import pandas as pd

from config import CSV_DIR, TEX_DIR, ECON_COL, DECIMAL_SEP

CACHE_FILE = THIS_DIR / "outputs" / "cache" / "pipeline_cache.pkl"
ESP_COL = "espectro"

# ── Top 10 itens por |beta_esp| (não singulares) ──────────────────────────────
TOP10_DVS = [
    "privatização_estatais_benéfica",
    "governo_atual_sabe",
    "governo_regulamenta_negócios",
    "mulheres_força_trabalho",
    "lucros_empresariais_ocorrem",
    "brasil_deveria_adotar",
    "governo_deve_intervir",
    "indústria_nacional_deve",
    "altos_executivos_ganham",
    "déficit_federal_grande",
]

LABEL_MAP = {
    "privatização_estatais_benéfica": "Privatização de estatais é benéfica",
    "governo_atual_sabe": "Governo atual sabe administrar a economia",
    "governo_regulamenta_negócios": "Governo regulamenta negócios demais",
    "mulheres_força_trabalho": "Mulheres na força de trabalho causa desemprego",
    "lucros_empresariais_ocorrem": "Lucros empresariais à custa dos trabalhadores",
    "brasil_deveria_adotar": "Brasil deveria adotar livre-comércio amplo",
    "governo_deve_intervir": "Governo deve intervir nos preços",
    "indústria_nacional_deve": "Indústria nacional deve ser protegida",
    "altos_executivos_ganham": "Executivos ganham mais do que merecem",
    "déficit_federal_grande": "O déficit federal é grande demais",
}


def _ame_espectro_discrete(result_obj, X: pd.DataFrame, s_from: float, s_to: float,
                           esp_col: str = ESP_COL) -> pd.Series:
    """AME via diferença finita discreta de espectro=s_from para s_to."""
    X1 = X.copy(); X1[esp_col] = s_to
    X0 = X.copy(); X0[esp_col] = s_from
    p1 = result_obj.predict(X1)
    p0 = result_obj.predict(X0)
    diff = p1 - p0
    ame = diff.mean(axis=0)
    classes = np.array([0, 1, 2])
    delta_y = float(np.sum(classes * ame))
    return pd.Series(
        list(ame) + [delta_y],
        index=["AME_cat0", "AME_cat1", "AME_cat2", "delta_Y"],
    )


def _build_X(bundle, dv: str):
    """Reconstrói a matriz X para um DV usando primary_control_cols (sem engajado)."""
    import pandas as pd
    ctrl_cols = bundle.primary_control_cols
    df_m = bundle.df_all[[dv] + ctrl_cols].dropna()
    y = pd.to_numeric(df_m[dv], errors="coerce").astype(float).dropna()
    X = df_m[ctrl_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.loc[y.index]
    nuniq = X.nunique()
    X = X.drop(columns=nuniq[nuniq <= 1].index)
    return y, X


def compute_ame_espectro(bundle, model_results) -> pd.DataFrame:
    rows = []

    for dv in TOP10_DVS:
        mr = model_results.get(dv)
        if mr is None or mr.result_obj is None or mr.model_type != "ordered_logit":
            print(f"  SKIP {dv}: sem modelo ordenado")
            continue

        # Reconstrói X a partir do bundle
        try:
            y, X = _build_X(bundle, dv)
        except Exception as e:
            print(f"  SKIP {dv}: erro ao construir X: {e}")
            continue

        # Encontra coluna de espectro no modelo (pode ter nome longo)
        esp_col_in_model = next(
            (c for c in mr.params.index
             if "espectro" in c.lower() or "Com qual" in c.lower()),
            None,
        )
        # Encontra coluna de espectro em X
        esp_col_in_X = next(
            (c for c in X.columns
             if "espectro" in c.lower() or "Com qual" in c.lower()),
            None,
        )
        if esp_col_in_model is None or esp_col_in_X is None:
            print(f"  SKIP {dv}: espectro não encontrado (model={esp_col_in_model}, X={esp_col_in_X})")
            continue

        beta_esp = float(mr.params[esp_col_in_model])
        se_esp   = float(mr.se[esp_col_in_model]) if esp_col_in_model in mr.se.index else np.nan
        p_esp    = float(mr.pvalues[esp_col_in_model]) if esp_col_in_model in mr.pvalues.index else np.nan

        beta_esp = float(mr.params[ESP_COL])
        se_esp   = float(mr.se[ESP_COL]) if ESP_COL in mr.se.index else np.nan
        p_esp    = float(mr.pvalues[ESP_COL]) if ESP_COL in mr.pvalues.index else np.nan

        # AME avaliado em transições s=0→+1 (centro→direita) e s=-1→0 (esq→centro)
        try:
            ame_centro_direita = _ame_espectro_discrete(mr.result_obj, X, 0, 1, esp_col=esp_col_in_X)
            ame_esq_centro     = _ame_espectro_discrete(mr.result_obj, X, -1, 0, esp_col=esp_col_in_X)
        except Exception as e:
            print(f"  ERRO {dv}: {e}")
            continue

        rows.append({
            "DV":              dv,
            "Label":           LABEL_MAP.get(dv, dv),
            "beta_esp":        round(beta_esp, 3),
            "se_esp":          round(se_esp, 3),
            "p_esp":           round(p_esp, 4) if not np.isnan(p_esp) else np.nan,
            # AME centro→direita
            "AME_cd_cat0":     round(float(ame_centro_direita["AME_cat0"]), 3),
            "AME_cd_cat1":     round(float(ame_centro_direita["AME_cat1"]), 3),
            "AME_cd_cat2":     round(float(ame_centro_direita["AME_cat2"]), 3),
            "delta_Y_cd":      round(float(ame_centro_direita["delta_Y"]), 3),
            # AME esquerda→centro
            "AME_ec_cat0":     round(float(ame_esq_centro["AME_cat0"]), 3),
            "AME_ec_cat1":     round(float(ame_esq_centro["AME_cat1"]), 3),
            "AME_ec_cat2":     round(float(ame_esq_centro["AME_cat2"]), 3),
            "delta_Y_ec":      round(float(ame_esq_centro["delta_Y"]), 3),
        })
        print(f"  OK {dv}: beta_esp={beta_esp:.3f}, ΔY_cd={ame_centro_direita['delta_Y']:.3f}")

    return pd.DataFrame(rows)


def _fmt(v, decimals=3) -> str:
    """Formata número com vírgula decimal (ABNT)."""
    if pd.isna(v):
        return "--"
    s = f"{v:.{decimals}f}"
    return s.replace(".", DECIMAL_SEP)


def _stars(p) -> str:
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


def export_latex(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"% AME do espectro político nos 10 itens de maior magnitude")
    lines.append(r"% Gerado automaticamente por ame_espectro.py")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Efeitos Marginais Médios (AME) do Espectro Político nos 10 Itens de Maior Magnitude.}")
    lines.append(r"\label{tab:ame_espectro_top10}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{p{5.2cm}rrrrrr}")
    lines.append(r"\hline\hline")
    lines.append(r" & \multicolumn{3}{c}{\textbf{Centro $\to$ Direita}} & \multicolumn{3}{c}{\textbf{Esq. $\to$ Centro}} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append(r"\textbf{Item (DV)} & \textbf{$\Delta\bar{Y}$} & \textbf{AME\textsubscript{P=2}} & \textbf{$\hat{\beta}$} & \textbf{$\Delta\bar{Y}$} & \textbf{AME\textsubscript{P=2}} & \textbf{SE} \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        label = row["Label"].replace("&", r"\&")
        beta  = _fmt(row["beta_esp"])
        se    = _fmt(row["se_esp"])
        p     = row["p_esp"]
        stars = _stars(p)
        dy_cd = _fmt(row["delta_Y_cd"])
        a2_cd = _fmt(row["AME_cd_cat2"])
        dy_ec = _fmt(row["delta_Y_ec"])
        a2_ec = _fmt(row["AME_ec_cat2"])
        lines.append(
            f"{label} & {dy_cd} & {a2_cd} & {beta}{stars} & {dy_ec} & {a2_ec} & {se} \\\\"
        )

    lines.append(r"\hline\hline")
    lines.append(r"\multicolumn{7}{p{14cm}}{\footnotesize \textit{Nota:} "
                 r"AME via primeira diferença finita discreta: $\Delta P(Y=j \mid \texttt{espectro}=s+1) - \Delta P(Y=j \mid \texttt{espectro}=s)$, "
                 r"médio sobre todos os $i$. Centro$\to$Direita: $s=0\to1$; Esq.$\to$Centro: $s=-1\to0$. "
                 r"$\Delta\bar{Y}$: variação na resposta esperada (escala 0--2). "
                 r"AME\textsubscript{P=2}: variação em P(concordo totalmente). "
                 r"$\hat{\beta}$: coeficiente logit ordenado. "
                 r"Estrelas: $p$-valor assintótico (*$p<0{,}05$; **$p<0{,}01$; ***$p<0{,}001$). "
                 r"Fonte: elaboração do autor.}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    print("Carregando cache...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)

    bundle        = cache["bundle"]
    model_results = cache["model_results"]

    print("Computando AME para espectro (top 10 DVs)...")
    df = compute_ame_espectro(bundle, model_results)

    # Salva CSV
    out_csv = CSV_DIR / "tabela_ame_espectro_top10.csv"
    df.to_csv(out_csv, sep=";", index=False)
    print(f"CSV salvo: {out_csv}")

    # Gera LaTeX
    tex = export_latex(df)
    out_tex = TEX_DIR / "tab_ame_espectro_top10.tex"
    out_tex.write_text(tex, encoding="utf-8")
    print(f"LaTeX salvo: {out_tex}")
    print("\nAmostra dos resultados:")
    print(df[["DV", "beta_esp", "delta_Y_cd", "delta_Y_ec"]].to_string(index=False))


if __name__ == "__main__":
    main()
