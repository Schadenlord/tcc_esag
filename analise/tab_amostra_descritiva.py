"""tab_amostra_descritiva.py — Tabela descritiva da amostra e cruzamento econ × espectro.

Gera:
  1. tab_amostra_geral.tex  — perfil sociodemográfico da amostra
  2. tab_crosstab_econ_esp.tex  — cruzamento econ × espectro
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np
import pandas as pd

from config import CSV_DIR, TEX_DIR, ECON_COL, IDEOL_MAP, DECIMAL_SEP

CACHE_FILE = THIS_DIR / "outputs" / "cache" / "pipeline_cache.pkl"

ESP_LABELS = {-2: "Ext. esquerda", -1: "Esquerda", 0: "Centro", 1: "Direita", 2: "Ext. direita"}
ESP_ORDER  = [-2, -1, 0, 1, 2]


def _fmt(v, decimals=1) -> str:
    if pd.isna(v): return "--"
    s = f"{v:.{decimals}f}"
    return s.replace(".", DECIMAL_SEP)


def build_crosstab(bundle) -> pd.DataFrame:
    df = bundle.df_controles[["econ", "espectro"]].copy()
    df["espectro_num"] = pd.to_numeric(df["espectro"], errors="coerce")
    df["econ_num"]     = pd.to_numeric(df["econ"],     errors="coerce")
    # Remove NaN em espectro (Independente/Sem opinião)
    df_valid = df.dropna(subset=["espectro_num", "econ_num"])
    df_valid["esp_label"] = df_valid["espectro_num"].map(ESP_LABELS)

    ct = pd.crosstab(
        df_valid["esp_label"],
        df_valid["econ_num"].map({0: "Público geral (econ=0)", 1: "Economistas (econ=1)"}),
        margins=True,
        margins_name="Total",
    )
    # Reordena linhas
    ordered_labels = [ESP_LABELS[k] for k in ESP_ORDER if ESP_LABELS[k] in ct.index] + ["Total"]
    ct = ct.reindex(ordered_labels)
    # Adiciona % por coluna
    ct_pct = ct.div(ct.loc["Total"], axis=1) * 100
    return ct, ct_pct


def export_crosstab_latex(ct: pd.DataFrame, ct_pct: pd.DataFrame) -> str:
    lines = []
    lines.append(r"% Cruzamento econ × espectro político")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Distribuição do espectro político por grupo (econ=0 vs.\ econ=1).}")
    lines.append(r"\label{tab:crosstab_econ_esp}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\hline\hline")
    lines.append(r"\textbf{Espectro} & \textbf{Públ.\ geral} & \textbf{\%} & \textbf{Economistas} & \textbf{\%} & \textbf{Total} \\")
    lines.append(r"\hline")

    col_pub = "Público geral (econ=0)"
    col_eco = "Economistas (econ=1)"

    for label in [ESP_LABELS[k] for k in ESP_ORDER]:
        if label not in ct.index:
            continue
        n_pub = int(ct.loc[label, col_pub]) if col_pub in ct.columns else 0
        n_eco = int(ct.loc[label, col_eco]) if col_eco in ct.columns else 0
        p_pub = ct_pct.loc[label, col_pub] if col_pub in ct_pct.columns else np.nan
        p_eco = ct_pct.loc[label, col_eco] if col_eco in ct_pct.columns else np.nan
        tot   = n_pub + n_eco
        lines.append(
            f"{label} & {n_pub} & {_fmt(p_pub)}{r'\%'} & {n_eco} & {_fmt(p_eco)}{r'\%'} & {tot} \\\\"
        )

    # Linha de total
    t_pub = int(ct.loc["Total", col_pub]) if col_pub in ct.columns else 0
    t_eco = int(ct.loc["Total", col_eco]) if col_eco in ct.columns else 0
    lines.append(r"\hline")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{t_pub}}} & 100\\% & \\textbf{{{t_eco}}} & 100\\% & \\textbf{{{t_pub+t_eco}}} \\\\")

    lines.append(r"\hline\hline")
    lines.append(
        r"\multicolumn{6}{p{11cm}}{\footnotesize \textit{Nota:} "
        r"Respondentes que se declararam ``Independente'' ou ``Sem opinião'' não possuem "
        r"valor de espectro e foram excluídos desta tabela ($n_{\text{excl}} \approx$ "
        r"correspondente à diferença entre $N=184$ e o total acima). "
        r"Percentuais calculados dentro de cada coluna (grupo). "
        r"Fonte: elaboração do autor.}"
    )
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    print("Carregando cache...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    bundle = cache["bundle"]

    print("Construindo cruzamento econ × espectro...")
    ct, ct_pct = build_crosstab(bundle)
    print(ct)

    tex = export_crosstab_latex(ct, ct_pct)
    out_tex = TEX_DIR / "tab_crosstab_econ_esp.tex"
    out_tex.write_text(tex, encoding="utf-8")
    print(f"\nLaTeX salvo: {out_tex}")

    # Salva CSV também
    ct.to_csv(CSV_DIR / "tabela_crosstab_econ_esp.csv", sep=";")
    print(f"CSV salvo: {CSV_DIR / 'tabela_crosstab_econ_esp.csv'}")


if __name__ == "__main__":
    main()
