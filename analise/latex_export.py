"""latex_export.py — converte CSVs em tabelas LaTeX (booktabs + longtable).
Formatação ABNT: vírgula decimal, p-valores com estrelas, singulares com dag.
Padrão visual: longtable + p{Xcm} columns + minipage para notas de rodapé.
Tabelas largas (sintese_v2, descritiva) recebem wrapper landscape.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from config import TEX_DIR, ALPHA, PVAL_STARS, DECIMAL_SEP

TEX_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_float(x, dec: int = 3) -> str:
    """Formata float com vírgula decimal ABNT."""
    if pd.isna(x) or (isinstance(x, float) and np.isnan(x)):
        return "---"
    try:
        s = f"{float(x):.{dec}f}"
        return s.replace(".", DECIMAL_SEP)
    except Exception:
        return str(x)


def _fmt_pval(p, stars: bool = True) -> str:
    if pd.isna(p):
        return "---"
    try:
        p = float(p)
    except Exception:
        return str(p)
    s = ""
    for thresh, star in sorted(PVAL_STARS.items()):
        if p < thresh:
            s = star
            break
    fmt = f"$<$ 0{DECIMAL_SEP}001{s}" if p < 0.001 else f"{_fmt_float(p, 3)}{s}"
    return fmt


def _latex_escape(s: str) -> str:
    """Escapa caracteres especiais LaTeX (char-a-char para evitar double-escape)."""
    _MAP = {
        "\\": r"\textbackslash{}",
        "&":  r"\&",   "%": r"\%",   "$": r"\$",  "#": r"\#",
        "_":  r"\_\allowbreak{}",   "{": r"\{",   "}": r"\}",
        "~":  r"\textasciitilde{}",  "^": r"\^{}",
    }
    return "".join(_MAP.get(c, c) for c in s)


def _write_tex(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"  Gerado: {path.name}")


# ── 1. Tabela Síntese Completa (landscape, longtable, apêndice) ───────────────

def tex_sintese_completa(df: pd.DataFrame, label: str = "tab:sintese_v2") -> str:
    N = 11
    rows = []
    for _, r in df.iterrows():
        dv   = _latex_escape(str(r.get("DV", "")))
        bloc = _latex_escape(str(r.get("Bloc", "")))
        sing = r"$\dag$" if r.get("Singular", False) else ""
        rows.append(
            f"  {dv}{sing} & {bloc} & {int(r.get('N', 0))} & "
            f"{_fmt_float(r.get('beta_econ'))} & "
            f"{_fmt_pval(r.get('p_econ'))} & "
            f"{_fmt_pval(r.get('p_econ_bh'))} & "
            f"{_fmt_pval(r.get('p_econ_holm'))} & "
            f"{_fmt_float(r.get('OR'))} & "
            f"{_fmt_float(r.get('beta_esp'))} & "
            f"{_fmt_pval(r.get('p_esp'))} & "
            f"{_fmt_float(r.get('mcfadden'), 3)} \\\\"
        )
    body = "\n".join(rows)
    n_dvs = len(df)
    note = (
        rf"$\dag$ = Hessiana singular (Firth-Ridge, $\lambda=0{{,}}1$); "
        rf"*\,$p<0{DECIMAL_SEP}05$, **\,$p<0{DECIMAL_SEP}01$, "
        rf"***\,$p<0{DECIMAL_SEP}001$ (nominal). "
        r"BH = Benjamini-Hochberg; Holm = Holm-Bonferroni. "
        r"OR = \textit{odds ratio} (logit ordenado padrão; "
        r"para DVs singulares ver Tab.~Firth-Ridge). $R^2$ = McFadden."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begin{{landscape}}
\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{3.5cm}}%
  >{{\RaggedRight\arraybackslash}}p{{2.0cm}}%
  >{{\centering\arraybackslash}}p{{0.6cm}}%
  >{{\centering\arraybackslash}}p{{1.1cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.1cm}}%
  >{{\centering\arraybackslash}}p{{1.1cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{0.7cm}}}}
\caption{{Síntese completa dos modelos logit ordenados ({n_dvs} DVs)}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & \textbf{{Bloc}} & $N$ & $\hat{{\beta}}_\text{{econ}}$ & $p$ (nom.) & $p$ (BH) & $p$ (Holm) & OR & $\hat{{\beta}}_\text{{esp}}$ & $p_\text{{esp}}$ & $R^2$ \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & \textbf{{Bloc}} & $N$ & $\hat{{\beta}}_\text{{econ}}$ & $p$ (nom.) & $p$ (BH) & $p$ (Holm) & OR & $\hat{{\beta}}_\text{{esp}}$ & $p_\text{{esp}}$ & $R^2$ \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{18.0cm}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
\end{{landscape}}
"""


# ── 2. Tabela Síntese Resumo (portrait, top-N para corpo do texto) ────────────

def tex_sintese_resumo(df: pd.DataFrame, n_items: int = 20, label: str = "tab:sintese_resumo") -> str:
    N = 7
    df2 = df.copy()
    df2["p_econ_num"] = pd.to_numeric(df2.get("p_econ", np.nan), errors="coerce")
    df2 = df2.sort_values("p_econ_num").head(n_items)

    rows = []
    for _, r in df2.iterrows():
        dv   = _latex_escape(str(r.get("DV", "")))
        sing = r"$\dag$" if r.get("Singular", False) else ""
        rows.append(
            f"  {dv}{sing} & "
            f"{_fmt_float(r.get('beta_econ'))} & "
            f"{_fmt_pval(r.get('p_econ'))} & "
            f"{_fmt_pval(r.get('p_econ_bh'))} & "
            f"{_fmt_float(r.get('beta_esp'))} & "
            f"{_fmt_pval(r.get('p_esp'))} & "
            f"{_fmt_float(r.get('mcfadden'))} \\\\"
        )
    body = "\n".join(rows)
    n_total = len(df)
    note = (
        rf"$\dag$ = Hessiana singular (Firth-Ridge); "
        rf"*\,$p<0{DECIMAL_SEP}05$, **\,$p<0{DECIMAL_SEP}01$, "
        rf"***\,$p<0{DECIMAL_SEP}001$. "
        r"BH = Benjamini-Hochberg FDR. "
        rf"Ordenado por $p$ nominal crescente; {n_total} DVs no total."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}}}
\caption{{Resultados selecionados --- {n_items} DVs com menor $p$ nominal ($\alpha=0{DECIMAL_SEP}05$; {n_total} testes)}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & $\hat{{\beta}}_\text{{econ}}$ & $p$ (nom.) & $p$ (BH) & $\hat{{\beta}}_\text{{esp}}$ & $p_\text{{esp}}$ (nom.) & $R^2_\text{{McF.}}$ \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & $\hat{{\beta}}_\text{{econ}}$ & $p$ (nom.) & $p$ (BH) & $\hat{{\beta}}_\text{{esp}}$ & $p_\text{{esp}}$ (nom.) & $R^2_\text{{McF.}}$ \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 3. Tabela AME chave ────────────────────────────────────────────────────────

def tex_ame_chave(df: pd.DataFrame, label: str = "tab:ame_chave") -> str:
    cat_cols = [c for c in df.columns if c.startswith("AME_cat")]
    N = 4 + len(cat_cols)
    rows = []
    for _, r in df.iterrows():
        dv = _latex_escape(str(r.get("DV", "")))
        cats = " & ".join(_fmt_float(r.get(c, np.nan)) for c in cat_cols)
        rows.append(
            f"  {dv} & {_fmt_float(r.get('E_Y_econ0'))} & "
            f"{_fmt_float(r.get('E_Y_econ1'))} & "
            f"{_fmt_float(r.get('delta_Y'))} & {cats} \\\\"
        )
    cat_headers = " & ".join(f"AME cat{i}" for i in range(len(cat_cols)))
    body = "\n".join(rows)
    cat_col_spec = "".join(
        "  >{\\centering\\arraybackslash}p{1.4cm}%\n"
        for _ in cat_cols
    )
    n_dvs = len(df)
    note = (
        r"AME via diferenças finitas. "
        r"$\Delta\bar{Y} = \sum_j j \cdot \widehat{\text{AME}}_j$. "
        rf"Categorias em escala 0--2. {n_dvs} DVs reportados."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
{cat_col_spec}}}
\caption{{Efeitos Marginais Médios (AME) de \texttt{{econ}} para os itens-chave}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & $\bar{{Y}}_{{e=0}}$ & $\bar{{Y}}_{{e=1}}$ & $\Delta \bar{{Y}}$ & {cat_headers} \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & $\bar{{Y}}_{{e=0}}$ & $\bar{{Y}}_{{e=1}}$ & $\Delta \bar{{Y}}$ & {cat_headers} \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 4. Tabela Bootstrap ────────────────────────────────────────────────────────

def tex_bootstrap(df: pd.DataFrame, label: str = "tab:bootstrap_v2") -> str:
    N = 5
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("beta_mediana")):
            continue
        dv = _latex_escape(str(r.get("DV", "")))
        ic = (f"[{_fmt_float(r.get('IC95_lo'))};\\;"
              f"{_fmt_float(r.get('IC95_hi'))}]")
        rows.append(
            f"  {dv} & {int(r.get('B_valid', 0))} & "
            f"{_fmt_float(r.get('beta_mediana'))} & "
            f"{ic} & "
            f"{_fmt_pval(r.get('p_boot'))} \\\\"
        )
    body = "\n".join(rows) if rows else rf"  \multicolumn{{{N}}}{{c}}{{---}} \\"
    n_dvs = len(df)
    note = (
        r"Reamostras sem variação em \texttt{econ} descartadas. "
        r"$p_\text{boot} = 2\min(f_{<0},f_{>0})$ (teste de sinal). "
        r"BC$_0$ = bootstrap com correção de viés pela mediana "
        r"(Efron \& Tibshirani 1993, \S12.4). "
        rf"DVs singulares usam Firth-Ridge nas reamostras. {n_dvs} DVs reportados."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{4.0cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}}}
\caption{{Bootstrap BC$_0$ ($B=999$, semente 42) para $\hat{{\beta}}_\text{{econ}}$}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & $B_\text{{válido}}$ & $\tilde{{\beta}}_\text{{econ}}$ & IC$_{{95\%}}$ BC$_0$ & $p_\text{{boot}}$ \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & $B_\text{{válido}}$ & $\tilde{{\beta}}_\text{{econ}}$ & IC$_{{95\%}}$ BC$_0$ & $p_\text{{boot}}$ \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 5. Tabela GOF ─────────────────────────────────────────────────────────────

def tex_gof(df: pd.DataFrame, label: str = "tab:gof") -> str:
    N = 7
    rows = []
    for _, r in df.iterrows():
        dv = _latex_escape(str(r.get("DV", "")))
        rows.append(
            f"  {dv} & {int(r.get('n_obs', 0))} & "
            f"{_fmt_float(r.get('mcfadden_r2'))} & "
            f"{_fmt_float(r.get('nagelkerke_r2'))} & "
            f"{_fmt_float(r.get('aic'), 1)} & "
            f"{_fmt_float(r.get('bic'), 1)} & "
            f"{_fmt_pval(r.get('lr_p'))} \\\\"
        )
    body = "\n".join(rows)
    n_dvs = len(df)
    note = (
        r"$R^2_\text{McF}$ = McFadden; $R^2_\text{Nag}$ = Nagelkerke; "
        r"$p_\text{LR}$ = razão de verossimilhança. "
        rf"Todos os {n_dvs} DVs estimados."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{0.8cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}}}
\caption{{Métricas de ajuste dos modelos logit ordenados ({n_dvs} DVs)}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & $N$ & $R^2_\text{{McF}}$ & $R^2_\text{{Nag}}$ & AIC & BIC & $p_\text{{LR}}$ \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & $N$ & $R^2_\text{{McF}}$ & $R^2_\text{{Nag}}$ & AIC & BIC & $p_\text{{LR}}$ \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 6. Tabela Brant ────────────────────────────────────────────────────────────

def tex_brant(df: pd.DataFrame, label: str = "tab:brant") -> str:
    N = 6
    rows = []
    for _, r in df.iterrows():
        dv   = _latex_escape(str(r.get("DV", "")))
        po   = "Sim" if r.get("po_ok") is True else ("Não" if r.get("po_ok") is False else "---")
        bdf  = int(r.get('brant_df', 0)) if not pd.isna(r.get('brant_df')) else '---'
        rows.append(
            f"  {dv} & {int(r.get('N_cats', 0))} & "
            f"{_fmt_float(r.get('brant_chi2'))} & "
            f"{bdf} & "
            f"{_fmt_pval(r.get('brant_p'))} & {po} \\\\"
        )
    body = "\n".join(rows)
    n_dvs = len(df)
    note = (
        r"H$_0$: coeficientes idênticos entre thresholds (suposição de chances proporcionais). "
        r"Deseja-se $p > 0{,}05$. "
        r"$K$ = número de categorias da DV. "
        rf"{n_dvs} DVs testados."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{0.6cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{0.8cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.0cm}}}}
\caption{{Brant test --- suposição de chances proporcionais ({n_dvs} DVs)}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & $K$ & $\chi^2$ & df & $p$ & PO ok? \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & $K$ & $\chi^2$ & df & $p$ & PO ok? \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 7. Tabela Power Analysis ───────────────────────────────────────────────────

def tex_power(df: pd.DataFrame, label: str = "tab:power") -> str:
    N = 5
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"  {int(r.get('n_econ', 0))} & {int(r.get('n_total', 0))} & "
            f"{_fmt_float(r.get('OR'), 1)} & "
            f"{_fmt_float(r.get('power'))} & "
            f"{str(r.get('power_pct', '---')).replace('%', r'\%')} \\\\"
        )
    body = "\n".join(rows)
    note = (
        r"Aproximação de Whittemore (1981) para logit binário (conservadora para ordinal). "
        r"Cenário baseline $p_0=0{,}5$; $\alpha=0{,}05$ (bilateral)."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.8cm}}}}
\caption{{Análise de poder estatístico para $\hat{{\beta}}_\text{{econ}}$ ($\alpha=0{DECIMAL_SEP}05$)}}\label{{{label}}}\\
\toprule
$n_\text{{econ}}$ & $N$ & OR & Poder & Poder (\%) \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
$n_\text{{econ}}$ & $N$ & OR & Poder & Poder (\%) \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 8. Tabela Firth-Ridge ──────────────────────────────────────────────────────

def tex_firth(df: pd.DataFrame, label: str = "tab:firth") -> str:
    N = 5
    rows = []
    for _, r in df.iterrows():
        dv = _latex_escape(str(r.get("DV", "")))
        rows.append(
            f"  {dv} & "
            f"{_fmt_float(r.get('beta_econ_ridge'))} & "
            f"{_fmt_float(r.get('se_ridge'))} & "
            f"{_fmt_pval(r.get('p_ridge'))} & "
            f"{_fmt_float(r.get('mcfadden_ridge'))} \\\\"
        )
    body = "\n".join(rows)
    n_dvs = len(df)
    note = (
        r"Penalização L2 (Ridge, $\lambda=0{,}1$) via \texttt{scipy.optimize} "
        r"sobre log-verossimilhança do logit ordenado. "
        rf"Aplicado às {n_dvs} DVs com Hessiana singular (separação quase-perfeita). "
        r"OR não reportado como inferência formal (ver texto)."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}}}
\caption{{Estimativas penalizadas (Ridge, $\lambda=0{DECIMAL_SEP}1$) para DVs com Hessiana singular}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & $\hat{{\beta}}_\text{{econ}}$ & SE & $p$ & $R^2_\text{{McF}}$ \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & $\hat{{\beta}}_\text{{econ}}$ & SE & $p$ & $R^2_\text{{McF}}$ \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 9. Tabela Interação ────────────────────────────────────────────────────────

def tex_interacao(df: pd.DataFrame, label: str = "tab:interacao") -> str:
    N = 5
    df2 = df.dropna(subset=["beta_interacao"])
    rows = []
    for _, r in df2.iterrows():
        dv = _latex_escape(str(r.get("DV", "")))
        rows.append(
            f"  {dv} & "
            f"{_fmt_float(r.get('beta_interacao'))} & "
            f"{_fmt_pval(r.get('p_interacao'))} & "
            f"{_fmt_float(r.get('AME_econ_Q1'))} & "
            f"{_fmt_float(r.get('AME_econ_Q3'))} \\\\"
        )
    body = "\n".join(rows) if rows else rf"  \multicolumn{{{N}}}{{c}}{{---}} \\"
    n_valid = len(df2)
    note = (
        r"Interação \texttt{econ\_x\_espectro} adicionada ao modelo PO. "
        r"AME avaliado nos quartis 1 e 3 do espectro político. "
        rf"{n_valid} DVs com estimativa válida."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{2.5cm}}%
  >{{\centering\arraybackslash}}p{{2.5cm}}}}
\caption{{Efeitos de interação \texttt{{econ}} $\times$ espectro político ({n_valid} DVs com estimativa válida)}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & $\hat{{\beta}}_\text{{int}}$ & $p_\text{{int}}$ & AME$_\text{{econ}}$ (Q1 esp) & AME$_\text{{econ}}$ (Q3 esp) \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & $\hat{{\beta}}_\text{{int}}$ & $p_\text{{int}}$ & AME$_\text{{econ}}$ (Q1 esp) & AME$_\text{{econ}}$ (Q3 esp) \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 10. Tabela Permutation ─────────────────────────────────────────────────────

def tex_permutation(df: pd.DataFrame, label: str = "tab:permutation") -> str:
    N = 5
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("mwu_p")):
            continue
        dv = _latex_escape(str(r.get("DV", "")))
        rows.append(
            f"  {dv} & "
            f"{_fmt_pval(r.get('mwu_p'))} & "
            f"{_fmt_pval(r.get('mwu_p_bh'))} & "
            f"{_fmt_pval(r.get('ks_p'))} & "
            f"{_fmt_pval(r.get('perm_p'))} \\\\"
        )
    body = "\n".join(rows) if rows else rf"  \multicolumn{{{N}}}{{c}}{{---}} \\"
    n_dvs = len(df)
    note = (
        r"MWU = Mann-Whitney U (two-sided, dominância estocástica); "
        r"KS = Kolmogorov-Smirnov; Permut.\ = diferença de medianas ($B=5000$). "
        r"Testes \textit{marginais}: não controlam covariates. "
        rf"BH sobre família de {n_dvs} testes."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begingroup
\footnotesize
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{4.5cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}%
  >{{\centering\arraybackslash}}p{{1.4cm}}}}
\caption{{Testes não-paramétricos de diferença entre grupos (econ=0 vs.\ econ=1)}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & MWU $p$ & MWU $p$ (BH) & KS $p$ & Permut.\ $p$ \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & MWU $p$ & MWU $p$ (BH) & KS $p$ & Permut.\ $p$ \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{\linewidth}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
"""


# ── 11. Tabela Descritiva (landscape) ─────────────────────────────────────────

def tex_descritiva(df: pd.DataFrame, label: str = "tab:descritiva") -> str:
    N = 10
    rows = []
    for _, r in df.iterrows():
        dv    = _latex_escape(str(r.get("DV", "")))
        bloc  = _latex_escape(str(r.get("Bloc", "")))
        n_tot = int(r.get("N_total", 0))
        n_e0  = int(r.get("N_econ0", 0))
        n_e1  = int(r.get("N_econ1", 0))
        m_e0  = _fmt_float(r.get("media_e0"))
        m_e1  = _fmt_float(r.get("media_e1"))
        sd_e0 = _fmt_float(r.get("sd_e0"))
        sd_e1 = _fmt_float(r.get("sd_e1"))
        diff  = _fmt_float(r.get("diff_media"))
        rows.append(
            f"  {dv} & {bloc} & {n_tot} & {n_e0} & {n_e1} & "
            f"{m_e0} & {m_e1} & {sd_e0} & {sd_e1} & {diff} \\\\"
        )
    body = "\n".join(rows) if rows else rf"  \multicolumn{{{N}}}{{c}}{{---}} \\"
    n_dvs = len(df)
    note = (
        r"$e=0$ = público geral ($n\approx166$); $e=1$ = economistas ($n\approx17$); "
        r"$\Delta\bar{Y} = \bar{Y}_{e=1} - \bar{Y}_{e=0}$; escala 0--2. "
        rf"Médias observadas (não ajustadas). {n_dvs} DVs."
    )
    return rf"""
% Tabela gerada automaticamente por latex_export.py
\begin{{landscape}}
\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{longtable}}{{%
  >{{\RaggedRight\arraybackslash}}p{{3.5cm}}%
  >{{\RaggedRight\arraybackslash}}p{{2.0cm}}%
  >{{\centering\arraybackslash}}p{{0.6cm}}%
  >{{\centering\arraybackslash}}p{{0.6cm}}%
  >{{\centering\arraybackslash}}p{{0.6cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.2cm}}%
  >{{\centering\arraybackslash}}p{{1.0cm}}}}
\caption{{Estatísticas descritivas por DV e grupo ({n_dvs} DVs)}}\label{{{label}}}\\
\toprule
\textbf{{DV}} & \textbf{{Bloc}} & $N$ & $n_{{e=0}}$ & $n_{{e=1}}$ & $\bar{{Y}}_{{e=0}}$ & $\bar{{Y}}_{{e=1}}$ & $\sigma_{{e=0}}$ & $\sigma_{{e=1}}$ & $\Delta\bar{{Y}}$ \\
\midrule
\endfirsthead
\multicolumn{{{N}}}{{l}}{{\footnotesize \textit{{(continuação)}}}}\\\toprule
\textbf{{DV}} & \textbf{{Bloc}} & $N$ & $n_{{e=0}}$ & $n_{{e=1}}$ & $\bar{{Y}}_{{e=0}}$ & $\bar{{Y}}_{{e=1}}$ & $\sigma_{{e=0}}$ & $\sigma_{{e=1}}$ & $\Delta\bar{{Y}}$ \\
\midrule
\endhead
\midrule
\multicolumn{{{N}}}{{r}}{{\footnotesize \textit{{continua na próxima página}}}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\bottomrule
\end{{longtable}}
\par\smallskip\noindent\begin{{minipage}}{{15.5cm}}\footnotesize\sloppy
\noindent\textit{{\textbf{{Notas:}}}}\ {note}
\end{{minipage}}
\endgroup
\end{{landscape}}
"""


# ── Pipeline ──────────────────────────────────────────────────────────────────

def export_all_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Converte todos os DataFrames em arquivos .tex."""
    dispatch = {
        "sintese_v2":    (tex_sintese_completa, "tab:sintese_v2"),
        "sintese_resumo": (tex_sintese_resumo,  "tab:sintese_resumo"),
        "ame_chave":     (tex_ame_chave,        "tab:ame_chave"),
        "bootstrap_v2":  (tex_bootstrap,        "tab:bootstrap_v2"),
        "gof":           (tex_gof,              "tab:gof"),
        "brant":         (tex_brant,            "tab:brant"),
        "power":         (tex_power,            "tab:power"),
        "firth":         (tex_firth,            "tab:firth"),
        "interacao":     (tex_interacao,        "tab:interacao"),
        "permutation":   (tex_permutation,      "tab:permutation"),
        "descritiva":    (tex_descritiva,        "tab:descritiva"),
    }

    # Gera versão resumo da síntese se não existir como chave separada
    if "sintese_v2" in tables and "sintese_resumo" not in tables:
        tables["sintese_resumo"] = tables["sintese_v2"]

    for key, (func, label) in dispatch.items():
        if key not in tables:
            continue
        df = tables[key]
        if df is None or df.empty:
            continue
        content = func(df, label=label)
        _write_tex(TEX_DIR / f"tab_{key}.tex", content)

    print(f"\nTabelas LaTeX salvas em: {TEX_DIR}")
