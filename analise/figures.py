"""figures.py — geração de todas as figuras da análise v2."""
from __future__ import annotations
import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sem display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import (
    FIG_CF_DIR, FIG_FP_DIR, FIG_HM_DIR, FIG_DS_DIR,
    BLOCS, ECON_COL, ESPECTRO_COL, ALPHA,
)

log = logging.getLogger(__name__)

# Paleta de cores por bloc
BLOC_COLORS = {
    "H3-Antimercado":      "#e41a1c",
    "H4-Antiestrangeiro":  "#377eb8",
    "H5-Antitrabalho":     "#4daf4a",
    "H6-Pessimista":       "#984ea3",
    "H7-Ideologia":        "#ff7f00",
    "Percepção-Específica":"#a65628",
    "Outro":               "#999999",
}

for d in (FIG_CF_DIR, FIG_FP_DIR, FIG_HM_DIR, FIG_DS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _bloc_of(dv: str) -> str:
    for bloc, dvs in BLOCS.items():
        if dv in dvs:
            return bloc
    return "Outro"


# ─── 1. Gráficos contrafactuais (53 DVs) ─────────────────────────────────────

def plot_counterfactual(
    dv: str,
    media_e0: float,
    media_e1: float,
    media_cf: float,
    pergunta: str = "",
    y_min: float = 0.0,
    y_max: float = 2.0,
) -> None:
    # Ordem narrativa: Público (base) → CF (economistas sem formação) → Economistas
    # CF=NaN (DVs singulares): exibir como barra hachurada com legenda "N/D"
    cf_available = not np.isnan(media_cf)

    labels = ["Público\n(econ=0)", "Economistas\ns/ formação (CF)", "Economistas\n(econ=1)"]
    valores = [media_e0, media_cf, media_e1]
    colors  = ["#6baed6", "#fd8d3c", "#2171b5"]

    valid = [v if not np.isnan(v) else 0.0 for v in valores]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, valid, color=colors, edgecolor="white", height=0.55)

    # Hachura na barra CF quando não disponível (DVs Firth-Ridge)
    if not cf_available:
        bars[1].set_hatch("///")
        bars[1].set_facecolor("#dddddd")
        bars[1].set_edgecolor("#888888")

    for bar, val, lbl in zip(bars, valores, labels):
        if not np.isnan(val):
            ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9)
        elif "CF" in lbl:
            ax.text(0.05, bar.get_y() + bar.get_height() / 2,
                    "N/D (singular)", va="center", fontsize=8, color="#666666", style="italic")

    ax.set_xlim(0, max(y_max + 0.4, max(v for v in valid if v > 0) + 0.4))
    ax.set_xlabel("Média na escala de resposta (0–2)")
    ax.set_title(textwrap.fill(pergunta or dv, width=65), fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIG_CF_DIR / f"{dv}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_counterfactuals(df_sintese: pd.DataFrame, acronimos: dict) -> None:
    log.info("Gerando %d gráficos contrafactuais...", len(df_sintese))
    for _, r in df_sintese.iterrows():
        dv  = str(r["DV"])
        pergunta = acronimos.get(dv, dv)
        plot_counterfactual(
            dv=dv,
            media_e0=float(r.get("media_e0", np.nan)),
            media_e1=float(r.get("media_e1", np.nan)),
            media_cf=float(r.get("media_cf", np.nan)),
            pergunta=pergunta,
        )


# ─── 2. Forest plot — β_econ ─────────────────────────────────────────────────

def plot_forest(
    df: pd.DataFrame,
    beta_col: str = "beta_econ",
    lo_col: str   = "IC95_lo",
    hi_col: str   = "IC95_hi",
    p_col: str    = "p_econ_bh",
    p_nom_col: str = "p_econ",
    title: str    = r"Forest plot: $\hat{\beta}_{econ}$ com IC$_{95\%}$ BC$_0$",
    filename: str = "econ_forest.png",
    x_clip: float = 6.0,
    bh_note: str = "0/53 sobrevivem BH",
) -> None:
    df2 = df.dropna(subset=[beta_col]).copy()
    df2["bloc"] = df2["DV"].apply(_bloc_of)
    df2 = df2.sort_values([beta_col], ascending=True)

    n   = len(df2)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    y_pos = range(n)

    for i, (_, row) in enumerate(df2.iterrows()):
        beta = float(row[beta_col])
        lo   = float(row.get(lo_col, np.nan))
        hi   = float(row.get(hi_col, np.nan))
        p_bh  = float(row.get(p_col, 1.0))
        p_nom = float(row.get(p_nom_col, 1.0)) if p_nom_col in df2.columns else 1.0
        bloc  = str(row.get("bloc", "Outro"))
        color = BLOC_COLORS.get(bloc, "#999999")

        # Opacidade: cheio = nominalmente significativo; esmaecido = não
        # (0/53 passam BH, mas 5 têm p_nom<0.05 — esses recebem destaque visual)
        alpha_val = 0.90 if p_nom < ALPHA else 0.40
        msize     = 5.5  if p_nom < ALPHA else 4.0

        # Clipar ICs para não distorcer escala (ICs truncados indicados por seta)
        lo_plot = max(lo, -x_clip) if not np.isnan(lo) else np.nan
        hi_plot = min(hi,  x_clip) if not np.isnan(hi) else np.nan
        beta_plot = max(min(beta, x_clip), -x_clip)

        ax.errorbar(beta_plot, i,
                    xerr=[[beta_plot - lo_plot] if not np.isnan(lo_plot) else [0],
                          [hi_plot - beta_plot] if not np.isnan(hi_plot) else [0]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.2,
                    capsize=2.5, markersize=msize, alpha=alpha_val)

        # Seta indicando IC truncado
        if not np.isnan(lo) and lo < -x_clip:
            ax.annotate("", xy=(-x_clip, i), xytext=(-x_clip + 0.3, i),
                        arrowprops=dict(arrowstyle="<-", color=color, lw=0.8))
        if not np.isnan(hi) and hi > x_clip:
            ax.annotate("", xy=(x_clip, i), xytext=(x_clip - 0.3, i),
                        arrowprops=dict(arrowstyle="<-", color=color, lw=0.8))

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(-x_clip - 0.5, x_clip + 0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df2["DV"].tolist(), fontsize=6.5)
    ax.set_xlabel(r"$\hat{\beta}_{econ}$", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # Nota sobre destaque visual
    ax.text(0.01, 0.01,
            rf"Pontos escuros: $p_\mathrm{{nom}}<0{{,}}05$ ({bh_note})",
            transform=ax.transAxes, fontsize=7, color="#444444", va="bottom")

    # Legenda por bloc
    patches = [mpatches.Patch(color=c, label=b) for b, c in BLOC_COLORS.items()
               if b in df2["bloc"].values]
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              framealpha=0.7, ncol=2)

    plt.tight_layout()
    path = FIG_FP_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Forest plot salvo: %s", path.name)


def plot_forest_espectro(df_sintese: pd.DataFrame) -> None:
    plot_forest(
        df_sintese,
        beta_col="beta_esp",
        lo_col="__no_ic__",
        hi_col="__no_ic__",
        p_col="p_esp_bh",
        p_nom_col="p_esp",
        title=r"Forest plot: $\hat{\beta}_{espectro}$ (sem IC bootstrap; 37/53 passam BH)",
        filename="espectro_forest.png",
        x_clip=3.0,
        bh_note="37/53 sobrevivem BH",
    )


# ─── 3. Heatmap de p-valores ──────────────────────────────────────────────────

def plot_pval_heatmap(df_sintese: pd.DataFrame) -> None:
    dvs  = df_sintese["DV"].tolist()
    blocs = [_bloc_of(dv) for dv in dvs]

    p_econ = pd.to_numeric(df_sintese.get("p_econ", np.nan), errors="coerce").values
    p_esp  = pd.to_numeric(df_sintese.get("p_esp",  np.nan), errors="coerce").values

    # Converte p-valor → nível de significância (0=ns, 1=*, 2=**, 3=***)
    def sig_level(p_arr):
        levels = np.zeros(len(p_arr))
        levels[p_arr < 0.05]  = 1
        levels[p_arr < 0.01]  = 2
        levels[p_arr < 0.001] = 3
        levels[np.isnan(p_arr)] = -1
        return levels

    mat = np.column_stack([sig_level(p_econ), sig_level(p_esp)])
    n   = len(dvs)

    fig, ax = plt.subplots(figsize=(4, max(5, n * 0.22)))
    cmap = matplotlib.colors.ListedColormap(["#cccccc", "#fee8c8", "#fdbb84", "#e34a33", "#b30000"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$p_{econ}$", r"$p_{esp}$"], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(dvs, fontsize=6)
    ax.set_title("P-valores nominais por DV e preditor\n"
                 r"(p$_\mathrm{econ}$: 0/53 sobrevivem BH; p$_\mathrm{esp}$: 37/53)", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.set_ticklabels(["NaN", "n.s.", "*", "**", "***"])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = FIG_HM_DIR / "pval_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap salvo: %s", path.name)


def plot_bloc_heatmap(df_resumo: pd.DataFrame) -> None:
    """Heatmap resumo por bloc: proporção de significância econ/espectro."""
    blocs = df_resumo["Bloc"].tolist()

    def parse_frac(s: str) -> float:
        try:
            a, b = str(s).split("/")
            return float(a) / float(b) if float(b) > 0 else 0.0
        except Exception:
            return 0.0

    econ_nom = [parse_frac(r) for r in df_resumo.get("Econ_nom", ["0/0"] * len(blocs))]
    esp_nom  = [parse_frac(r) for r in df_resumo.get("Esp_nom", ["0/0"] * len(blocs))]
    econ_bh  = [parse_frac(r) for r in df_resumo.get("Econ_BH", ["0/0"] * len(blocs))]

    mat = np.column_stack([econ_nom, econ_bh, esp_nom])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Econ nom.", "Econ BH", "Esp nom."], fontsize=9)
    ax.set_yticks(range(len(blocs)))
    ax.set_yticklabels(blocs, fontsize=9)
    ax.set_title("Proporção de itens significativos por Bloc", fontsize=10)
    for i in range(len(blocs)):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="black" if mat[i, j] < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    path = FIG_HM_DIR / "bloc_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Bloc heatmap salvo: %s", path.name)


# ─── 4. Volcano plot ─────────────────────────────────────────────────────────

def plot_volcano(df_sintese: pd.DataFrame) -> None:
    betas = pd.to_numeric(df_sintese.get("beta_econ", np.nan), errors="coerce")
    pvals = pd.to_numeric(df_sintese.get("p_econ",    np.nan), errors="coerce")
    valid = ~(betas.isna() | pvals.isna())

    x = betas[valid].values
    y = -np.log10(pvals[valid].values + 1e-10)
    dvs = df_sintese["DV"].values[valid.values]
    blocs = [_bloc_of(dv) for dv in dvs]
    colors_pt = [BLOC_COLORS.get(b, "#999") for b in blocs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, c=colors_pt, alpha=0.7, s=40, edgecolors="none")

    sig_threshold = -np.log10(0.05)
    ax.axhline(sig_threshold, color="red", linewidth=0.8, linestyle="--",
               label=r"$p=0.05$")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")

    # Labels para pontos significativos
    for xi, yi, dv in zip(x, y, dvs):
        if yi > sig_threshold:
            ax.annotate(dv, (xi, yi), fontsize=6.5, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(r"$\hat{\beta}_{econ}$", fontsize=11)
    ax.set_ylabel(r"$-\log_{10}(p_\text{econ})$", fontsize=11)
    ax.set_title("Volcano plot — efeito de formação econômica", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=c, label=b) for b, c in BLOC_COLORS.items()
               if b in blocs]
    ax.legend(handles=patches, fontsize=7, loc="upper right", ncol=2)

    plt.tight_layout()
    path = FIG_FP_DIR / "volcano_econ.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Volcano plot salvo: %s", path.name)


# ─── 5. Distribuições empilhadas por grupo ────────────────────────────────────

def plot_distributions(
    df_all: pd.DataFrame,
    dependentes_cols: list[str],
    acronimos: dict,
    econ_col: str = ECON_COL,
) -> None:
    log.info("Gerando %d gráficos de distribuição...", len(dependentes_cols))
    for dv in dependentes_cols:
        df_dv = df_all[[dv, econ_col]].dropna()
        y = pd.to_numeric(df_dv[dv], errors="coerce")
        e = df_dv[econ_col]
        cats = sorted(y.dropna().unique())
        if len(cats) < 2:
            continue
        y0_counts = y[e == 0].value_counts(normalize=True).reindex(cats, fill_value=0)
        y1_counts = y[e == 1].value_counts(normalize=True).reindex(cats, fill_value=0)

        fig, ax = plt.subplots(figsize=(5, 3))
        x   = np.arange(len(cats))
        w   = 0.35
        ax.bar(x - w / 2, y0_counts.values, w, label="Público (econ=0)",
               color="#6baed6", edgecolor="white")
        ax.bar(x + w / 2, y1_counts.values, w, label="Economistas (econ=1)",
               color="#2171b5", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) for c in cats])
        ax.set_ylabel("Proporção")
        ax.set_title(textwrap.fill(acronimos.get(dv, dv), width=55), fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = FIG_DS_DIR / f"{dv}_dist.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)


# ─── 6. Power curve ───────────────────────────────────────────────────────────

def plot_power_curve(df_power: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    n_econ_vals = sorted(df_power["n_econ"].unique())
    colors_pw   = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_econ_vals)))

    for n, col in zip(n_econ_vals, colors_pw):
        sub = df_power[df_power["n_econ"] == n].sort_values("OR")
        ax.plot(sub["OR"], sub["power"], marker="o", label=f"$n_{{econ}}={n}$",
                color=col, linewidth=1.8, markersize=5)

    ax.axhline(0.80, color="gray", linewidth=0.8, linestyle="--", label="Poder = 80%")
    ax.set_xlabel("Odds Ratio")
    ax.set_ylabel("Poder estatístico")
    ax.set_title(r"Curva de poder para detecção do efeito de $\mathtt{econ}$ ($\alpha=0{,}05$)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIG_FP_DIR / "power_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Power curve salvo: %s", path.name)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def generate_all_figures(
    df_sintese: pd.DataFrame,
    df_bootstrap: pd.DataFrame,
    df_resumo_blocs: pd.DataFrame,
    df_power: pd.DataFrame,
    df_all: pd.DataFrame,
    dependentes_cols: list[str],
    acronimos: dict,
    skip_distributions: bool = False,
) -> None:
    log.info("=== Gerando figuras ===")

    log.info("Contrafactuais...")
    plot_all_counterfactuals(df_sintese, acronimos)

    log.info("Forest plot (econ)...")
    df_fp = df_sintese.merge(
        df_bootstrap[["DV", "IC95_lo", "IC95_hi"]],
        on="DV", how="left"
    ) if df_bootstrap is not None and not df_bootstrap.empty else df_sintese
    plot_forest(df_fp)

    log.info("Forest plot (espectro)...")
    plot_forest_espectro(df_sintese)

    log.info("Heatmap p-valores...")
    plot_pval_heatmap(df_sintese)

    log.info("Heatmap por bloc...")
    plot_bloc_heatmap(df_resumo_blocs)

    log.info("Volcano plot...")
    plot_volcano(df_sintese)

    log.info("Power curve...")
    plot_power_curve(df_power)

    if not skip_distributions:
        log.info("Distribuições empilhadas...")
        plot_distributions(df_all, dependentes_cols, acronimos)

    log.info("=== Figuras concluídas ===")
