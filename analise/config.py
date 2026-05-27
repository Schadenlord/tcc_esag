"""config.py — parâmetros globais da análise v2.
Edite aqui para alterar qualquer comportamento sem tocar nos outros módulos.
"""
import os
from pathlib import Path

# ── Dados ────────────────────────────────────────────────────────────────────
GSHEET_URL = "https://docs.google.com/spreadsheets/d/1kn6u7jnv-ktwiY0TMYxYWnQhr54dWOYSOYvrqf9W3WY/export?format=csv"

# ── Inferência ────────────────────────────────────────────────────────────────
ALPHA = 0.05               # nível de significância padrão
N_BOOTSTRAP = 999          # Davidson & MacKinnon (2004, cap.4): B≥999 para ICs bootstrap acurados
BOOTSTRAP_SEED = 42
N_PERMUTATIONS = 5000      # permutation test
RIDGE_ALPHA = 0.1          # penalização L2 para os 5 DVs singulares (Cameron & Trivedi Cap.23: alpha ∝ p/n ≈ 6/180; 0.1 defensável)

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
CSV_DIR    = OUTPUT_DIR / "tables" / "csv"
TEX_DIR    = OUTPUT_DIR / "tables" / "tex"
FIG_CF_DIR = OUTPUT_DIR / "figures" / "counterfactual"
FIG_FP_DIR = OUTPUT_DIR / "figures" / "forestplots"
FIG_HM_DIR = OUTPUT_DIR / "figures" / "heatmaps"
FIG_DS_DIR = OUTPUT_DIR / "figures" / "distributions"
LOG_DIR    = OUTPUT_DIR / "logs"
CACHE_DIR  = OUTPUT_DIR / "cache"

# ── Colunas-chave ─────────────────────────────────────────────────────────────
ECON_COL     = "econ"
ESPECTRO_COL = "Com qual espectro político você mais se identifica? "

# ── DVs que produzem Hessiana singular no modelo original ─────────────────────
SINGULAR_DVS = [
    "impostos_altos",
    "corte_impostos",
    "aumento_uso_tecnologia",
    "acordos_comerciais_outros",
    "acha_preços_combustíveis",
]

# ── Mapeamento DV → Bloc (H3-H7) ─────────────────────────────────────────────
BLOCS: dict[str, list[str]] = {
    "H3-Antimercado": [
        "impostos_altos",
        "deduções_demais_empresas",
        "governo_regulamenta_negócios",
        "empresas_lucram_demais",
        "altos_executivos_ganham",
        "corte_impostos",
        "mulheres_força_trabalho",
        "pessoas_poupam_bastante",
        "pessoas_dão_valor",
    ],
    "H4-Antiestrangeiro": [
        "gasto_ajuda_externa",
        "temos_imigrantes_demais",
        "produtos_importados_benéficos",
        "acordos_comerciais_outros",
        "acha_acordos_comerciais",
        "entrada_estrangeiros_mercado",
        "indústria_nacional_deve",
        "brasil_deveria_priorizar",
        "brasil_deveria_adotar",
    ],
    "H5-Antitrabalho": [
        "tecnologia_causa_demissões",
        "empresas_estão_enviando",
        "empresas_estão_reduzindo",
        "empresas_investem_suficiente",
        "aumento_uso_tecnologia",
        "redução_recente_postos",
        "acha_novos_postos",
        "automação_prejudica_mercado",
        "competição_entre_empresas",
        "educação_qualificação_profissional",
    ],
    "H6-Pessimista": [
        "produtividade_aumentando_devagar",
        "desigualdade_entre_ricos",
        "últimos_20_anos",
        "pensando_apenas_salários",
        "próximos_cinco_anos",
        "espera_geração_seus",
        "filhos_menos_30",
        "algumas_pessoas_dizem",
        "algumas_pessoas_dizem_2",
        "brasil_chance_virar",
    ],
    "H7-Ideologia": [
        "déficit_federal_grande",
        "seguridade_social_previdência",
        "mulheres_minorias_têm",
        "taxa_juros_selic",
        "governo_atual_sabe",
        "governo_deve_intervir",
        "reforma_previdência_necessária",
        "reforma_trabalhista_necessária",
        "reforma_tributária_necessária",
        "privatização_estatais_benéfica",
        "lucros_empresariais_ocorrem",
        "corrupção_principal_causa",
    ],
    "Percepção-Específica": [
        "quem_considera_maior",
        "acha_preços_combustíveis",
        "acha_presidente_pode",
    ],
}

# ── Itens de interesse especial (para AME detalhado e bootstrap) ─────────────
KEY_DVS = [
    "déficit_federal_grande",
    "gasto_ajuda_externa",
    "produtividade_aumentando_devagar",
    "produtos_importados_benéficos",
]

# ── Formatação LaTeX ──────────────────────────────────────────────────────────
PVAL_STARS = {0.001: "***", 0.01: "**", 0.05: "*"}  # limiar → estrelas
DECIMAL_SEP = ","   # ABNT usa vírgula

FORM_ECON_COL     = "Qual seu nível de formação em Ciências Econômicas? "
TRABALHA_ECON_COL = "Você trabalha com economia, finanças, contabilidade ou áreas correlatas? "

# ── Mapeamentos de controle ───────────────────────────────────────────────────

ECON_MAP = {
    "Não tenho formação em Economia": 0,
    "Graduação em Economia":          0,   # graduação ≠ formação profissional plena
    "Mestrado em Economia":           1,
    "Doutorado em Economia":          1,
}

# Espectro: escala ordinal contínua (-2 = extrema-esquerda … +2 = extrema-direita)
# "Independente" fica em NaN para não distorcer a escala esquerda-direita
IDEOL_MAP = {
    "Extrema esquerda": -2,
    "Esquerda":          -1,
    "Centro":             0,
    "Direita":            1,
    "Extrema direita":    2,
    "Independente":      float("nan"),   # sem posição ideológica clara
    "Sem opinião":       float("nan"),
}

# Escolaridade: ordinal 0–6 (evita dummies com n=1/3/4 — separação perfeita)
ESCOLARIDADE_MAP = {
    "Ensino Fundamental Incompleto": 0,
    "Ensino Fundamental Completo":   1,
    "Ensino Médio Incompleto":       2,
    "Ensino Médio Completo":         3,
    "Ensino Superior Incompleto":    4,
    "Ensino Superior Completo":      5,
    "Pós-graduação":                 6,
}

# Faixa etária: ordinal 0–6
IDADE_MAP = {
    "Até 18 anos":    0,
    "19 a 25 anos":   1,
    "26 a 35 anos":   2,
    "36 a 45 anos":   3,
    "46 a 55 anos":   4,
    "56 a 65 anos":   5,
    "66 anos ou mais":6,
}
