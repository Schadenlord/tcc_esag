"""data_loader.py — carregamento, limpeza e pré-processamento dos dados.

Especificação de controles (Wooldridge §17.1 + Angrist-Pischke cap.3):
  Incluem-se apenas as variáveis que afetam simultaneamente o tratamento
  (econ) e os desfechos, sem criar separação perfeita.

  Controles PRIMÁRIOS (usados nos modelos principais):
  1. espectro         — ideologia política (-2…+2); confundidor principal
  2. escolaridade_num — ordinal 0-6 (elimina separação perfeita das dummies)
  3. idade_num        — ordinal 0-6 (idem)
  4. genero           — binária (1=homem)
  5. trabalha_econ    — binária; captura exposição econômica profissional
                        sem formação formal (controla contaminação do grupo ctrl)

  Controles de SENSIBILIDADE (adicionados em análise secundária):
  6. engajado         — binária (1=sim); MEDIADOR POTENCIAL na cadeia
                        econ→engajado→opinião (Angrist-Pischke cap.3 pp.64-68).
                        Incluir aqui subestima o efeito total de econ; por isso
                        é mantido FORA do modelo primário e testado como
                        análise de sensibilidade.

  Dropados:
  - Dummies de escolaridade/idade (n=1/3/4 → separação perfeita; Wooldridge §17.1)
  - Dummies raciais (n=2 asiático; fraca justificativa teórica)
  - Vínculo empregatício (encoding inconsistente; fraca teoria)
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    GSHEET_URL,
    FORM_ECON_COL, TRABALHA_ECON_COL, ECON_MAP,
    IDEOL_MAP, ESCOLARIDADE_MAP, IDADE_MAP,
    ESPECTRO_COL, ECON_COL,
)

log = logging.getLogger(__name__)

_META_COLS = [
    "Carimbo de data/hora",
    'Ao clicar em "Aceito", você declara que compreendeu as informações e consente em participar.',
]

# Colunas de controle brutas (serão convertidas para numérico/ordinal)
_CONTROL_RAW = [
    FORM_ECON_COL,
    TRABALHA_ECON_COL,
    "Você é homem? ",
    "Com qual grupo racial/étnico você mais se identifica? ",
    "Qual seu nível de escolaridade? ",
    "Qual é a sua faixa etária? ",
    "Qual é o seu vínculo empregatício? ",
    "Você se considera uma pessoa politicamente engajada? ",
    ESPECTRO_COL,
]

_SKIP_COLS = [
    "Em qual região do Brasil você reside? ",
    "Você trabalha com política ou áreas correlatas? ",
    "Você costuma acompanhar notícias sobre economia e política? ",
]


@dataclass
class DataBundle:
    df_all:               pd.DataFrame
    df_dependentes:       pd.DataFrame
    df_controles:         pd.DataFrame
    control_cols:         List[str]   # controles completos (inclui engajado — sensibilidade)
    primary_control_cols: List[str]   # controles primários (sem engajado — modelo principal)
    dependentes_cols:     List[str]
    acronimos:            Dict[str, str]
    N_econ:               int
    N_total:              int
    missingness:          pd.DataFrame


def _detect_number(x) -> float:
    """Extrai o código Likert do início da string de resposta.

    Formatos suportados:
      '1 - Discordo'        → 1   (dígito no início)
      '(2) Motivo principal' → 2   (dígito após parêntese)
      '10 - Texto'          → 10  (dois dígitos; min() daria 0 — bug evitado)

    Regex r'^\\s*\\(?(\\d+)': opcional '(' antes dos dígitos captura ambos os
    formatos sem extrair dígitos de anos/numerações no corpo do label.
    """
    if pd.isna(x):
        return np.nan
    m = re.match(r"^\s*\(?(\d+)", str(x))
    return float(m.group(1)) if m else np.nan


def _make_acronym(colname: str) -> str:
    stop = {
        "de","da","do","das","dos","a","o","as","os","em","na","no","nas","nos",
        "para","por","com","ao","à","às","e","ou","que","se","é","uma","você",
        "qual","sua","são","sobre","tem","há","ter","ser","foi","está","isso",
        "eles","elas","nós","vós","um","uns","umas","mais","muito","bem",
        "mal","já","não","sim","mas","pois","então","também","ainda","só",
    }
    clean = re.sub(r"[^\w\s]", " ", colname, flags=re.UNICODE)
    toks  = [t.lower() for t in clean.split() if t.lower() not in stop]
    return "_".join(toks[:3]) if len(toks) >= 2 else (toks[0] if toks else colname.lower())


def _resolve_duplicates(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
    return result


def load_data(url: str = GSHEET_URL) -> DataBundle:
    log.info("Baixando dados de %s", url)
    df_raw = pd.read_csv(url)
    log.info("Shape bruto: %s", df_raw.shape)
    df = df_raw.copy()

    # ── 1. Variável tratamento ────────────────────────────────────────────────
    econ_series = df[FORM_ECON_COL].map(ECON_MAP)
    N_econ  = int((econ_series == 1).sum())
    N_total = len(df)
    log.info("N_econ=%d  N_total=%d", N_econ, N_total)

    # ── 2. Controles ordinais e binários ──────────────────────────────────────
    ctrl = pd.DataFrame(index=df.index)

    # Tratamento
    ctrl[ECON_COL] = econ_series.astype(float)

    # Espectro: ordinal contínuo; Independente/Sem-opinião → NaN (não distorce escala)
    # NOTA: NaN aqui reduz o N efetivo por modelo. Verificar proporção por grupo.
    ctrl["espectro"] = df[ESPECTRO_COL].map(IDEOL_MAP).astype(float)

    # Escolaridade ordinal 0-6 (elimina separação perfeita de dummies com n=1)
    # Pressuposto de equidistância entre categorias — limitação declarada (Wooldridge §17.1)
    ctrl["escolaridade_num"] = (
        df["Qual seu nível de escolaridade? "].map(ESCOLARIDADE_MAP).astype(float)
    )

    # Faixa etária ordinal 0-6 (intervalos desiguais tratados como equidistantes — limitação)
    ctrl["idade_num"] = (
        df["Qual é a sua faixa etária? "].map(IDADE_MAP).astype(float)
    )

    # Gênero binário
    ctrl["genero"] = (df["Você é homem? "] == "Sim").astype(float)

    # Exposição econômica profissional: controla contaminação do grupo ctrl por
    # financistas/contadores sem formação formal em Economia (Angrist-Pischke cap.3)
    ctrl["trabalha_econ"] = (df[TRABALHA_ECON_COL] == "Sim").astype(float)

    # Engajamento político: MEDIADOR POTENCIAL (econ→engajado→opinião).
    # Incluído no controle COMPLETO para análise de sensibilidade; excluído do
    # modelo primário para não subestimar o efeito total de econ.
    ctrl["engajado"] = (
        df["Você se considera uma pessoa politicamente engajada? "] == "Sim"
    ).astype(float)

    # primary_control_cols: sem engajado (modelo principal)
    primary_control_cols = [c for c in ctrl.columns if c != "engajado"]
    # control_cols: completo com engajado (análise de sensibilidade)
    control_cols = ctrl.columns.tolist()
    log.info("Controles primários (%d): %s", len(primary_control_cols), primary_control_cols)
    log.info("Controles sensibilidade (%d): %s", len(control_cols), control_cols)

    # ── 3. Variáveis dependentes ──────────────────────────────────────────────
    excluir = set(_META_COLS) | set(_CONTROL_RAW) | set(_SKIP_COLS)
    dep_raw = [c for c in df.columns if c not in excluir]
    df_dep  = df[dep_raw].apply(lambda s: s.apply(_detect_number)).astype("Int64")

    acron_raw  = [_make_acronym(c) for c in df_dep.columns]
    acron_uniq = _resolve_duplicates(acron_raw)
    acronimos  = dict(zip(acron_uniq, df_dep.columns))
    df_dep.columns = acron_uniq
    df_dep = df_dep.dropna(axis=1, how="all")
    dependentes_cols = df_dep.columns.tolist()

    # ── 4. Missingness report ─────────────────────────────────────────────────
    rows = []
    for dv in dependentes_cols:
        n_modelo = df_dep[[dv]].join(ctrl).dropna().shape[0]
        rows.append({"DV": dv,
                     "N_bruto":     len(df_dep),
                     "N_DV_valido": int(df_dep[dv].notna().sum()),
                     "N_modelo":    n_modelo})
    df_miss = pd.DataFrame(rows)

    # ── 5. df_all ─────────────────────────────────────────────────────────────
    df_all = pd.concat([df_dep, ctrl], axis=1)

    log.info("DVs: %d  Controles: %d", len(dependentes_cols), len(control_cols))

    return DataBundle(
        df_all=df_all,
        df_dependentes=df_dep,
        df_controles=ctrl,
        control_cols=control_cols,
        primary_control_cols=primary_control_cols,
        dependentes_cols=dependentes_cols,
        acronimos=acronimos,
        N_econ=N_econ,
        N_total=N_total,
        missingness=df_miss,
    )
