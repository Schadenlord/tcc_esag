"""run_all.py — Pipeline master da análise v2.

Uso:
    python run_all.py                        # tudo (bootstrap B=999 + permutation; ~110 min)
    python run_all.py --fast                 # sem bootstrap/permutation (~2 min)
    python run_all.py --only tables          # relê CSVs + re-exporta .tex (~3s)
    python run_all.py --only figures         # relê CSVs + regenera figuras (~35s)
    python run_all.py --only effects+tables  # re-roda efeitos + recria tabelas
    python run_all.py --only tables --no-cache  # força rebuild completo das tabelas
    python run_all.py --compile-tcc          # + tectonic tcc_pronto.tex no final

Variáveis de ambiente:
    GSHEET_URL   URL pública da planilha Google Sheets (export?format=csv)

Notas sobre o cache:
    O pickle em outputs/cache/pipeline_cache.pkl armazena bundle, model_results,
    diag, eff, df_boot e df_perm.  O benefício principal é evitar bootstrap (B=999)
    e permutation (N=5000), que somam ~100 min.  Com --only tables/figures, as
    tabelas são recarregadas dos CSVs (load_tables_from_csv), não re-executando
    nenhum modelo.  Use --no-cache para forçar rebuild completo.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve PYTHONPATH para importar módulos da pasta
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from config import (
    GSHEET_URL,
    OUTPUT_DIR,
    CSV_DIR,
    TEX_DIR,
    LOG_DIR,
    BLOCS,
    ECON_COL,
)
FIG_DIR = OUTPUT_DIR / "figures"

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_FILE = CACHE_DIR / "pipeline_cache.pkl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{ts}.log"

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("run_all")
    log.info("Log: %s", log_file)
    return log


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save_cache(data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache() -> dict | None:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# Etapas do pipeline
# ---------------------------------------------------------------------------

def step_load(log: logging.Logger, url: str):
    from data_loader import load_data
    log.info("=== ETAPA 1: Carregamento de dados ===")
    bundle = load_data(url)
    log.info(
        "Dados carregados: N_total=%d  N_econ=%d  DVs=%d  controles=%d",
        bundle.N_total, bundle.N_econ,
        len(bundle.dependentes_cols), len(bundle.primary_control_cols),
    )
    return bundle


def step_models(log: logging.Logger, bundle, fast: bool):
    """Roda modelo ordenado para todos os DVs; retorna dict de ModelResult.

    Usa primary_control_cols (sem engajado) como especificação principal.
    engajado é mediador potencial; incluí-lo subestimaria o efeito total de econ
    (Angrist-Pischke cap.3, pp.64-68). Análise de sensibilidade com engajado
    pode ser rodada substituindo primary_control_cols por control_cols abaixo.
    """
    from model_base import fit_ordered_logit, fit_firth_ridge
    from config import SINGULAR_DVS, RIDGE_ALPHA

    log.info("=== ETAPA 2: Modelos ordenados (%d DVs) ===", len(bundle.dependentes_cols))
    results: dict = {}
    t0 = time.time()

    for i, dv in enumerate(bundle.dependentes_cols, 1):
        df_m = bundle.df_all[[dv] + bundle.primary_control_cols].dropna()
        import pandas as pd, numpy as np
        y = pd.to_numeric(df_m[dv], errors="coerce").dropna().astype(float)
        X = df_m[bundle.primary_control_cols].loc[y.index].apply(pd.to_numeric, errors="coerce").astype(float)
        nuniq = X.nunique()
        X = X.drop(columns=nuniq[nuniq <= 1].index.tolist())

        if dv in SINGULAR_DVS:
            mr = fit_firth_ridge(y, X, dv=dv, alpha=RIDGE_ALPHA)
            model_tag = "firth_ridge"
        else:
            mr = fit_ordered_logit(y, X, dv=dv)
            model_tag = "ordered_logit"

        results[dv] = mr
        status = "OK" if mr is not None else "FALHOU"
        log.info("[%2d/%2d] %-40s  %s  %s", i, len(bundle.dependentes_cols), dv, model_tag, status)

    log.info("Modelos concluídos em %.1fs", time.time() - t0)
    return results


def step_diagnostics(log: logging.Logger, bundle, model_results: dict):
    """Brant test, VIF, GOF para todos os DVs."""
    from diagnostics import brant_test, compute_vif, goodness_of_fit, power_analysis
    import pandas as pd, numpy as np

    log.info("=== ETAPA 3: Diagnósticos ===")
    diag: dict = {"brant": {}, "vif": {}, "gof": {}, "power": None}

    for dv, mr in model_results.items():
        if mr is None:
            continue
        df_m = bundle.df_all[[dv] + bundle.primary_control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").dropna().astype(float)
        X = df_m[bundle.primary_control_cols].loc[y.index].apply(pd.to_numeric, errors="coerce").astype(float)
        nuniq = X.nunique()
        X = X.drop(columns=nuniq[nuniq <= 1].index.tolist())

        try:
            diag["brant"][dv] = brant_test(y, X)
        except Exception as e:
            log.warning("Brant [%s]: %s", dv, e)
            diag["brant"][dv] = None

        try:
            diag["vif"][dv] = compute_vif(X)
        except Exception as e:
            log.warning("VIF [%s]: %s", dv, e)

        try:
            diag["gof"][dv] = goodness_of_fit(mr)
        except Exception as e:
            log.warning("GOF [%s]: %s", dv, e)

    try:
        diag["power"] = power_analysis(n_econ_values=[bundle.N_econ, 30, 50, 100])
    except Exception as e:
        log.warning("Power analysis: %s", e)

    log.info("Diagnósticos concluídos.")
    return diag


def step_effects(log: logging.Logger, bundle, model_results: dict):
    """AME (todos DVs), ORs, counterfactual, interações."""
    from effects import (
        compute_ame_all,
        compute_odds_ratios,
        compute_counterfactual,
        compute_interaction,
    )
    import pandas as pd, numpy as np
    from config import ESPECTRO_COL

    log.info("=== ETAPA 4: Efeitos (AME, OR, CF, Interação) ===")
    eff: dict = {"ame": {}, "or": {}, "cf": {}, "inter": {}}

    for dv, mr in model_results.items():
        if mr is None:
            continue
        df_m = bundle.df_all[[dv] + bundle.primary_control_cols].dropna()
        y = pd.to_numeric(df_m[dv], errors="coerce").dropna().astype(float)
        X = df_m[bundle.primary_control_cols].loc[y.index].apply(pd.to_numeric, errors="coerce").astype(float)
        nuniq = X.nunique()
        X = X.drop(columns=nuniq[nuniq <= 1].index.tolist())

        try:
            eff["ame"][dv]  = compute_ame_all(mr, X)
        except Exception as e:
            log.warning("AME [%s]: %s", dv, e)

        try:
            eff["or"][dv]   = compute_odds_ratios(mr)
        except Exception as e:
            log.warning("OR [%s]: %s", dv, e)

        try:
            eff["cf"][dv]   = compute_counterfactual(y, X, mr)
        except Exception as e:
            log.warning("CF [%s]: %s", dv, e)

        try:
            if "espectro" in bundle.primary_control_cols and "espectro" in X.columns:
                eff["inter"][dv] = compute_interaction(
                    y, X, dv=dv, econ_col=ECON_COL, espectro_col="espectro"
                )
        except Exception as e:
            log.warning("Interação [%s]: %s", dv, e)

    log.info("Efeitos concluídos.")
    return eff


def step_bootstrap(log: logging.Logger, bundle, model_results: dict, fast: bool):
    if fast:
        log.info("=== ETAPA 5: Bootstrap — PULADA (--fast) ===")
        return None

    from bootstrap_ci import bootstrap_all_dvs
    from config import N_BOOTSTRAP
    log.info("=== ETAPA 5: Bootstrap BC₀ (B=%d) ===", N_BOOTSTRAP)
    t0 = time.time()
    df_boot = bootstrap_all_dvs(
        bundle.df_all,
        bundle.dependentes_cols,
        bundle.primary_control_cols,
        model_results=model_results,
    )
    log.info("Bootstrap concluído em %.1fs", time.time() - t0)
    return df_boot


def step_permutation(log: logging.Logger, bundle, fast: bool):
    if fast:
        log.info("=== ETAPA 6: Permutation — PULADA (--fast) ===")
        return None

    from permutation_tests import run_permutation_tests
    log.info("=== ETAPA 6: Testes de Permutação ===")
    t0 = time.time()
    df_perm = run_permutation_tests(
        bundle.df_all,
        bundle.dependentes_cols,
        bundle.primary_control_cols,
    )
    log.info("Permutações concluídas em %.1fs", time.time() - t0)
    return df_perm


def step_tables(log: logging.Logger, bundle, df_boot, df_perm, quick: bool = False):
    """Gera todos os CSVs e .tex.

    quick=True: lê CSVs já existentes sem re-executar modelos (~3s).
    quick=False: re-executa build_all_tables() do zero (~180s).
    """
    from tables import build_all_tables, load_tables_from_csv
    from latex_export import export_all_tables

    log.info("=== ETAPA 7: Tabelas (CSVs + .tex) ===")
    t0 = time.time()

    if quick:
        log.info("Modo rápido: carregando tabelas dos CSVs existentes...")
        tables = load_tables_from_csv()
        # Garante bootstrap/permutation atualizados se presentes no cache
        if df_boot is not None and "bootstrap_v2" not in tables:
            tables["bootstrap_v2"] = df_boot
        if df_perm is not None and "permutation" not in tables:
            tables["permutation"] = df_perm
    else:
        tables = build_all_tables(
            bundle=bundle,
            df_bootstrap=df_boot,
            df_permutation=df_perm,
        )

    export_all_tables(tables)
    log.info("Tabelas concluídas em %.1fs", time.time() - t0)
    return tables


def step_figures(log: logging.Logger, bundle, tables):
    """Gera todas as figuras."""
    from figures import generate_all_figures

    log.info("=== ETAPA 9: Figuras ===")
    t0 = time.time()

    try:
        generate_all_figures(
            df_sintese=tables.get("sintese_v2"),
            df_bootstrap=tables.get("bootstrap_v2"),
            df_resumo_blocs=tables.get("resumo_blocs"),
            df_power=tables.get("power"),
            df_all=bundle.df_all,
            dependentes_cols=bundle.dependentes_cols,
            acronimos=bundle.acronimos,
        )
    except Exception as e:
        log.error("Erro ao gerar figuras: %s", e, exc_info=True)

    log.info("Figuras geradas em %.1fs", time.time() - t0)


def step_compile_tcc(log: logging.Logger):
    """Roda tectonic para recompilar o TCC."""
    import subprocess

    tcc_root = THIS_DIR.parent.parent
    tex_main = tcc_root / "tcc_pronto.tex"
    if not tex_main.exists():
        log.warning("tcc_pronto.tex não encontrado em %s — skip compilação.", tcc_root)
        return

    log.info("=== ETAPA 10: Compilação TCC (tectonic) ===")
    t0 = time.time()
    cmd = ["tectonic", str(tex_main)]
    result = subprocess.run(cmd, cwd=str(tcc_root), capture_output=True, text=True)
    if result.returncode == 0:
        log.info("TCC compilado com sucesso em %.1fs", time.time() - t0)
    else:
        log.error("Tectonic falhou:\n%s", result.stderr[-2000:])


# ---------------------------------------------------------------------------
# Relatório final
# ---------------------------------------------------------------------------

def print_summary(log: logging.Logger, tables: dict, t_start: float):
    import os
    log.info("=" * 60)
    log.info("RESUMO FINAL")
    log.info("=" * 60)

    # CSVs gerados
    csvs = list(CSV_DIR.glob("*.csv")) if CSV_DIR.exists() else []
    log.info("CSVs gerados: %d", len(csvs))
    for f in sorted(csvs):
        log.info("  %s", f.name)

    # .tex gerados
    texs = list(TEX_DIR.glob("*.tex")) if TEX_DIR.exists() else []
    log.info(".tex gerados: %d", len(texs))

    # Figuras geradas
    figs = list(FIG_DIR.rglob("*.pdf")) + list(FIG_DIR.rglob("*.png")) \
        if FIG_DIR.exists() else []
    log.info("Figuras geradas: %d", len(figs))

    log.info("Tempo total: %.1fs", time.time() - t_start)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline de análise econométrica v2 do TCC"
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Pula bootstrap e permutation (~2 min em vez de ~15 min)",
    )
    p.add_argument(
        "--only", choices=["tables", "figures", "effects+tables"],
        help="Pula modelos e diagnósticos; usa cache de execução anterior",
    )
    p.add_argument(
        "--compile-tcc", action="store_true",
        help="Roda tectonic no final para recompilar o TCC",
    )
    p.add_argument(
        "--url", default=None,
        help="URL da planilha (sobrescreve GSHEET_URL e config.py)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Ignora cache existente e re-roda todos os modelos",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging(LOG_DIR)
    t_start = time.time()

    url = args.url or os.environ.get("GSHEET_URL", GSHEET_URL)
    if not url:
        log.error("GSHEET_URL não definido. Use --url ou exporte a variável de ambiente.")
        sys.exit(1)

    log.info("Pipeline analise_v2 iniciado. fast=%s only=%s compile=%s",
             args.fast, args.only, args.compile_tcc)

    # -----------------------------------------------------------------------
    # Modo --only: pula cálculo pesado e usa cache
    # -----------------------------------------------------------------------
    if args.only in ("tables", "figures", "effects+tables"):
        cache = _load_cache()
        if cache is None:
            log.error("Cache não encontrado em %s. Rode sem --only primeiro.", CACHE_FILE)
            sys.exit(1)
        bundle        = cache["bundle"]
        model_results = cache["model_results"]
        df_boot       = cache.get("df_boot")
        df_perm       = cache.get("df_perm")
        log.info("Cache carregado de %s", CACHE_FILE)

        if args.only == "effects+tables":
            # Re-roda efeitos (útil após corrigir effects.py) e salva cache
            eff = step_effects(log, bundle, model_results)
            _save_cache({**cache, "eff": eff})
            # effects+tables sempre reconstrói tabelas do zero
            tables = step_tables(log, bundle, df_boot, df_perm, quick=False)
        else:
            # --no-cache força rebuild completo; caso contrário usa CSVs existentes
            quick = not args.no_cache
            tables = step_tables(log, bundle, df_boot, df_perm, quick=quick)
            if args.only == "figures":
                step_figures(log, bundle, tables)
        if args.compile_tcc:
            step_compile_tcc(log)
        print_summary(log, tables, t_start)
        return

    # -----------------------------------------------------------------------
    # Pipeline completo
    # -----------------------------------------------------------------------
    bundle        = step_load(log, url)
    model_results = step_models(log, bundle, fast=args.fast)
    diag          = step_diagnostics(log, bundle, model_results)
    eff           = step_effects(log, bundle, model_results)
    df_boot       = step_bootstrap(log, bundle, model_results, fast=args.fast)
    df_perm       = step_permutation(log, bundle, fast=args.fast)

    # Salva cache para --only
    _save_cache({
        "bundle":        bundle,
        "model_results": model_results,
        "diag":          diag,
        "eff":           eff,
        "df_boot":       df_boot,
        "df_perm":       df_perm,
    })
    log.info("Cache salvo em %s", CACHE_FILE)

    tables = step_tables(log, bundle, df_boot, df_perm, quick=False)
    step_figures(log, bundle, tables)

    if args.compile_tcc:
        step_compile_tcc(log)

    print_summary(log, tables, t_start)


if __name__ == "__main__":
    main()
