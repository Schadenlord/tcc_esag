# analise_v2 — Pipeline Econométrico do TCC

> Análise completa e reproduzível para o TCC *Racionalidade Coletiva em Xeque*  
> UDESC/ESAG · Administração Pública · 2025

## Início Rápido

```bash
export GSHEET_URL="https://docs.google.com/spreadsheets/d/1kn6u7jnv-ktwiY0TMYxYWnQhr54dWOYSOYvrqf9W3WY/export?format=csv"
cd Textuais/analise_v2

python run_all.py --fast          # ~2 min, sem bootstrap
python run_all.py                 # ~15 min, análise completa
python run_all.py --compile-tcc   # + recompila o TCC
```

## O que este pipeline faz

| Módulo | O que faz |
|--------|-----------|
| `data_loader.py` | Carrega planilha Google Sheets, limpa, codifica, gera dummies |
| `model_base.py` | Logit ordenado, Ridge-Firth (singulares), LPM, GOL |
| `diagnostics.py` | Brant test, VIF, McFadden R², AIC, BIC, poder estatístico |
| `multiple_testing.py` | Bonferroni, Holm, BH, BY sobre 53 p-valores |
| `effects.py` | AME todos 53 DVs, Odds Ratios, counterfactual, interação econ×espectro |
| `bootstrap_ci.py` | Bootstrap BCa B=500 para β_econ (Davidson-McKinnon cap.4) |
| `permutation_tests.py` | Mann-Whitney U, KS, permutation test por DV |
| `tables.py` | Gera 13 CSVs em `outputs/tables/csv/` |
| `latex_export.py` | Converte CSVs em .tex booktabs/longtable prontos para `\input{}` |
| `figures.py` | Forest plots, heatmaps, volcano, counterfactual, power curve |
| `run_all.py` | Orquestra tudo, flags `--fast / --only / --compile-tcc` |

## Estrutura de saída

```
outputs/
├── tables/
│   ├── csv/   ← 13 CSVs de resultados
│   └── tex/   ← 11 .tex prontos para \input{}
├── figures/
│   ├── counterfactual/   ← 53 gráficos CF
│   ├── forestplots/      ← forest econ + espectro
│   ├── heatmaps/         ← heatmap p-valores + bloc + volcano
│   └── distributions/    ← stacked bars por DV
├── cache/     ← pickle do pipeline para --only
└── logs/      ← run_YYYYMMDD_HHMMSS.log
```

## Documentação completa

Leia [ANALISE_COMPLETA.md](ANALISE_COMPLETA.md) para:
- Decisões metodológicas com referências bibliográficas exatas
- Como interpretar cada tabela e figura
- Limitações e cuidados de interpretação
- Como integrar os .tex no TCC (`\input{}`)
