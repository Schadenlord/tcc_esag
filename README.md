# Racionalidade Coletiva em Xeque
### Revisão Pós-Entrega — UDESC/ESAG · Ciências Econômicas · 2025

**Autor:** Bruno Francisco Schaden  
**Orientadora:** Prof.ª Dra. Marianne Zwilling Stampe (UDESC)  
**Título completo:** *Racionalidade Coletiva em Xeque: Uma Investigação Comportamental sobre Percepções Econômicas no Brasil*

---

Este repositório contém a **revisão pós-entrega** do TCC: o documento LaTeX v2 já entregue à banca, o pipeline econométrico completo em Python, e o ambiente de trabalho configurado com Claude Code. O repositório reflete o estado atual do projeto após as correções metodológicas da auditoria pós-defesa — não a versão exata entregue originalmente.

---

## Estrutura do repositório

```
tcc_esag/
├── tcc_v2.tex                   ← entry point do LaTeX (documento principal)
├── PacotesBasicos.tex           ← pacotes e configurações LaTeX
├── abntex2-alf_revNBR2023.bst  ← estilo ABNT (BibTeX, citação Autor, Ano)
├── referencias.bib              ← base bibliográfica em BibTeX (UTF-8)
│
├── PreTextuais/                 ← capa, folha de rosto, resumo, sumário, etc.
├── Textuais_v2/                 ← capítulos do TCC (versão final)
│   ├── introducao.tex
│   ├── revisao_literatura.tex
│   ├── metodologia.tex
│   ├── resultados.tex
│   └── conclusao.tex
├── PosTextuais_v2/              ← apêndices, glossário, anexos
│   ├── Apendices.tex
│   ├── Glossario.tex
│   └── Anexos.tex
│
├── analise/                     ← pipeline econométrico Python (ver abaixo)
│   ├── run_all.py               ← ponto de entrada único
│   ├── *.py                     ← módulos do pipeline
│   ├── ANALISE_COMPLETA.md      ← documentação metodológica detalhada
│   └── outputs/
│       ├── tables/csv/          ← 15 CSVs de resultados
│       ├── tables/tex/          ← 13 .tex prontos para \input{}
│       ├── figures/             ← counterfactual, forestplots, heatmaps, distributions
│       ├── cache/               ← pickle do pipeline para --fast
│       └── logs/                ← run_YYYYMMDD_HHMMSS.log
│
├── material_referencia/         ← PDFs de referência bibliográfica
│   ├── MAPA_REFERENCIAS.md      ← índice de tópico → fonte → páginas
│   └── *.pdf                    ← Wooldridge, Angrist & Pischke, Cameron & Trivedi, etc.
│
└── .claude/                     ← ambiente Claude Code (ver abaixo)
    ├── settings.json
    ├── settings.local.json
    └── agents/
        └── econometrics-professor.md
```

---

## Pipeline de Análise (`analise/`)

O pipeline é modular e reprodutível. Ponto de entrada único:

```bash
cd analise
python3 run_all.py           # análise completa (~90 min com bootstrap B=999)
python3 run_all.py --fast    # usa cache, sem re-rodar bootstrap (~2 min)
python3 run_all.py --no-cache  # força reanálise completa dos dados brutos
```

Os dados são carregados diretamente do Google Sheets (sem arquivo local — a planilha é pública em modo CSV).

### Módulos

| Módulo | Função |
|--------|--------|
| `data_loader.py` | Carrega planilha, limpa, codifica variáveis ordinais e dummies |
| `model_base.py` | Logit ordenado (PO), Firth-Ridge para DVs singulares, LPM |
| `diagnostics.py` | Teste de Brant, VIF, McFadden R², AIC/BIC, poder estatístico |
| `multiple_testing.py` | Correções Bonferroni, Holm, BH e BY sobre 53 p-valores |
| `effects.py` | AME para todos os 53 DVs, Odds Ratios, contrafactuais, interação econ × espectro |
| `bootstrap_ci.py` | Bootstrap BC₀ (B=999) para β_econ (Davidson & MacKinnon, Cap. 4) |
| `permutation_tests.py` | Mann-Whitney U, KS, testes de permutação por DV |
| `ame_espectro.py` | AMEs do espectro político (top-10 por magnitude) |
| `tables.py` | Gera os 15 CSVs em `outputs/tables/csv/` |
| `latex_export.py` | Converte CSVs em `.tex` booktabs/longtable para `\input{}` |
| `figures.py` | Forest plots, heatmaps, volcano, contrafactuais, curva de poder |
| `tab_amostra_descritiva.py` | Tabela descritiva da amostra por DV e grupo |
| `run_all.py` | Orquestrador: 7 etapas sequenciais com cache e flags |

### Modelo estatístico

- 53 variáveis dependentes (DVs ordinais, 5 pontos)
- 48 estimados via logit ordenado (chances proporcionais)
- 5 estimados via Firth-Ridge (Hessiana singular)
- Controles: espectro político, escolaridade, idade, gênero, exposição profissional
- N efetivo por modelo: 144–146 (excluídos 38 sem espectro declarado)
- Correção múltipla: Benjamini-Hochberg (BH)

Documentação metodológica completa — decisões, referências exatas, alertas de interpretação — em [`analise/ANALISE_COMPLETA.md`](analise/ANALISE_COMPLETA.md).

---

## Compilação LaTeX

O documento é compilado no **Overleaf** (repositório sincronizado via GitHub). Para compilar localmente com TeX Live:

```bash
pdflatex tcc_v2.tex
bibtex tcc_v2
pdflatex tcc_v2.tex
pdflatex tcc_v2.tex
```

Defina `tcc_v2.tex` como arquivo principal no Overleaf.

---

## Ambiente Claude Code (`.claude/`)

O projeto usa Claude Code como assistente de desenvolvimento e análise. A pasta `.claude/` contém a configuração do ambiente local:

### `.claude/settings.json`
Configuração base do projeto: modo de permissões padrão (`bypassPermissions` — autorização prévia de todas as ferramentas para o projeto).

### `.claude/settings.local.json`
Configuração local de sessão:
- **Permissões explícitas** para Bash, Read, Write, Edit, Glob, Grep, WebFetch e WebSearch
- **Plugins habilitados**: dbt, voltagent (data-ai, lang, dev-exp), databricks-ai-dev-kit, snowflake-ai-kit
- `effortLevel: "high"` — nível máximo de raciocínio
- `skipDangerousModePermissionPrompt: true` — modo autônomo para o projeto

### `.claude/agents/econometrics-professor.md`
Agente especializado em econometria que lê os PDFs da pasta `material_referencia/` e fornece análise acadêmica baseada nas referências do projeto. Ao ser invocado, o agente:
1. Lê o `MAPA_REFERENCIAS.md` para identificar quais PDFs e páginas consultar
2. Lê os PDFs relevantes na ordem de prioridade (P1 → P2 → P3)
3. Responde com fundamento teórico, pressupostos, aplicação ao contexto, alertas e recomendações

Referências disponíveis para o agente: Wooldridge (2016, 2001), Davidson & MacKinnon (2004), Angrist & Pischke (2009), Cameron & Trivedi (2005), Hosmer, Lemeshow & Sturdivant (2013), Rudas (2024), Imbens & Rubin (2015), Hernán & Robins (2020), Manski (2009), Lohr (2009/2021), Särndal & Lundström (2005), Gelman & Carlin (2014), Elliott & Valliant (2017), Mercer et al. (2017), Cornesse et al. (2020).

---

## Dados e ética

Os dados foram coletados via survey (Google Forms, ago–set 2025), aprovado pelo CEP/UDESC — **Parecer nº 7.719.326**, CAAE: 89374225.9.0000.0118. Os dados disponibilizados neste repositório são **agregados e anonimizados**, em conformidade com a LGPD. Os dados brutos individuais não são publicados.

---

## Citação

```bibtex
@monography{schaden2025racionalidade,
  author      = {Schaden, Bruno Francisco},
  title       = {Racionalidade Coletiva em Xeque: Uma Investigação Comportamental
                 sobre Percepções Econômicas no Brasil},
  school      = {Universidade do Estado de Santa Catarina},
  year        = {2025},
  address     = {Florianópolis},
  type        = {Trabalho de Conclusão de Curso},
  note        = {Curso de Ciências Econômicas, UDESC/ESAG}
}
```
