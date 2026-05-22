# Mapa de Referências — material_referencia/

> Leia este arquivo PRIMEIRO. Ele indica quais PDFs priorizar para cada tópico,
> evitando varredura desnecessária em livros irrelevantes para a consulta.

---

## 1. Inventário completo dos PDFs

### Livros-texto — Econometria (núcleo do TCC)

| Sigla | Arquivo (prefixo) | Referência completa | Páginas aprox. |
|-------|-------------------|---------------------|----------------|
| **WOOL-INTRO** | `Introductory Econometrics_ A Modern...Wooldridge` | Wooldridge, J.M. *Introductory Econometrics: A Modern Approach*. 6ª ed. Cengage, 2016. | ~910 |
| **WOOL-PANEL** | `Econometric Analysis of Cross Section...Wooldridge` | Wooldridge, J.M. *Econometric Analysis of Cross Section and Panel Data*. MIT Press, 2001. | ~750 |
| **DM** | `Econometric Theory and Methods...Davidson & Mackinnon` | Davidson, R. & MacKinnon, J.G. *Econometric Theory and Methods*. Oxford, 2004. | ~750 |
| **AP** | `Mostly Harmless Econometrics...Angrist & Pischke` | Angrist, J.D. & Pischke, J.S. *Mostly Harmless Econometrics*. Princeton UP, 2009. | ~392 |
| **CT** | `Microeconometrics...Cameron & Trivedi` | Cameron, A.C. & Trivedi, P.K. *Microeconometrics: Methods and Applications*. Cambridge UP, 2005. | ~1050 |

### Livros-texto — Modelos de Resposta Limitada / Dados Categóricos

| Sigla | Arquivo (prefixo) | Referência completa | Páginas aprox. |
|-------|-------------------|---------------------|----------------|
| **HLS** | `Applied Logistic Regression...Hosmer` | Hosmer, D.W., Lemeshow, S. & Sturdivant, R.X. *Applied Logistic Regression*. 3ª ed. Wiley, 2013. | ~510 |
| **RUDAS** | `Lectures on Advanced Topics in Categorical Data...Rudas` | Rudas, T. *Lectures on Advanced Topics in Categorical Data Analysis*. Springer, 2024. | ~300 |

### Livros-texto — Inferência Causal

| Sigla | Arquivo (prefixo) | Referência completa | Páginas aprox. |
|-------|-------------------|---------------------|----------------|
| **IR** | `Causal Inference for Statistics...Imbens & Rubin` | Imbens, G.W. & Rubin, D.B. *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge UP, 2015. | ~625 |
| **HR** | `Causal Inference_ What If...Hernán & Robins` | Hernán, M.A. & Robins, J.M. *Causal Inference: What If*. CRC Press, 2020. | ~312 |
| **MANSKI** | `Identification for Prediction and Decision...Manski` | Manski, C.F. *Identification for Prediction and Decision*. Harvard UP, 2009. | ~330 |

### Livros-texto — Amostragem e Surveys

| Sigla | Arquivo (prefixo) | Referência completa | Páginas aprox. |
|-------|-------------------|---------------------|----------------|
| **LOHR-ADV** | `Sampling_ Design and Analysis (Advanced Series)...Lohr` | Lohr, S.L. *Sampling: Design and Analysis*. 2ª ed. Duxbury Press, 2009. | ~600 |
| **LOHR-CRC** | `Sampling_ Design and Analysis (Chapman & Hall)...Lohr` | Lohr, S.L. *Sampling: Design and Analysis*. 3ª ed. Chapman & Hall/CRC, 2021. | ~700 |
| **SL** | `Estimation in Surveys with Nonresponse...Särndal & Lundström` | Särndal, C.E. & Lundström, S. *Estimation in Surveys with Nonresponse*. Wiley, 2005. | ~300 |

### Artigos científicos

| Sigla | Arquivo | Referência completa | Tema |
|-------|---------|---------------------|------|
| **GELMAN2014** | `gelman2014.pdf` | Gelman, A. & Carlin, J. "Beyond Power Calculations: Assessing Type S and Type M Errors." *Perspectives on Psychological Science*, 9(6), 641–651, 2014. | Poder estatístico, erros de tipo S/M |
| **ELLIOTT2017** | `elliott2017.pdf` | Elliott, M.R. & Valliant, R. "Inference for Nonprobability Samples." *Statistical Science*, 32(2), 249–264, 2017. | Inferência em amostras não probabilísticas |
| **MERCER2017** | `mercer2017.pdf` | Mercer, A.W. et al. "Theory and Practice in Nonprobability Surveys: Parallels Between Causal Inference and Survey Inference." *Public Opinion Quarterly*, 81(S1), 250–279, 2017. | Surveys não probabilísticos e viés de seleção |
| **CORNESSE2020** | `cornesse2020.pdf` | Cornesse, C. et al. "A Review of Conceptual Approaches and Empirical Evidence on Probability and Nonprobability Sample Survey Research." *Journal of Survey Statistics and Methodology*, 2020. | Comparação surveys probabilísticos vs. não probabilísticos |

---

## 2. Mapa tópico → fontes prioritárias

Para cada tópico abaixo, os PDFs estão ordenados por relevância (**P1** = leia primeiro, **P2** = complementar, **P3** = referência pontual).

### Ordered Logit / Logit Ordenado / Proportional Odds
- **P1:** HLS — *Applied Logistic Regression*, Cap. 8 (Polytomous Logistic Regression) e Cap. 7 (Ordinal Logistic Regression). Seção crítica: p. 287–340.
- **P1:** WOOL-INTRO — Cap. 17 (Limited Dependent Variable Models). p. 575–619.
- **P2:** CT — Cap. 15 (Discrete Choice Models), p. 490–560.
- **P2:** RUDAS — Cap. 4 e 5 (Cumulative models, Proportional odds).
- **P3:** DM — Cap. 11 (Qualitative and Limited Dependent Variables).

### Brant Test (Proportional Odds Assumption)
- **P1:** HLS — *Applied Logistic Regression*, p. 302–310 (Score test for proportional odds).
- **P1:** RUDAS — seções sobre teste de proporcionalidade.
- **P2:** CT — Cap. 15, subseção sobre testes de especificação em logit ordenado.
- **P3:** WOOL-INTRO — Cap. 17 (discussão de pressupostos do logit ordenado).

### Separação Perfeita / Quasi-Separação (Ridge-Firth)
- **P1:** HLS — Cap. 5, p. 183–200 (Numerical Problems in Maximum Likelihood Estimation; Firth's penalized likelihood).
- **P2:** RUDAS — seções sobre estimação com tabelas esparsas.
- **P3:** CT — Cap. 23 (Maximum Likelihood Methods), sobre problemas numéricos.

### Average Marginal Effects (AME)
- **P1:** AP — Cap. 3 (Making Regression Make Sense), p. 31–68. Discussão de interpretação de efeitos em modelos não-lineares.
- **P1:** WOOL-INTRO — Cap. 17, p. 594–601 (Partial Effects, Average Partial Effects).
- **P2:** CT — Cap. 15, p. 495–505 (Marginal Effects para modelos discretos).
- **P2:** HLS — Cap. 8, p. 315–330 (Predicted probabilities e mudanças marginais).

### VIF / Multicolinearidade
- **P1:** WOOL-INTRO — Cap. 3, p. 97–104 (Multicollinearity; VIF).
- **P2:** DM — Cap. 3 (The Statistical Properties of Ordinary Least Squares), p. 100–115.
- **P3:** CT — Cap. 3 (Regression and Projection).

### Bootstrap / Intervalos de Confiança Reamostrados
- **P1:** DM — Cap. 4 (The Bootstrap), p. 147–190. Referência principal para BC₀ e BCa.
- **P2:** AP — Cap. 8, p. 290–301 (Bootstrap Standard Errors).
- **P3:** CT — Cap. 11 (Bootstrap Methods), p. 357–395.

### Testes de Permutação / Não-Paramétricos
- **P1:** DM — Cap. 4, seção sobre randomization tests, p. 183–190.
- **P2:** CT — Cap. 11 (Simulation-Based Inference).
- **P3:** AP — Cap. 8 (Nonstandard Standard Error Issues).

### Correções para Múltiplos Testes (Bonferroni, Holm, BH, BY)
- **P1:** WOOL-INTRO — Cap. 4, p. 132–135 (discussão sobre múltiplos testes).
- **P2:** AP — Cap. 1, p. 18–24 (The Experimental Ideal; multiple comparisons).
- **P3:** CT — Cap. 9 (Hypothesis Tests).

### Análise de Poder Estatístico
- **P1:** GELMAN2014 — artigo completo (Type S/M errors, retrospecive power). ~11 páginas.
- **P2:** WOOL-INTRO — Cap. 4 (Multiple Regression Analysis: Inference), p. 128–145.
- **P3:** DM — Cap. 3 (Statistical Properties of OLS), sobre tamanho amostral.

### Efeitos de Interação / Moderação
- **P1:** WOOL-INTRO — Cap. 6, p. 196–212 (Interaction Terms).
- **P2:** AP — Cap. 3, p. 60–68 (Heterogeneous Treatment Effects).
- **P3:** CT — Cap. 2 e 15 (sobre interações em modelos não-lineares).

### Goodness-of-Fit: McFadden R², AIC, BIC, LR Test
- **P1:** WOOL-INTRO — Cap. 17, p. 583–590 (GOF para modelos de resposta limitada).
- **P1:** HLS — Cap. 5, p. 160–185 (Model Building Strategies; Goodness-of-Fit).
- **P2:** DM — Cap. 8 (Specification Tests), p. 291–330.
- **P3:** CT — Cap. 5 (Hypothesis Testing), p. 153–220.

### Contrafactual / Tratamento / Design Observacional
- **P1:** AP — Cap. 2 (The Experimental Ideal), p. 11–24.
- **P1:** IR — Cap. 1 e 2 (Framework para inferência causal com observacionais).
- **P2:** HR — Cap. 1 e 2 (Causal inference introdução e DAGs).
- **P3:** MANSKI — Cap. 1 (Identification).

### Surveys Não Probabilísticos / Viés de Seleção Amostral
- **P1:** ELLIOTT2017 — artigo completo (inferência em amostras não probabilísticas). ~15 páginas.
- **P1:** MERCER2017 — artigo completo (paralelo causal-inferência e survey). ~29 páginas.
- **P1:** CORNESSE2020 — artigo completo (revisão comparativa). ~33 páginas.
- **P2:** LOHR-CRC — Cap. 1 e 2 (Introdução, probability sampling).
- **P2:** SL — Cap. 1–3 (Nonresponse e viés).

### Linear Probability Model (LPM / OLS como robustez)
- **P1:** AP — Cap. 3, p. 42–53 (Regression Anatomy; LPM como benchmark).
- **P1:** WOOL-INTRO — Cap. 17, p. 578–583 (LPM vs. probit/logit).
- **P2:** CT — Cap. 14 (Binary Outcome Models), p. 464–475.

---

## 3. Hierarquia geral de importância para este TCC

```
NÍVEL A — Leitura obrigatória para qualquer consulta técnica do TCC
  WOOL-INTRO   (ordered logit, VIF, poder, LPM, múltiplos testes)
  AP           (AME, causalidade, bootstrap, interação)
  HLS          (logit ordenado, Brant, separação perfeita, GOF)
  DM           (bootstrap, permutation, teoria de estimação)

NÍVEL B — Consultar quando o tópico exigir aprofundamento
  CT           (modelos discretos, bootstrap, microeconometria)
  WOOL-PANEL   (teoria, cross-section, endogeneidade)
  IR           (inferência causal, observacionais)
  RUDAS        (dados categóricos, cumulative models)

NÍVEL C — Referência pontual / metodologia de survey
  HR           (DAGs, causalidade)
  MANSKI       (identificação, bounds)
  LOHR-CRC / LOHR-ADV  (amostragem)
  SL           (não-resposta)
  GELMAN2014   (poder, erros tipo S/M)
  ELLIOTT2017 / MERCER2017 / CORNESSE2020  (surveys não probabilísticos)
```

---

## 4. Instrução de uso para o agente

1. **Leia este arquivo antes de qualquer busca nos PDFs.**
2. Identifique o tópico da consulta e consulte a seção 2 acima.
3. **Priorize os PDFs de nível P1** para o tópico. Leia apenas as páginas indicadas.
4. Se P1 não cobrir suficientemente, avance para P2.
5. Consulte P3 apenas para citações pontuais ou teoria de suporte.
6. Para novos PDFs adicionados à pasta sem entrada neste mapa,
   leia as primeiras 3 páginas para identificar título/autor, classifique
   provisoriamente e sinalize ao usuário que o mapa precisa ser atualizado.
