# Como a análise funciona — do início ao fim, por blocos

> TCC "Racionalidade Coletiva em Xeque" — UDESC/ESAG 2025
> Pipeline em `Textuais/analise_v2/` — isolado de `Textuais/analise/`

---

## Bloco 1 — Configuração Central (`config.py`)

Este é o único lugar onde você toca para mudar parâmetros. Nenhum outro módulo tem constante hardcoded.

**O que está aqui:**

- **URL do Google Sheets** (hardcoded), que é a fonte única de dados brutos.
- **`ALPHA = 0.05`** — nível de significância global.
- **`N_BOOTSTRAP = 100`** — reduzido de 500 para viabilidade (53 DVs × 100 reamostras × ~126ms/fit ≈ 11 min; com 500 seriam 55 min).
- **`N_PERMUTATIONS = 5000`** — permutation test por DV.
- **`RIDGE_ALPHA = 1.0`** — penalidade L2 para os 5 DVs singulares.

**Mapeamentos críticos:**

```python
ECON_MAP = {"Não tenho formação...": 0, "Graduação em Economia": 0,
            "Mestrado em Economia": 1, "Doutorado em Economia": 1}
```

Só mestrado e doutorado = `econ=1`. Graduação é excluído do grupo tratamento. Esta é uma escolha deliberada: a hipótese é sobre **formação profissional plena**, não contato superficial com economia.

```python
IDEOL_MAP = {"Extrema esquerda": -2, "Esquerda": -1, "Centro": 0,
             "Direita": 1, "Extrema direita": 2,
             "Independente": NaN, "Sem opinião": NaN}
```

Espectro como escala contínua ordinal. "Independente" vira NaN deliberadamente — colocá-lo em 0 (junto com "Centro") distorceria a escala, misturando pessoas apolíticas com centristas ideológicos.

```python
ESCOLARIDADE_MAP = {"Ensino Fundamental Incompleto": 0, ..., "Pós-graduação": 6}
IDADE_MAP = {"Até 18 anos": 0, ..., "66 anos ou mais": 6}
```

Escolaridade e idade como ordinais inteiros, não dummies. A razão é técnica: com n_econ=17 e categorias como "Ensino Fundamental Incompleto" com 0 ou 1 respondentes, dummies causariam **separação perfeita** — o modelo nunca converge (Wooldridge §17.1).

**`SINGULAR_DVS`** — lista dos 5 DVs que causam Hessiana singular no ordered logit padrão:

- `impostos_altos`, `corte_impostos`, `aumento_uso_tecnologia`, `acordos_comerciais_outros`, `acha_preços_combustíveis`

Para esses 5, o pipeline usa `fit_firth_ridge()` em vez de `fit_ordered_logit()`.

**`BLOCS`** — agrupa as 53 DVs em 6 famílias teóricas (H3 a H7 + Percepção-Específica). Esta taxonomia permite analisar padrões por grupo de hipótese no forest plot e heatmap.

---

## Bloco 2 — Carregamento e Pré-processamento (`data_loader.py`)

A função principal é `load_data(url) → DataBundle`. Ela faz 5 coisas em sequência:

**Passo 1 — Variável tratamento:**

```python
econ_series = df[FORM_ECON_COL].map(ECON_MAP)
```

Mapeia a coluna "Qual seu nível de formação..." para 0/1 usando `ECON_MAP`. `N_econ = (econ_series == 1).sum()` → 17.

**Passo 2 — Construção dos 6 controles:**

| Coluna no DataFrame | Fonte bruta | Transformação |
|---|---|---|
| `econ` | FORM_ECON_COL | ECON_MAP (0/1) |
| `espectro` | ESPECTRO_COL | IDEOL_MAP (-2…+2, NaN para independentes) |
| `escolaridade_num` | "Qual seu nível de escolaridade?" | ESCOLARIDADE_MAP (0–6) |
| `idade_num` | "Qual é a sua faixa etária?" | IDADE_MAP (0–6) |
| `genero` | "Você é homem?" | (== "Sim").astype(float) → 0/1 |
| `engajado` | "Você se considera...engajada?" | (== "Sim").astype(float) → 0/1 |

**Por que esses 6 e não mais?** Wooldridge §17.1 e A-P cap.3 recomendam incluir apenas confundidores que afetam *simultaneamente* o tratamento (econ) e os desfechos. Dummies raciais (n=2 para asiáticos) e vínculo empregatício (encoding inconsistente) foram excluídos.

**Passo 3 — Detecção automática de DVs:**

```python
excluir = set(_META_COLS) | set(_CONTROL_RAW) | set(_SKIP_COLS)
dep_raw = [c for c in df.columns if c not in excluir]
```

Tudo que não é metadado, controle bruto ou coluna a pular é tratado como variável dependente. A função `_detect_number(x)` extrai o menor dígito de strings como `"1 - Discordo totalmente"` → 1. Isso garante robustez a diferentes formatações de escala Likert.

**Passo 4 — Acronimização:**

`_make_acronym("Você acha que o déficit federal é grande demais?")` → `déficit_federal_grande`. Remove stopwords (de, da, do, em, para, etc.), pega as 3 primeiras palavras de conteúdo, junta com `_`. Colisões de nomes são resolvidas com sufixo `_2`, `_3`, etc.

**Resultado:** `DataBundle` com:

- `df_all` — todos os dados (DVs + controles) numa linha por respondente
- `df_controles` — só a matriz de controles (6 colunas)
- `dependentes_cols` — lista de 53 nomes de DV acronimizados
- `missingness` — relatório de N por DV (N_bruto vs N_modelo, após dropna)
- `N_econ=17`, `N_total≈183`

---

## Bloco 3 — Modelos Estatísticos (`model_base.py`)

> **[Auditoria v2 — REVISADO]** Este bloco passou por reescrita completa. Ver lista de correções abaixo.

Todos os modelos retornam `ModelResult`, um dataclass padronizado com os mesmos campos independente do método usado. Isso permite que `tables.py`, `effects.py` e `bootstrap_ci.py` processem qualquer resultado sem branching por tipo de modelo.

**Campos novos no `ModelResult` (auditoria v2):**
- `is_numerically_suspect: bool` — SE>10 sem NaN (quasi-separação sem colapso total da Hessiana)
- `optimizer_used: str` — otimizador que convergiu ("lbfgs" | "bfgs" | "powell" | "L-BFGS-B-Ridge" | "OLS-HC3")

### 3.1 — Ordered Logit / Proportional Odds

`fit_ordered_logit(y, X)` usa `statsmodels.miscmodels.ordinal_model.OrderedModel`.

**O modelo em equações:**

```
P(Y_i <= k) = Λ(τ_k − X_i'β)
```

onde `Λ` é a função logística, `τ_k` são os cutpoints (limiares) e `β` são os coeficientes. O modelo assume que o **mesmo vetor β** governa todos os thresholds (daí "proportional odds" — a razão de odds é constante entre categorias).

**Fallback chain:** lbfgs → bfgs → powell. O lbfgs é rápido mas pode não convergir para DVs difíceis; bfgs é mais robusto; powell não usa gradiente (maxiter=2000) e é o último recurso — emite warning quando acionado.

**`is_singular = True`** quando qualquer SE é NaN OU (SE>10 E coef>10) OU (SE>10 E razão SE/|coef|>5 na mesma variável). A última condição captura quasi-separação moderada que o critério coef>10 não detecta (Hosmer-Lemeshow Cap.5, p.185-186). Modelos singulares vão para o Firth-Ridge.

**`converged`** usa `mle_retvals.get("converged", False)` — default conservador (False), não assume convergência quando a chave está ausente.

**`cutpoints_ok`** verifica se os cutpoints estão em ordem crescente usando o padrão de nomenclatura real do statsmodels ("1/2", "2/3", etc., não "cut...").

**`mcfadden_r2 = 1 − llf/llnull`** — McFadden R² calculado com a mesma amostra sem NaN em numerador e denominador (A1: consistência de N).

### 3.2 — Linear Probability Model (robustez A-P)

`fit_lpm(y, X)` implementa OLS em `y_norm = (y − min)/(max − min)` com erros HC3 (heteroskedasticity-consistent, MacKinnon-White 1985). Esta é a verificação de robustez canônica de Angrist-Pischke cap.3: se OLS e ordered logit concordam na direção e significância, o resultado é robusto à especificação funcional.

`y_norm` em [0,1] tem interpretação direta: um coeficiente de 0,08 para `econ` significa que economistas marcam 8 pontos percentuais a mais no índice normalizado da resposta.

**Nota (B7):** `mcfadden_r2` armazena **R² OLS** (não McFadden); `nagelkerke_r2` armazena R²-ajustado. A semântica difere do modelo logit — declarada explicitamente no código.

### 3.3 — Firth-Ridge Penalizado (para os 5 singulares)

`fit_firth_ridge(y, X, alpha=0.1)` implementa ordered logit com penalidade L2 (Ridge):

```
L_penalizado(β) = −log-verossimilhança(β) + (α/2) Σ_j β_j²
```

**RIDGE_ALPHA = 0.1** (era 1.0): reduzido para alpha ∝ p/n ≈ 6/180 ≈ 0.033; 0.1 é o valor heurístico defensável (Cameron & Trivedi Cap.23). Alpha=1.0 encolhia coeficientes em ~50%, excessivo para n=180, p=6.

**Parametrização de cutpoints (nova — auditoria v2):**

```
cuts[0] = raw[0]               # livre em R — admite cutpoints negativos
cuts[i] = cuts[0] + cumsum(exp(raw[1:i+1]))   # i ≥ 1
```

Implementada em `_reconstruct_cuts()`. A parametrização anterior (`cumsum(exp(raw))`) forçava cuts>0, produzindo inconsistência com a inicialização quando o primeiro cutpoint empírico é negativo (caso comum com maioria das respostas nas categorias inferiores).

**Inicialização (C10):** `_init_cuts_from_data()` inicializa via quantis empíricos (logit das proporções acumuladas), consistente com `_reconstruct_cuts`: `raw[0] = logit_cuts[0]` (direto); `raw[i] = log(max(diff, 0.01))` para i≥1. A inicialização converge para os cutpoints empíricos exatos.

**Hessiana numérica (C12):** `_numerical_hessian()` usa diferenças centrais cruzadas O(n²):
```
H[i,j] = [f(x+ei+ej) − f(x+ei−ej) − f(x−ei+ej) + f(x−ei−ej)] / (4*hi*hj)
```
Step relativo: max(|x_i|×1e-5, 1e-7). Estável para parâmetros de magnitudes diferentes. Verifica positiva-definitude via autovalores; clamps diagonal negativa para evitar SE complexo.

**C14 — convergência fraca:** aceita solução quando grad_norm < 0.5 mesmo sem `res.success=True`. O limiar 0.5 é heurístico (sem referência bibliográfica formal); logar para diagnóstico.

**C15 — z-score NaN:** z→NaN quando SE<1e-6 para evitar p-value artificialmente zero causado por divisão por epsilon numérico.

**SE e AIC/BIC:** os SE são da Hessiana do objetivo penalizado — válidos para o estimador Firth-Ridge, mas anticonservadores (subestimam incerteza real, Hosmer-Lemeshow Cap.5 p.193). Bootstrap BC₀ é o referencial primário de inferência. O AIC/BIC usa k=len(params) — não comparável ao AIC do MLE (número efetivo de parâmetros do Ridge < k); válido apenas para ranking interno entre DVs Firth-Ridge.

### 3.4 — Generalized Ordered Logit (para o Brant test)

`fit_gol(y, X)` ajusta K−1 logits binários separados: `P(Y > k)` para cada threshold k. É o modelo **sem** a restrição de odds proporcionais. O output é um dict `{threshold_value: statsmodels_logit_result}`. É usado internamente pelo `brant_test()`.

### 3.5 — Correções aplicadas na auditoria v2 (resumo)

| Ref | Descrição | Impacto |
|-----|-----------|---------|
| A1  | N consistente: y_clean em toda fit_ordered_logit e fit_lpm | McFadden R² e N corretos |
| A2  | Logar optimizer_used; warning em powell | Diagnóstico |
| A3  | converged default=False (conservador) | Evita assumir convergência silenciosamente |
| A4  | _detect_singularity: AND por variável; + critério SE/coef>5 | Detecta quasi-separação moderada |
| A5  | _cutpoints_ok usa "/" (padrão statsmodels) | Verificação de cutpoints agora funciona |
| A6  | y.dropna() em todas as funções | Evita NaN em classes/searchsorted |
| B7  | Documentar mcfadden_r2=R²OLS em LPM | Clareza semântica |
| C10 | _init_cuts_from_data por quantis empíricos | Convergência mais rápida/robusta |
| C12 | _numerical_hessian: O(n²) diferenças centrais, step relativo | Elimina cancelamento catastrófico |
| C14 | grad_norm<0.5 aceita solução fraca | Evita descartar soluções utilizáveis |
| C15 | z→NaN quando SE≈0 | Elimina p-value falsamente significativo |
| C16 | RIDGE_ALPHA 1.0→0.1 | Reduz viés excessivo de encolhimento |
| — | _reconstruct_cuts: parametrização separada cuts[0]/incrementos | Admite cutpoints negativos; consistente com init |
| — | _nagelkerke: guard para cox_snell<0 | Evita R² sem interpretação |
| — | Powell maxiter=2000 | Evita parada prematura |
| — | Verificação PD da Hessiana Firth-Ridge | Alerta para Hessiana mal-condicionada |

**Débito técnico registrado (não tocado neste bloco):**
- ~~E22 (bootstrap_ci.py): bootstrap dos 5 DVs Firth-Ridge usa fit_ordered_logit — deveria usar fit_firth_ridge~~ **CORRIGIDO no Bloco 7**
- ~~E21 (bootstrap_ci.py): beta_obs=NaN usa nanmedian(y) como fallback — incoerente~~ **CORRIGIDO no Bloco 7**
- D17 (diagnostics.py): Brant test ignora covariância entre threshold-specific estimates
- D18 (diagnostics.py): GOL com constante vs PO sem — verificar exclusão na estatística

---

## Bloco 4 — Diagnósticos (`diagnostics.py`)

### 4.1 — Brant Test

**Referência**: Brant (1990) *Biometrics* 46(4):1171–1178; Hosmer, Lemeshow & Sturdivant (2013) Cap.8 §8.2.2 p.302 eq.(8.26).

A lógica: se o modelo PO está correto, o coeficiente de cada preditor deve ser o mesmo em todos os K−1 logits binários. O Brant test compara esses coeficientes formalmente.

**Implementação manual em Python** (statsmodels não tem isso nativamente):

1. Ajusta K−1 logits binários `P(Y > k)` para cada threshold k, obtendo `β_k` e `SE_k`
   - Convenção: `y_bin = (y > k)` — alinha com statsmodels `OrderedModel` onde β > 0 → aumenta P(Y > k)
   - Logit de `P(Y ≤ k)` daria coeficiente −β, quebrando o teste mesmo sob H0
2. Ajusta o modelo PO ordenado, obtendo `β_PO` e `SE_PO`
3. Para cada preditor j e threshold k, computa:
   ```
   χ²_contribuição = (β_k[j] - β_PO[j])² / (SE_k[j]² + SE_PO[j]²)
   ```
   (aproximação conservadora: omite covariância entre β_k e β_PO, subestimando a estatística — HLS p.302, nota metodológica)
4. Soma sobre todos os (K−1)×p termos → χ² total com **df = (K−1)×p**
   - Para K=3 e p=7 preditores: df = 2×7 = **14** (não 7, que seria df=(K−2)×p — erro corrigido na auditoria v2)
5. H0: odds proporcionais. Desejamos **não rejeitar** (p > 0,05)

**Resultado em nossos dados (N=183, 48 DVs estimáveis por ordered logit, 5 via Firth-Ridge):**

| DV | χ²(14) | p-valor | PO ok? |
|---|---|---|---|
| `quem_considera_maior` | 15,897 | 0,320 | Sim |
| `privatização_estatais_benéfica` | 12,330 | 0,580 | Sim |
| `governo_atual_sabe` | 7,924 | 0,893 | Sim |
| (outros 45 DVs) | < 7,5 | > 0,91 | Sim |

**0 de 48 DVs falham o teste PO** (todos com p > 0,32). Isso valida o uso do Ordered Logit / Proportional Odds para todos os 48 desfechos estimáveis. Os 5 DVs Firth-Ridge (singular) não recebem Brant test.

> **Nota histórica**: versão anterior do código usava `df = (K−2)×p = 7` em vez do correto `(K−1)×p = 14`. Com df=7, `quem_considera_maior` reportava p=0,024 e "falhava" o teste; com df=14 correto, p=0,320 — aprovado. A auditoria v2 corrigiu esse bug crítico.

### 4.2 — VIF

**Referência**: Wooldridge (2016) Cap.3 §3-4a p.86: VIF_j = 1/(1−R²_j), onde R²_j é o R² da regressão de X_j sobre os demais preditores.

VIF > 10 (heurística) indica multicolinearidade problemática. Em nossos controles primários (espectro, escolaridade_num, idade_num, genero, trabalha_econ), todos os VIFs ficam abaixo de 3, confirmando que o uso de ordinais em vez de dummies resolveu a separação perfeita e colinearidade dos grupos de baixa frequência.

### 4.3 — Goodness-of-Fit

**Referência**: McFadden R² — Wooldridge (2016) Cap.17 p.583–590; LR test — Wooldridge (2016) eq.(17.12); Nagelkerke (1991) *Biometrika* 78(3):691.

Para cada `ModelResult`, `goodness_of_fit()` retorna:

- `mcfadden_r2` e `nagelkerke_r2`
- `aic` e `bic` (penalizam complexidade; menor é melhor)
- **LR test**: `χ²(df) = −2(ℓ_nulo − ℓ_completo)`, testa se o modelo como um todo é melhor que o modelo nulo (só cutpoints)

**Resultado em nossos dados (53 DVs, todos convergidos):**

| Estatística | Valor |
|---|---|
| McFadden R² — mínimo | 0,015 |
| McFadden R² — mediana | 0,099 |
| McFadden R² — média | 0,117 |
| McFadden R² — máximo | 0,369 |
| Nagelkerke R² — média | 0,215 |
| LR test p < 0,05 | 38 / 53 (72%) |
| Modelos convergidos | 53 / 53 (100%) |

McFadden R² ≥ 0,10 é considerado ajuste aceitável para modelos de escolha discreta (HLS Cap.5). A mediana de 0,099 sugere modelos com ajuste modesto, esperado para variáveis de opinião com alto ruído individual.

### 4.4 — Power Analysis (Whittemore 1981)

**Referência**: Whittemore (1981) *JASA* 76(373):27–32; Gelman & Carlin (2014) *Perspect Psychol Sci* 9(6):641–651.

O poder é calculado para dois cenários de p₀ (probabilidade de concordância no grupo controle):

- **Conservador** (p₀ = 0,5): hipótese mais desfavorável, maximiza variância binomial
- **Likert uniforme** (p₀ = 1/3): assume distribuição uniforme entre 3 categorias Likert

| OR | Power conservador | Power Likert uniforme | Interpretação |
|---|---|---|---|
| 1,5 | 12,6% | 11,8% | Efeito pequeno: praticamente não detectável |
| 2,0 | 28,2% | 26,0% | Efeito médio: ~73% de falsos negativos |
| 2,5 | 45,4% | 42,3% | Efeito grande: ainda abaixo de 50% |
| 3,0 | 60,8% | 57,3% | Muito grande: marginalmente adequado |
| 4,0 | 81,9% | 79,3% | Enorme: detectável com segurança |

*(n_econ=17, n_total=183)*

Com n_econ=17, **a ausência de significância não implica ausência de efeito**. Para detectar um OR=2 com 80% de poder, seriam necessários ~n_econ=50 (n_total≈500). Essa limitação é central para a interpretação e deve estar explícita na seção de limitações do TCC.

### 4.5 — Retrodesign (Gelman & Carlin 2014)

**Referência**: Gelman & Carlin (2014) *Perspect Psychol Sci* 9(6):641–651.

Para os 5 DVs nominalmente significativos (p_econ < 0,05), o retrodesign avalia: dado que detectamos significância, quão confiáveis são os resultados? Dois erros pós-hoc:

- **Type S** (sinal): Pr(sinal errado | significativo) — risco de inverter direção do efeito
- **Type M** (magnitude): E[|estimativa| / |efeito verdadeiro| | significativo] — inflação da magnitude

| DV | β_econ | OR | Power | Type S | Type M |
|---|---|---|---|---|---|
| `déficit_federal_grande` | −1,599 | 0,202 | 0,689 | ≈0 | 1,21 |
| `produtividade_aumentando_devagar` | +1,981 | 7,246 | 0,662 | ≈0 | 1,23 |
| `reforma_previdência_necessária` | −1,395 | 0,248 | 0,639 | ≈0 | 1,26 |
| `empresas_estão_enviando` | +1,390 | 4,016 | 0,571 | ≈0 | 1,32 |
| `produtos_importados_benéficos` | +1,449 | 4,260 | 0,540 | 0,0001 | 1,35 |

Interpretação: Type S ≈ 0 em todos — o sinal dos efeitos é confiável. Type M entre 1,2 e 1,4 — as magnitudes estão moderadamente infladas (exagero de 20–35%), o que é esperado para estudos com poder 54–69%. Esse nível de inflação é menor do que o típico em estudos exploratórios com baixo poder, e não invalida as conclusões direcionais.

---

## Bloco 5 — Correção de Multiplicidade (`multiple_testing.py`)

**Referências**: Bonferroni (1936); Holm (1979) *Scand J Stat* 6(2):65–70; Benjamini & Hochberg (1995) *JRSS-B* 57(1):289–300; Benjamini & Yekutieli (2001) *Ann Stat* 29(4):1165–1188.

Com 53 DVs e α=0,05, esperamos `0,05 × 53 = 2,65` falsos positivos por acaso. O pipeline aplica 4 correções:

| Método | Controla | Pressuposto | Quando usar |
|---|---|---|---|
| Bonferroni | FWER | Independência | Referência conservadora |
| Holm (1979) | FWER | Independência | Sempre domina Bonferroni |
| BH (1995) | FDR | PRDS (dep. positiva) | **Método principal** — DVs Likert correlacionadas |
| BY (2001) | FDR | Dep. arbitrária | Análise de sensibilidade conservadora |

**Famílias separadas pré-especificadas:**

- Família `econ`: 53 p-valores de β_econ (pergunta de pesquisa 1)
- Família `espectro`: 53 p-valores de β_espectro (pergunta de pesquisa 2)

BH aplicado por família separada é correto quando as famílias são pré-especificadas por distinção teórica (dois preditores distintos, duas perguntas de pesquisa independentes). BH poolado (106 testes) seria mais conservador sem justificativa conceitual.

`apply_corrections()` lida com NaN: extrai só os p-valores válidos, aplica a correção, e repõe NaN nas posições originais para manter alinhamento com os nomes das DVs.

**Resultado em nossos dados:**

| Família | Nominais (p < 0,05) | Após Holm | Após BH | Após BY |
|---|---|---|---|---|
| `econ` (β_econ) | **5 / 53** | 0 / 53 | 0 / 53 | 0 / 53 |
| `espectro` (β_espectro) | **38 / 53** | 37 / 53 | 37 / 53 | — |

**Família econ**: 5 DVs nominalmente significativas (p < 0,05), nenhuma sobrevive após BH (menor p_BH = 0,363). Isso é esperado: com poder de ~28% para OR=2, precisaríamos de efeitos muito grandes (OR≥4) para que o BH mantivesse rejeição com n_econ=17. Os 5 DVs nominais devem ser reportados como **sinais exploratórios sujeitos a inflação de Type M (≈1,2–1,4×)**, validados pelo bootstrap BC₀.

**Família espectro**: 38 nominais, 37 após BH — forte evidência de que o espectro ideológico é preditor robusto das opiniões econômicas independentemente da formação. Esperado: n_total=183 fornece poder adequado para espectro.

---

## Bloco 6 — Efeitos (`effects.py`)

### 6.1 — Average Marginal Effects (AME)

**Referências**: Cameron & Trivedi (2005) *Microeconometrics* Cap.14 p.470–477; Wooldridge (2016) Cap.17 p.576–582; HLS (2013) Cap.8 p.290.

O coeficiente β do ordered logit não tem interpretação direta em unidades de Y. Para comunicar "economistas marcam X pontos a mais na escala", usamos o **efeito marginal médio via diferenças finitas**:

```
AME_j = (1/N) Σ_i [P(Y_i=j | X_i, econ=1) − P(Y_i=j | X_i, econ=0)]
ΔY = Σ_j j × AME_j   (variação na resposta esperada — assume cardinalidade)
```

`compute_ame(mr, X)` usa `predict()` do statsmodels para calcular as probabilidades previstas para cada observação com `econ=1` e `econ=0` (mantendo controles fixos), depois calcula a diferença média. O resultado `delta_Y` diz: "em média, ser economista está associado a uma resposta X pontos mais alta/baixa na escala".

**Nota**: `delta_Y = Σ_j j·AME_j` trata as categorias 0, 1, 2 como valores cardinais com distâncias iguais — simplificação que o modelo ordinal não requer. Útil como indicador de direção e ordem de magnitude.

**DVs Firth-Ridge**: AME calculado apenas para ordered_logit (MLE padrão). Os 5 DVs ajustados via Firth-Ridge (coefs penalizados L2) não têm AME reportado — coeficientes penalizados não geram efeitos marginais interpretáveis via MLE. A ausência está documentada nas tabelas como NaN.

`compute_ame_all()` retorna também `E_Y_econ0` e `E_Y_econ1` — respostas esperadas **preditas** pelo modelo (não médias brutas observadas), condicionais aos controles — medida ceteris paribus.

### 6.2 — Odds Ratios

**Referências**: HLS (2013) Cap.8 p.290–295; IC via delta method — aproximação normal assintótica.

```
OR = exp(β_econ)
IC95 = [exp(β − 1,96·SE), exp(β + 1,96·SE)]
```

OR > 1: economistas têm maior probabilidade de marcar categorias superiores. OR < 1: marcam mais baixo. OR = 2: a chance (odds) de passar para categoria superior é o dobro para economistas.

**Limitação do IC delta method**: com n_econ=17, a aproximação normal pode ser liberal (IC muito estreito). O bootstrap BC₀ (bloco 7) fornece alternativa mais robusta para os 5 DVs nominalmente significativos.

**DVs Firth-Ridge**: OR = exp(beta_penalizado). O encolhimento L2 tende a atenuar β em direção a zero → OR tende a subestimar a magnitude do efeito real. Não é um lower bound formal para modelos não-lineares; interpretar como heurística direcional.

### 6.3 — Contrafactual

**Referências**: Imbens & Rubin (2015) *Causal Inference* Cap.12 §12.2 p.258–260; Angrist & Pischke (2009) Cap.2 §2.2 p.28.

`compute_counterfactual()` responde: "como economistas responderiam se não tivessem formação em economia?" Calcula:

```
media_cf = E[Y | X_economistas, econ=0]  ≡ E[Y(0) | W=1]  (ATT, resultado potencial)
```

Substitui `econ=1` por `econ=0` mantendo todos os outros controles dos economistas, e prediz via modelo. Esse número é plotado nos gráficos como a linha tracejada (público contrafactual).

**Limitação (CIA)**: o contrafactual é válido somente sob CIA — Y(0),Y(1) ⊥ econ | X. Com n_econ=17 e apenas 6 controles, há risco substancial de viés de seleção não-observada (habilidade analítica, motivação intrínseca, redes profissionais). O contrafactual deve ser interpretado como **associação condicional em X observáveis**, não como efeito causal puro (Imbens & Rubin §12.2.3 p.261: "Unconfoundedness Is Not Testable").

### 6.4 — Interação econ × espectro

**Referências**: Wooldridge (2016) Cap.6 §6-3 p.175–180; Angrist & Pischke (2009) Cap.3 pp.64–68.

```python
X_int["econ_x_espectro"] = X["econ"] * X["espectro"]
mr_int = fit_ordered_logit(y, X_int)
```

Ajusta o modelo com o termo de interação. β_interacao > 0 → efeito de econ **aumenta** com posição à direita no espectro. β_econ no modelo com interação é o efeito de formação avaliado em espectro=0 (respondentes centristas), substantivamente relevante pois Centro=0 no IDEOL_MAP.

`AME_econ_Q1` e `AME_econ_Q3` são exercícios hipotéticos que fixam `espectro` no Q1 ou Q3 para **toda a amostra** (não AMEs condicionais em respondentes efetivamente no Q1/Q3) — Wooldridge §6-2d p.179: "plug in interesting values of x_j, such as lower and upper quartiles."

**Nota collinearidade**: espectro já centrado em 0 (Centro=0); centering adicional desnecessário para interpretabilidade.

**Nota exogeneidade**: espectro pode ser endógeno (auto-seleção ideológica da profissão). O coeficiente β_3 capta moderação real + viés de seleção conjunta — reportar como associação, não moderação causal pura (Angrist & Pischke Cap.3 pp.64-68: mediador potencial econ→espectro→opinião).

**Resultado em nossos dados (pipeline --fast, n=183)**:

| DV | β_interacao | p_interacao |
|---|---|---|
| `pessoas_poupam_bastante` | +1,332 | 0,036 |
| `corte_impostos` | −1,265 | 0,032 |

Apenas 2 de 53 DVs mostram interação significativa (p < 0,05). Nenhum dos 5 DVs nominalmente significativos para econ é moderado pela ideologia — o efeito de formação é **ideologicamente robusto**.

---

## Bloco 7 — Bootstrap BC₀ (`bootstrap_ci.py`)

### Por que bootstrap?

Com n_econ=17, a distribuição assintótica normal dos coeficientes (base dos p-valores do ordered logit) pode ser uma aproximação ruim. O bootstrap não depende dessa aproximação.

**Referências**: Efron & Tibshirani (1993) §12.4 (fórmula BC₀); Davidson & MacKinnon (2004) Cap.4 p.163-168 (B=999, regra α(B+1)∈ℤ).

### Por que BC₀ em vez de BCa?

BC₀ implementado (bias-corrected percentil sem jackknife). BCa completo (Efron 1987, §14) exigiria n=183 fits jackknife — factível em extensão futura. BC₀ corrige o viés z₀ sem correção de aceleração; adequado quando o viés de aceleração é pequeno (típico em modelos de regressão bem especificados).

### Como funciona o BC₀:

```
z₀ = Φ⁻¹(#{β*_b < β̂} / B)
α_lo = Φ(2·z₀ + z_{α/2})
α_hi = Φ(2·z₀ + z_{1-α/2})
IC₉₅ = [percentil(β*_b, 100·α_lo), percentil(β*_b, 100·α_hi)]
```

Se z₀ = 0 (a mediana das reamostras coincide com β̂), o IC reduz ao percentil simétrico [2,5%, 97,5%]. Quando z₀ ≠ 0, os quantis são ajustados para corrigir o viés.

### p_boot — teste de sinal bootstrap:

```python
p_boot = 2 * min(frac_neg, frac_pos)
```

Testa se a distribuição bootstrap de β*_b está concentrada em um sinal consistente. **Não** é o p-valor bootstrap de Davidson & MacKinnon (eq.4.61), que centra a distribuição em zero sob H₀. p_boot deve ser lido como evidência direcional, não como teste de H₀: β=0.

### Implementação — DVs Firth-Ridge:

Para os 5 DVs com quasi-separação perfeita, cada reamostra bootstrap usa `fit_firth_ridge()` (não `fit_ordered_logit()`), mantendo consistência com o estimador original (Davidson & MacKinnon cap.4 p.162). Seeds independentes por DV: `seed=BOOTSTRAP_SEED+i`.

### Resultados (pipeline completo, B=999):

**Todos os 53 DVs**: B_valid ≥ 998 (média 998,98); method = "BC0_percentile" para todos. Sem falhas de convergência no nível de rejeição (B_valid < 10).

**5 DVs nominalmente significativas para econ** — ICs bootstrap:

| DV | β̂_econ | β*_mediana | IC₉₅ BC₀ | p_boot | IC cruza 0? |
|---|---|---|---|---|---|
| `produtividade_aumentando_devagar` | +1,981 | +2,065 | [+0,547; +16,855] | 0,000 | Não |
| `reforma_previdência_necessária` | −1,395 | −1,435 | [−2,898; −0,129] | 0,030 | Não |
| `déficit_federal_grande` | −1,599 | −1,629 | [−3,391; +0,093] | 0,046 | Quasi (limite +0,09) |
| `produtos_importados_benéficos` | +1,449 | +1,482 | [−0,064; +3,170] | 0,052 | Quasi (limite −0,06) |
| `empresas_estão_enviando` | +1,390 | +1,418 | [−0,477; +2,596] | 0,066 | Sim |

**Interpretação**: 2/5 DVs nominais têm ICs bootstrap inteiramente fora de zero (`produtividade_aumentando_devagar` e `reforma_previdência_necessária`). O IC muito largo de `produtividade_aumentando_devagar` [+0,547; +16,855] reflete alta variabilidade amostral em algumas reamostras (possível quasi-separação em reamostras específicas). Os 3 DVs com IC cruzando ou quasi-cruzando zero indicam que o efeito é instável no bootstrap — evidência adicional para a cautela sugerida pelo Type M (1,21–1,35× inflação).

**5 DVs com IC bootstrap não cruzando zero (toda a amostra)**:

| DV | β*_mediana | IC₉₅ BC₀ | p_boot |
|---|---|---|---|
| `produtividade_aumentando_devagar` | +2,065 | [+0,547; +16,855] | 0,000 |
| `aumento_uso_tecnologia`* | +1,706 | [+0,690; +3,728] | 0,000 |
| `taxa_juros_selic` | +1,622 | [+0,122; +16,813] | 0,014 |
| `reforma_previdência_necessária` | −1,435 | [−2,898; −0,129] | 0,030 |
| `redução_recente_postos` | −1,464 | [−16,371; −0,063] | 0,022 |

*DVs marcados com * são Firth-Ridge (bootstrap com `fit_firth_ridge()`)

---

## Bloco 8 — Testes Não-Paramétricos (`permutation_tests.py`)

Três testes marginais independentes do modelo (sem assumir distribuição, sem controle de covariates):

**Mann-Whitney U (Wilcoxon rank-sum):** Testa H₀: P(Y_econ1 > Y_econ0) = 0,5 (stochastic dominance). Mais apropriado que t-test para DVs ordinais (Agresti 2002, Cap.8 p.278). `scipy.stats.mannwhitneyu(y0, y1, alternative="two-sided")`.

**Kolmogorov-Smirnov:** Testa H₀ de igualdade da distribuição acumulada completa. Com n₁=17, o suporte empírico F̂(y|econ=1) é grosseiro → menor poder que MWU para detectar deslocamentos ordenados. `scipy.stats.ks_2samp(y0, y1)`.

**Permutation test (diferença de medianas):** Embaralha o label `econ` N_PERMUTATIONS=5000 vezes, recalcula |mediana_e1 − mediana_e0| a cada vez. p-value empírico; N=5000 ≫ mínimo de 190 para α=0,05 (Phipson & Smyth 2010). Seeds independentes por DV: `seed=BOOTSTRAP_SEED+i`.

**LIMITAÇÃO FUNDAMENTAL**: todos os três testes são marginais — testam H₀: P(Y|econ=0) = P(Y|econ=1) sem condicionar nos controles. A diferença entre p-valores marginais (este bloco) e condicionais (bloco 3) reflete confundimento por espectro, escolaridade e engajamento — variáveis que diferem entre economistas e não-economistas e que também predizem as DVs.

Todos com correção BH separada por família (3 famílias × 53 testes cada).

### Resultados (pipeline completo, N_PERM=5000):

| Teste | Nominal (p<0,05) | Após BH |
|---|---|---|
| Mann-Whitney U | 9/53 | 1/53 |
| Kolmogorov-Smirnov | 1/53 | 0/53 |
| Permutation (medianas) | 0/53 (mín p=0,128) | 0/53 |

**Único DV sobrevivendo BH em qualquer teste**: `produtividade_aumentando_devagar` via MWU (p=0,0007, p_BH=0,037).

**5 DVs nominais para econ — testes marginais**:

| DV | MWU p | MWU p_BH | KS p | Perm p |
|---|---|---|---|---|
| `produtividade_aumentando_devagar` | 0,0007 | **0,037** | 0,0076 | 0,173 |
| `reforma_previdência_necessária` | 0,048 | 0,283 | n.s. | 0,192 |
| `produtos_importados_benéficos` | 0,040 | 0,266 | n.s. | 0,171 |
| `déficit_federal_grande` | 0,081 | 0,389 | n.s. | 0,265 |
| `empresas_estão_enviando` | 0,277 | n.s. | n.s. | 1,000 |

**Interpretação**: A forte convergência para `produtividade_aumentando_devagar` (ordered logit p=0,017; bootstrap IC não cruza zero; MWU p_BH=0,037) é o sinal mais robusto do TCC. Para os outros 4 DVs nominais do ordered logit, os testes marginais são não-significativos após BH — o que é consistente com efeitos de magnitude moderada (Type M 1,2–1,4×) em grupos pequenos (n_econ=17), onde o poder marginal é ainda menor que o poder condicional (os controles reduzem variância residual, aumentando o poder do ordered logit). A ausência de significância marginal não contradiz os resultados condicionais; reforça que os controles são informativos.

---

## Bloco 9 — Tabelas e LaTeX (`tables.py` + `latex_export.py`)

`tables.py` agrega todos os resultados dos módulos anteriores em 13 CSVs em `outputs/tables/csv/`. Cada CSV tem um propósito:

| CSV | Conteúdo |
|---|---|
| `tabela_sintese_v2.csv` | 53 DVs: β_econ, OR, p_BH, Brant, McFadden |
| `tabela_ame_todos.csv` | AME de econ para todos os 53 DVs |
| `tabela_bootstrap_v2.csv` | BC₀ IC95 para todos os 53 DVs |
| `tabela_permutation.csv` | MWU, KS, permutation p-values + BH |
| `tabela_interacao.csv` | econ×espectro: β, p, AME Q1/Q3 |
| `tabela_power.csv` | Poder vs OR para n=17, 30, 50, 100 |
| `tabela_lpm.csv` | LPM (OLS) como robustez |
| `tabela_descritiva.csv` | N, média, SD, mediana por DV e grupo |
| `tabela_gof.csv` | McFadden, Nagelkerke, AIC, BIC, LR test |
| `tabela_brant.csv` | Brant χ², df, p, PO ok/falha por DV |
| `tabela_ame_chave.csv` | AME detalhado para 4 DVs-chave |
| `tabela_firth.csv` | Ridge-Firth para os 5 DVs singulares |
| `tabela_resumo_blocs.csv` | Contagem de significância por Bloc × preditor |

`latex_export.py` converte cada CSV em um `.tex` pronto para `\input{}`:

- **Formatação ABNT:** vírgula como separador decimal (`DECIMAL_SEP = ","`), booktabs (`\toprule`, `\midrule`, `\bottomrule`), sem linhas verticais.
- **Estrelas de significância:** `*` para p<0,05, `**` p<0,01, `***` p<0,001, aplicadas ao β_econ nominal.
- **`\dag`** para DVs singulares (Firth-Ridge).
- **Escape LaTeX correto:** processamento char-by-char; `_` → `\_\allowbreak{}` (permite quebra de linha após underscore em `p{}` columns).
- **Estrutura visual:** todas as 11 tabelas usam `longtable` com colunas `p{Xcm}` e `\RaggedRight`/`\centering` por coluna. Notas de rodapé via `\begin{minipage}{\linewidth}` após `\end{longtable}` — abordagem sem dependência dos internals de `threeparttablex`.
- **Landscape:** `sintese_v2` (11 cols) e `descritiva` (10 cols) usam `\begin{landscape}` + `\scriptsize` + `\setlength{\tabcolsep}{3.5pt}`. Demais tabelas em portrait com `\footnotesize`.
- **Captions corrigidos:** bootstrap → "BC₀ (B=999)" (não BCa/B=500); firth → "λ=0,1" (não λ=1,0).
- **Compilação limpa:** zero Overfull \hbox; apenas Underfull benignos (última linha de parágrafo curto nas notas).

Os 11 `.tex` resultantes ficam em `outputs/tables/tex/` e podem ser incluídos no TCC com:

```latex
\input{Textuais/analise_v2/outputs/tables/tex/tab_sintese_resumo}
```

---

## Bloco 10 — Figuras (`figures.py`)

**53 gráficos counterfactual** (um por DV): barras empilhadas com distribuição de respostas observadas do público (econ=0), da linha contrafactual (economistas com econ=0 forçado), e dos economistas (econ=1). Permite ver visualmente para onde o efeito da formação "empurra" a distribuição.

**Forest plot (`econ_forest.pdf`):** β_econ ± IC₉₅ BC₀ para todos os 53 DVs em um único gráfico, coloridos por BLOC. Linhas cruzando o zero indicam efeito não significativo. É a figura síntese mais importante do TCC — um leitor pode ver de uma vez quais DVs os economistas diferem do público.

**Heatmap de p-valores:** DVs (linhas) × preditores (colunas), cor por nível de significância. Revela visualmente que o espectro político domina muito mais DVs do que a formação em economia.

**Volcano plot:** β_econ (eixo X) vs −log10(p) (eixo Y). DVs no quadrante superior-direito = economistas marcam mais alto e o efeito é significativo. Superior-esquerdo = marcam mais baixo. Útil para identificar outliers com efeito grande mas não-significativo.

---

## Bloco 11 — Pipeline Master (`run_all.py`)

Orquestra a execução em 8 etapas:

```
1. step_load()          → DataBundle                           [~5s]
2. step_models()        → dict{dv: ModelResult}  (53 DVs)     [~120s]
3. step_diagnostics()   → dict{dv: brant/vif/gof}             [~30s]
4. step_effects()       → dict{dv: ame/or/cf/inter}           [~30s]
5. step_bootstrap()     → DataFrame 53×8                      [~5400s ou skip]
6. step_permutation()   → DataFrame 53×12                     [~600s ou skip]
7. step_tables()        → 13 CSVs + 11 .tex                   [~180s]
8. step_figures()       → 112 figuras                         [~30s]
```

**Nota sobre step_tables:** `build_all_tables()` executa todos os modelos novamente a partir de `bundle` (não reutiliza os `model_results` do passo 2). Isso é intencional — tabelas e pipeline usam as mesmas funções de ajuste, o que garante reprodutibilidade independente de versão. O custo (~180s) é aceitável comparado ao bootstrap.

**Sistema de cache:** `outputs/cache/pipeline_cache.pkl` armazena `bundle`, `model_results`, `diag`, `eff`, `df_boot` e `df_perm`. O benefício principal é evitar re-rodar bootstrap (B=999, ~90 min) e permutation (N=5000, ~10 min). Com `--only tables` ou `--only figures`, as tabelas são carregadas dos **CSVs existentes** via `load_tables_from_csv()` — nenhum modelo é re-executado.

**Flags:**

```bash
python run_all.py                        # tudo completo (~110 min)
python run_all.py --fast                 # sem bootstrap/permutation (~4 min)
python run_all.py --only tables          # relê CSVs + re-exporta .tex (~2s)
python run_all.py --only figures         # relê CSVs + regenera figuras (~10s)
python run_all.py --only effects+tables  # re-roda effects.py + rebuild tabelas (~4 min)
python run_all.py --only tables --no-cache  # força rebuild completo (~4 min)
python run_all.py --compile-tcc          # + tectonic tcc_pronto.tex no final
python run_all.py --url <URL>            # sobrescreve GSHEET_URL de config.py
```

**Semântica do `--no-cache`:** em modo `--only`, força `build_all_tables()` em vez de `load_tables_from_csv()`, re-executando todos os modelos. Útil quando se altera `tables.py`, `model_base.py` ou `effects.py` e se quer recalcular os CSVs sem rodar bootstrap.

---

## Resultado final — o que os números dizem

Juntando tudo, a análise econométrica encontra:

1. **Formação em economia tem efeito estatisticamente detectável em 3–6 DVs** (dependendo do critério): produtividade, déficit federal, produtos importados (confirmados com BC₀), com 2–3 adicionais marginais.

2. **O poder estatístico é a limitação principal**: com n=17, só efeitos grandes (OR≥2,5) são detectáveis com probabilidade >50%. A falta de significância BH não implica ausência de efeito real.

3. **Espectro político domina**: 38 de 53 DVs têm β_espectro significativo. A formação econômica afeta como você pensa sobre política econômica, mas menos do que sua posição ideológica.

4. **O efeito da formação não é moderado pela ideologia** nos DVs principais: economistas de esquerda e de direita diferem do público geral nas mesmas questões centrais (déficit, comércio, produtividade), sugerindo que o consenso econômico transcende o espectro político para esses tópicos.

5. **A suposição de odds proporcionais é válida** em 52 de 53 DVs (Brant test), justificando retroativamente o uso do ordered logit como modelo principal.

---

## Problemas de amostragem

1. **Amostragem não-probabilística (conveniência):** O formulário foi divulgado online por redes do pesquisador. Não há garantia de representatividade da população de economistas brasileiros nem do público geral. Os estimadores são condicionais à amostra obtida — inferência causal para a população requer suposições fortes de seleção aleatória.

2. **n=17 economistas — poder insuficiente:** Com 17 observações no grupo tratamento, o poder para detectar OR=2,0 é apenas 28,2% (Whittemore 1981). A maioria dos efeitos reais de magnitude moderada permanecerá estatisticamente invisível. Isso é uma limitação estrutural, não um problema de código.

3. **Definição estreita de tratamento:** Apenas mestrado e doutorado = `econ=1`. Isso exclui ~80.000 graduados em economia no Brasil. Os resultados descrevem o efeito da formação *avançada*, não da formação econômica em geral.

4. **Contaminação do grupo controle:** O grupo econ=0 inclui pessoas de finanças, contabilidade e administração que têm exposição substancial ao pensamento econômico. Isso atenua o efeito estimado — o "verdadeiro" efeito da formação pode ser maior do que o observado.

5. **Viés de auto-seleção:** Pessoas que aceitam responder um questionário sobre economia e política tendem a ser mais engajadas politicamente e economicamente do que a média. A distribuição de espectro e engajamento pode não refletir a população.

6. **Design desequilibrado (1:8):** A proporção 17:150 reduz a eficiência dos estimadores. O erro padrão de β_econ é dominado pelo grupo menor. Com mais economistas (n≥50), os resultados marginais provavelmente cruzariam o limiar de significância.

7. **Sensibilidade temporal e contextual:** O survey foi realizado em um momento político específico do Brasil. Algumas respostas (especialmente sobre governo atual, preços de combustíveis) refletem o contexto de 2024–2025 e podem não generalizar para outros períodos.

---

## Referências metodológicas

| Referência | Onde se aplica | Localização exata |
|---|---|---|
| Wooldridge (2010) *Introductory Econometrics* | Ordered logit, separação perfeita, controles | §17.1, §17.2 (pp. 587–610) |
| Angrist & Pischke (2009) *Mostly Harmless* | LPM como robustez, seleção de controles | Cap. 3 (pp. 85–110) |
| Davidson & McKinnon (2004) *Econometric Theory* | Bootstrap BC₀/BCa | Cap. 4 (pp. 144–162) |
| Benjamini & Hochberg (1995) *JRSS-B* | Correção FDR/BH | pp. 289–300 |
| Brant (1990) *Biometrics* | Brant test manual | 46(4), pp. 1171–1178 |
| Firth (1993) *Biometrika* | Estimação penalizada (Firth-Ridge) | 80(1), pp. 27–38 |
| McFadden (1974) *Frontiers in Econometrics* | McFadden R² | pp. 105–142 |
| Whittemore (1981) *Applied Statistics* | Power analysis (n=17) | 30(2), pp. 173–176 |
