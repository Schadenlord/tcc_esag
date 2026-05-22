# Contexto da Auditoria — TCC "Racionalidade Coletiva em Xeque"
**UDESC/ESAG, 2025 | Gerado em 2026-05-21**

---

## 1. Identidade do projeto

| Campo | Valor |
|---|---|
| Repositório | `/home/bruno_tcc/tcc_esag` |
| Notebook original | `Textuais/analise/analise.ipynb` |
| Notebook corrigido (parcial) | `Textuais/analise/analise_corrigida.ipynb` |
| Dados | Google Sheets — URL pública de edição: `https://docs.google.com/spreadsheets/d/1kn6u7jnv-ktwiY0TMYxYWnQhr54dWOYSOYvrqf9W3WY/edit?usp=sharing` |
| URL CSV para o notebook (`GSHEET_URL`) | `https://docs.google.com/spreadsheets/d/1kn6u7jnv-ktwiY0TMYxYWnQhr54dWOYSOYvrqf9W3WY/export?format=csv` |
| LaTeX principal | `tcc_pronto.tex` (165 linhas) |
| Metodologia | `Textuais/metodologia.tex` |
| Resultados | `Textuais/resultadosv2.tex` |
| Apêndices | `PosTextuais/Apendices.tex` |
| Saída dos modelos | `Textuais/analise/analises_ordinais.txt` |
| Tabela síntese | `Textuais/analise/tabela_sintese.csv` |

### Python disponível
- `/usr/bin/python3` versão **3.14.4**
- **Sem pip, sem conda, sem Jupyter instalado** no ambiente atual
- `statsmodels`, `seaborn`, `nbconvert` **não estão instalados**
- É necessário instalar dependências antes de executar o notebook
  - Sugestão: `sudo apt-get install -y python3-pip` e depois `pip install statsmodels seaborn openpyxl nbconvert`

### LaTeX
- `pdflatex`, `xelatex`, `biber` **não encontrados** no PATH atual
- `texlive-full` foi instalado via `sudo apt-get install` em sessão anterior, possivelmente ainda não disponível no PATH ou ainda instalando
- Verificar com: `which pdflatex` ou `tex --version`

---

## 2. Diagnóstico completo (já concluído)

### CRÍTICOS

**C1 — Hessianas singulares em 9 variáveis**
Modelos com erros-padrão NaN (inferência inválida):
`impostos_muito_altos`, `déficit_federal_grande`, `corte_impostos`, `aumento_uso_tecnologia`, `acordos_comerciais_outros`, `acha_preços_combustíveis`, `acha_presidente_pode`, `filhos_menos_30`, `competição_entre_empresas`

O TCC reporta p-valores significativos (bold) para 5 deles sem suporte numérico reproduzível:
- `impostos_muito_altos`: espectro p=0.000 bold
- `déficit_federal_grande`: espectro p=0.000 bold, econ p=0.026 bold
- `corte_impostos`: espectro e econ p=0.000 bold
- `acha_presidente_pode`: espectro p=0.042 bold
- `competição_entre_empresas`: espectro p=0.000 bold

**C2 — Tabela-Síntese (Apêndice C) não reproduzível**
- N no TCC: 169–172 (coletado durante a pesquisa)
- N no repositório (dataset final): 182–183
- Médias divergem (ex: `acha_novos_postos`: TCC 0.526/0.500 vs CSV 0.549/0.471)

**C3 — Colisão de acrônimos (`algumas_pessoas_dizem`)**
Duas perguntas distintas recebem o mesmo acrônimo via `make_acronym`:
1. "Algumas pessoas dizem que estes são **tempos economicamente instáveis**..." (posição 23 na lista)
2. "Algumas pessoas dizem que, para ter uma vida confortável, a **família média deve ter dois assalariados**..." (posição 32 na lista — duplicata na `dependentes_cols`)

Consequências:
- Loop `analises_ordinais.txt` pula ambas por erro de coluna duplicada
- `gerar_tabela_sintese` analisa sempre a 1ª ocorrência (tempos instáveis) mas o TCC rotula como "família com dois salários" → **mislabeling no Apêndice C, item 24**

**C4 — Sentido do contrafactual invertido na seção 5.1**
- O código calcula: economistas (econ=1) com econ→0 → "como economistas responderiam sem formação"
- `metodologia.tex` seção 3.5: descreve corretamente
- `resultadosv2.tex` seção 5.1: descreve o **oposto** ("como leigos responderiam com formação de economistas")
- Rótulo nos gráficos: `'Público esclarecido (CF)'` é enganoso

---

### MODERADOS

**M1 — Separação quase-perfeita em variáveis de raça**
Pelo menos `temos_imigrantes_demais` (coef raça ~10, SE ~197) e `mulheres_minorias_têm` (coef ~10, SE ~314) apresentam separação quase-perfeita. Modelos convergem tecnicamente, mas estimativas de raça são inúteis.

**M2 — Duplicata em `dependentes_cols` (célula `c8fa54ad`)**
`algumas_pessoas_dizem` aparece nas posições 23 e 32. Deve ser 52 itens únicos.

**M3 — FDR/Benjamini-Hochberg não implementado**
Estava nas "Etapas Analíticas" mas foi comentado no LaTeX e nunca implementado. Com 52 testes simultâneos, multiplicidade não controlada. TCC não menciona como limitação.

**M4 — Fórmula (4) usa indexação 1…J, código usa 0,1,2**
Inconsistência entre notação matemática (categorias 1,2,3) e implementação (classes 0,1,2).

---

### MENORES

**m1 — Colinearidade estrutural econ × Pós-graduação**
econ=1 implica pós-graduação em Economia, sobrepondo-se com dummy de escolaridade. Contribui para instabilidade do Hessiano.

**m2 — "Independente" codificado como 3 no espectro político**
Maior que "Extrema direita"=2, criando distorção não-linear em variável tratada como numérica.

**m3 — Seção 4.4 vs 5.1 (Brant): sem inconsistência**
Ambas afirmam que Brant não foi implementado. Consistente — não é um problema.

---

## 3. O que foi feito nesta sessão

### Notebook corrigido (`analise_corrigida.ipynb`) — 28 células

O notebook foi criado a partir do original com os seguintes fixes aplicados e **verificados**:

| Fix | Severidade | Célula | Status |
|---|---|---|---|
| **FIX-1**: Resolver colisão de acrônimos com sufixo `_2`, `_3` após `df_dependentes.rename()` | Crítico | `73d99ec3` | ✅ Aplicado |
| **FIX-2**: Corrigir `dependentes_cols` — posição 32 de `'algumas_pessoas_dizem'` → `'algumas_pessoas_dizem_2'` | Crítico | `c8fa54ad` | ✅ Aplicado |
| **FIX-3**: Label contrafactual `'Público esclarecido (CF)'` → `'Economistas s/ formação (CF)'` | Crítico | `35e0c3f2`, `3c2222d6` | ✅ Aplicado |
| **FIX-4a**: Diagnóstico de Hessiana singular em `analisar_variavel_para_latex_log` | Moderado | `3c2222d6` | ✅ Aplicado |
| **FIX-4b**: Diagnóstico de Hessiana singular em `gerar_tabela_sintese` | Moderado | `f3405e6b` | ✅ Aplicado |
| **FIX-5**: Correção FDR/BH (Benjamini-Hochberg) — nova célula `fdr_bh_fix5` | Moderado | nova célula [23] | ✅ Aplicado |
| **FIX-6**: Diagnóstico de colinearidade econ × Pós-graduação — nova célula `collinearity_fix6` | Menor | nova célula [24] | ✅ Aplicado |

### O que NÃO foi feito (bloqueado por falta de ambiente Python)

- ❌ **Execução do notebook** — `statsmodels` não instalado, `pip` não disponível
- ❌ **Verificação dos outputs** — impossível sem executar
- ❌ **Relatório `relatorio_auditoria.md`** (Tarefa 2) — não iniciado
- ❌ **Compilação LaTeX** (Tarefa 3) — `pdflatex`/`xelatex` não encontrados no PATH

---

## 4. O que precisa ser feito a seguir

### Prioridade 1 — Configurar ambiente e executar o notebook

```bash
# Instalar pip
sudo apt-get install -y python3-pip

# Instalar dependências
pip3 install statsmodels seaborn openpyxl nbconvert matplotlib pandas numpy

# Definir URL dos dados
export GSHEET_URL="https://docs.google.com/spreadsheets/d/1kn6u7jnv-ktwiY0TMYxYWnQhr54dWOYSOYvrqf9W3WY/export?format=csv"

# Executar o notebook corrigido
cd /home/bruno_tcc/tcc_esag/Textuais/analise
jupyter nbconvert --to notebook --execute analise_corrigida.ipynb --output analise_corrigida_executado.ipynb --ExecutePreprocessor.timeout=600
```

### Prioridade 2 — Após execução, verificar

1. Confirmar que `algumas_pessoas_dizem` e `algumas_pessoas_dizem_2` são colunas distintas
2. Confirmar que o log mostra `HESSIANA SINGULAR` para os 9 itens esperados
3. Verificar contagem de significantes antes/depois da correção BH
4. Verificar correlação econ × Pós-graduação no Fix-6
5. Salvar nova `tabela_sintese.csv` e `tabela_sintese.xlsx`

### Prioridade 3 — Escrever `relatorio_auditoria.md`

Arquivo: `Textuais/analise/relatorio_auditoria.md`

Estrutura requerida:
1. **Sumário executivo** (3–5 parágrafos)
2. **Tabela de problemas**: ID, Severidade, Localização exata, Descrição, Impacto nas conclusões, Status
3. **Comparação antes/depois** para cada correção
4. **Limitações remanescentes irremediáveis**: n≈17 economistas, amostragem não probabilística, NaN irredutíveis
5. **"O que o TCC pode e não pode afirmar"**: qualificando H1–H7 à luz dos problemas

### Prioridade 4 — Compilar LaTeX

```bash
# Verificar se texlive está disponível
which pdflatex || hash -r && which pdflatex

# Se não, instalar
sudo apt-get install -y texlive-full

# Compilar (2x para referências cruzadas)
cd /home/bruno_tcc/tcc_esag
pdflatex -interaction=nonstopmode tcc_pronto.tex
biber tcc_pronto
pdflatex -interaction=nonstopmode tcc_pronto.tex
pdflatex -interaction=nonstopmode tcc_pronto.tex
```

---

## 5. Estrutura do notebook corrigido (28 células)

```
[0]  3519e444   markdown  Título / objetivo
[1]  f1d84610   markdown  Seção: Importações
[2]  20ec9235   code      Imports (pandas, numpy, statsmodels, etc.)
[3]  6a747f7a   code      Lê dados via GSHEET_URL (os.getenv)
[4]  8e931956   code      Lista colunas
[5]  0ea6c208   code      (comentado) drop de colunas não usadas
[6]  9cb347a6   markdown  Seção: Tratamento das variáveis
[7]  c0e8f757   code      Codificações manuais + dummies de controle
[8]  2c0b5a7b   markdown  Seção: Variáveis dependentes
[9]  73d99ec3   code      detect_number, make_acronym, rename + FIX-1
[10] 17c48bb0   code      Exibe acronimos (original)
[11] ecaa1088   code      Inverte dicionário acronimos
[12] aa61d612   code      Teste de lookup (propositalmente falha)
[13] c90ed509   markdown  Seção: Join controles + dependentes
[14] de5118e7   code      df_final = pd.concat(...)
[15] c7377efd   markdown  Seção: Modelo Logit Ordenado
[16] 35e0c3f2   code      analisar_variavel_para_latex (versão antiga) + FIX-3
[17] c8fa54ad   code      dependentes_cols (lista hardcoded, 53 itens) + FIX-2
[18] 3c2222d6   code      analisar_variavel_para_latex_log + loop + FIX-3 + FIX-4a
[19] fa6ef259   code      Testes individuais de análise
[20] 0d92f572   code      Escreve analises_ordinais.txt
[21] 6b2acaa8   code      Drop de colunas extras de df_dependentes
[22] f3405e6b   code      gerar_tabela_sintese + chamada principal + FIX-4b
[23] fdr_bh_fix5 code     FIX-5: Correção FDR/BH (NOVA CÉLULA)
[24] collinearity_fix6 code FIX-6: Diagnóstico colinearidade (NOVA CÉLULA)
[25] c8d17a99   code      Salva tabela_sintese.xlsx
[26] 65148bc2   code      Formata p-valores
[27] 019859bb   code      Salva tabela_sintese.csv
```

---

## 6. Hipóteses do TCC e status pós-auditoria

| H | Enunciado | Status após auditoria |
|---|---|---|
| H1 | Economistas diferem do público em percepções econômicas | **Parcialmente suportada** — mas 5 dos itens com p<0.05 reportados têm Hessiana singular (inferência inválida) |
| H2 | Espectro político prediz percepções | **Parcialmente suportada** — mesmos problemas de NaN em 5 itens |
| H3 | Formação econômica reduz "erros" perceptivos | **Não testável rigorosamente** — n≈17 economistas, sem correção múltipla |
| H4–H7 | (específicas por tema) | **Dependem da re-execução** com dataset correto e n final confirmado |

---

## 7. Decisões de design não corrigíveis

1. **n≈17 economistas** — poder estatístico muito baixo para subgrupo; qualquer estimativa de `econ` é instável
2. **Amostragem não probabilística** (snowball/conveniência) — generalização impossível
3. **NaN irredutíveis** — 9 itens têm Hessiana singular por design (resposta muito concentrada + muitos regressores)
4. **"Independente" como 3** no espectro político — cria não-linearidade; solução seria dummy separada, mas muda toda a análise
5. **Fórmula (4) do TCC** — usa índices 1…J mas código usa 0,1,2; inconsistência de notação, não de cálculo

---

## 8. Pedido do usuário (mensagens da sessão atual, não atendidas)

O usuário pediu:
> "pegue os dados direto de lá e rode o código completo vendo os outputs para arrumar tudo na melhor pratica econometrica"

Ou seja: **executar o notebook corrigido com os dados reais** (via `GSHEET_URL`), ver os outputs, e fazer ajustes adicionais segundo boas práticas econométricas. Isso requer ambiente Python funcional (ver Prioridade 1 acima).

---

## 9. Notas técnicas do ambiente

- OS: Linux 6.6.87.2 (WSL2)
- Shell: bash
- Python: 3.14.4 em `/usr/bin/python3`
- pip: **NÃO instalado**
- Jupyter: **NÃO instalado**
- LaTeX: **NÃO encontrado no PATH** (texlive-full pode ter sido instalado em sessão anterior — verificar com `hash -r && which pdflatex`)
- Git branch atual: `main`
