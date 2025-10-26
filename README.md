# Racionalidade Coletiva em Xeque: Uma Investigação Comportamental sobre Percepções Econômicas no Brasil

## Sobre o Projeto

Este repositório contém o código-fonte LaTeX do Trabalho de Conclusão de Curso (TCC) desenvolvido por **Bruno Francisco Schaden**, sob orientação da **Profa. Marianne Zwiling Stampe**, apresentado ao **Curso de Ciências Econômicas** do **Centro de Ciências da Administração e Socioeconômicas (ESAG)** da **Universidade do Estado de Santa Catarina (UDESC)**.

## Resumo

Este trabalho investiga como vieses de julgamento moldam crenças econômicas e, por consequência, preferências por políticas públicas no Brasil. Partindo do paradoxo da racionalidade coletiva — entre o que a evidência recomenda e o que a esperança deseja —, adapta-se a *Survey of Americans and Economists on the Economy* (SAEE) ao contexto brasileiro e operacionaliza-se um contrafactual de "público contrafactual" (não-economistas com perfil socioeconômico/ideológico próximo ao de economistas), a fim de isolar o papel do conhecimento técnico frente à ideologia.

O desenho empírico combina survey digital (amostragem não probabilística) e modelos logit binário/ordenado com controles demográficos e ideológicos, mantendo especificações comparáveis para todos os itens. Os resultados apontam padrões consistentes com quatro famílias de viés (antimercado, antiestrangeiro, antitrabalho e pessimismo) e indicam que maior literacia econômica está associada a respostas mais próximas ao consenso técnico, embora efeitos ideológicos permaneçam substantivos em temas moralizados e distributivos.

## Palavras-chave

Economia política comportamental; Vieses cognitivos; Crenças econômicas; Logit ordenado; Público contrafactual; Educação econômica; Racionalidade coletiva.

## Estrutura do Repositório

```
.
├── tcc_pronto.tex           # Arquivo principal do documento LaTeX
├── tcc_pronto.pdf           # Documento compilado (PDF)
├── PacotesBasicos.tex       # Definição de pacotes LaTeX utilizados
├── referencias.bib          # Referências bibliográficas (BibTeX)
├── PreTextuais/             # Elementos pré-textuais
│   ├── Capa.tex
│   ├── FolhadeRosto.tex
│   ├── Resumo.tex
│   ├── Abstract.tex
│   └── ...
├── Textuais/                # Capítulos do trabalho
│   ├── introducao.tex
│   ├── revisao_literatura.tex
│   ├── metodologia.tex
│   ├── resultadosv2.tex
│   ├── conclusão.tex
│   └── analise/
└── PosTextuais/             # Elementos pós-textuais
```

## Metodologia

A pesquisa combina três eixos metodológicos principais:

1. **Análise histórico-teórica**: Revisão da literatura de economia comportamental, história do pensamento econômico e racionalidade limitada.

2. **Survey adaptado**: Adaptação da SAEE (*Survey of Americans and Economists on the Economy*) ao contexto brasileiro, aplicada via Google Forms entre agosto e outubro de 2025.

3. **Modelagem econométrica**: Aplicação de modelos logit binário e ordenado para estimar o impacto marginal de fatores individuais na propensão a adotar crenças divergentes do consenso técnico, utilizando Python (pandas, statsmodels).

### Amostragem

- **Tipo**: Não probabilística, com estratificação em dois grupos
- **Tamanho**: n = 183 participantes válidos
  - Grupo controle (não-economistas): ~90,7% (≈166)
  - Grupo de tratamento (economistas com pós-graduação stricto sensu): ~9,3% (≈17)
- **Período de coleta**: Agosto a outubro de 2025

### Aspectos Éticos

A pesquisa foi aprovada pelo Comitê de Ética em Pesquisa com Seres Humanos da UDESC, sob o número **CAAE: 89374225.9.0000.0118** (Parecer nº 7.719.326), em conformidade com a Resolução nº 510/2016 do Conselho Nacional de Saúde e a LGPD.

## Hipóteses Testadas

O estudo propõe e testa cinco hipóteses principais:

- **H1**: Eleitores brasileiros exibem vieses sistemáticos que distorcem sua percepção de fenômenos econômicos
- **H2**: Conhecimento econômico específico reduz a probabilidade de manifestação de vieses cognitivos
- **H3**: O viés antimercado está associado ao apoio a políticas intervencionistas
- **H4**: O viés antiestrangeiro está associado ao apoio a restrições comerciais e migratórias
- **H5**: O viés antitrabalho está associado ao apoio a políticas que priorizam a criação direta de empregos

## Compilação do Documento

Para compilar o documento LaTeX:

```bash
pdflatex tcc_pronto.tex
bibtex tcc_pronto
pdflatex tcc_pronto.tex
pdflatex tcc_pronto.tex
```

Ou utilize sua ferramenta LaTeX preferida (TeXstudio, Overleaf, etc.).

### Requisitos

- Distribuição LaTeX completa (TeX Live, MiKTeX)
- Classe `abntex2` (para formatação ABNT)
- Pacotes LaTeX listados em `PacotesBasicos.tex`

## Principais Contribuições

1. Proposta de um arcabouço mensurável e comparável para crenças econômicas no contexto brasileiro
2. Explicitação de critérios de refutabilidade e testes severos
3. Discussão de limites de validade externa e robustez
4. Derivação de implicações para educação econômica e desenho institucional

## Referências Principais

- Caplan, B. (2007). *The Myth of the Rational Voter: Why Democracies Choose Bad Policies*
- Kahneman, D. (2011). *Thinking, Fast and Slow*
- Downs, A. (1957). *An Economic Theory of Democracy*
- Survey of Americans and Economists on the Economy (SAEE, 1996)

## Autor e Orientação

- **Autor**: Bruno Francisco Schaden
- **Orientadora**: Profa. Marianne Zwiling Stampe
- **Instituição**: Universidade do Estado de Santa Catarina (UDESC)
- **Centro**: Centro de Ciências da Administração e Socioeconômicas (ESAG)
- **Curso**: Ciências Econômicas
- **Local**: Florianópolis, SC
- **Ano**: 2025

## Citação

Para citar este trabalho:

```
SCHADEN, Bruno Francisco. Racionalidade Coletiva em Xeque: Uma Investigação 
Comportamental sobre Percepções Econômicas no Brasil. 2025. Trabalho de Conclusão 
de Curso (Graduação em Ciências Econômicas) – Centro de Ciências da Administração 
e Socioeconômicas, Universidade do Estado de Santa Catarina, Florianópolis, 2025.
```

### BibTeX

```bibtex
@mastersthesis{schaden2025racionalidade,
  author  = {Schaden, Bruno Francisco},
  title   = {Racionalidade Coletiva em Xeque: Uma Investigação Comportamental sobre Percepções Econômicas no Brasil},
  school  = {Universidade do Estado de Santa Catarina},
  year    = {2025},
  address = {Florianópolis, SC},
  type    = {Trabalho de Conclusão de Curso (Graduação em Ciências Econômicas)}
}
```

## Licença

Este trabalho acadêmico segue as normas de propriedade intelectual da UDESC e está disponível para fins educacionais e de pesquisa.

## Contato

Para questões sobre o trabalho, entre em contato com o autor ou com a orientadora através dos canais institucionais da UDESC/ESAG.

---

**Nota**: Este repositório contém material acadêmico produzido como requisito parcial para obtenção do título de Bacharel em Ciências Econômicas pela UDESC.
