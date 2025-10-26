# Trabalho de Conclusão de Curso — Repositório do Projeto

Este repositório contém a **fonte LaTeX**, **bibliografia** e **artefatos** do TCC:

**RACIONALIDADE COLETIVA EM XEQUE: UMA INVESTIGAÇÃO COMPORTAMENTAL SOBRE PERCEPÇÕES ECONÔMICAS NO BRASIL**  
Curso de Ciências Econômicas — **UDESC/ESAG**  
**Orientadora:** Marianne Zwiling Stampe (UDESC)  
Ano: **2025**

O objetivo deste README é orientar a **reprodução local** do PDF, a **verificação pelos avaliadores** e a **revisão transparente** do material.

## Sumário
- Descrição
- Contexto acadêmico
- Estrutura do repositório (visão geral)
- Requisitos e ambiente de compilação
- Instruções de compilação (Windows / PowerShell)
- Metadados da pesquisa (para avaliadores)
- Arquivos principais e propósito
- Checklist rápido para avaliadores
- Como citar este repositório (BibTeX)
- Contribuições, licença e contato

---

## Descrição
Este repositório reúne os **arquivos-fonte LaTeX**, o **`.bst` ABNT**, o **`.bib`** e os **materiais suplementares** (códigos/saídas) que geram o PDF final do TCC com `abntex2`. Use-o para **revisar** o conteúdo, **recompilar** o documento e **inspecionar** anexos, tabelas e imagens.

## Contexto acadêmico
Material de apoio ao TCC apresentado ao Curso de Graduação em **Ciências Econômicas** (UDESC/ESAG). A versão foi preparada para **reprodutibilidade** por banca e avaliadores. Os principais arquivos-fonte estão na raiz e nas pastas `PreTextuais/`, `Textuais/` e `PosTextuais/`.

## Estrutura do repositório (visão geral)

- `tcc_pronto.tex` — arquivo principal (entry point).
- `PacotesBasicos.tex` — carregamento e configuração de pacotes.
- `abntex2-alf_revNBR2023.bst` — estilo ABNT (BibTeX) utilizado.
- `referencias.bib` — base bibliográfica em BibTeX (UTF-8).
- `PreTextuais/` — capa, folha de rosto, resumo(s), sumário etc.
- `Textuais/` — capítulos: introdução, teoria, método, resultados, conclusão etc.
- `PosTextuais/` — apêndices, anexos e materiais adicionais.
- `Textuais/analise/` — **códigos e saídas** de análise (ex.: `analise.ipynb`, tabelas, imagens).

> Observação: artefatos temporários de compilações (`*.aux`, `*.bbl`, `*.blg`, `*.ilg`, `*.lof`, `*.lot`, `*.out`, `*.synctex*`) **não são necessários** para compilar do zero.

## Requisitos e ambiente de compilação

Recomendado:
- **TeX Live** (ou MiKTeX) atualizado (≥ 2020).
- **BibTeX** (o projeto usa `.bst` próprio).
- **latexmk** (para compilar tudo de forma automática).

Dicas:
- Windows: instale TeX Live (https://www.tug.org/texlive/) ou MiKTeX (https://miktex.org/).
- Overleaf: é possível importar o repositório e compilar online (defina `tcc_pronto.tex` como principal).

## Instruções de compilação (Windows / PowerShell)

**Opção A — com `latexmk` (recomendado)**

```powershell
latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode -synctex=1" tcc_pronto.tex
