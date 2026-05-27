"""professor_agent.py — Professor Pesquisador de Econometria.

Varre TODOS os PDFs em material_referencia/, extrai texto, encontra
seções relevantes ao tópico e chama a API da Anthropic para obter
interpretação acadêmica com citações dos livros de referência.

Uso:
    from professor_agent import consult_professor
    commentary = consult_professor(
        topic="Brant test — proportional odds assumption violated",
        context={"chi2": 12.3, "p_value": 0.031, "dv": "corte_impostos"}
    )

Requer: ANTHROPIC_API_KEY definida como variável de ambiente.
Opcional: PROFESSOR_MODEL (padrão claude-haiku-4-5-20251001)
          PROFESSOR_ENABLED=0 para desativar sem remover chamadas.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Caminhos ─────────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).parent
_MAT_DIR     = _THIS_DIR.parent.parent / "material_referencia"
_CACHE_DIR   = _THIS_DIR / "outputs" / "cache" / "professor_pdfs"
_INSIGHTS_DIR = _THIS_DIR / "outputs" / "professor_insights"

# ── Deduplicação por sessão (evita N chamadas para o mesmo tópico) ────────────
_called_topics: set[str] = set()

# ── System prompt do professor ────────────────────────────────────────────────
_SYSTEM_PROMPT = """Você é um professor pesquisador sênior de econometria aplicada,
com domínio profundo de Wooldridge (Introductory Econometrics, 6ª ed.),
Davidson & MacKinnon (Econometric Theory and Methods) e
Angrist & Pischke (Mostly Harmless Econometrics).

Seu papel é fornecer análise técnica rigorosa baseada nos trechos dos livros
fornecidos pelo usuário, com citações precisas (livro, capítulo, página).

Estruture SEMPRE a resposta em:
1. **Fundamento teórico** — o que a literatura diz
2. **Pressupostos e condições** — quando o método/resultado é válido
3. **Interpretação do contexto** — como ler os números fornecidos
4. **Alertas e limitações** — o que pode estar errado
5. **Recomendações** — próximos passos concretos

Seja direto, técnico e conciso. Se um trecho não cobre o tópico,
diga explicitamente e use seu conhecimento geral como complemento.
Cite sempre: (Wooldridge, Cap. X, p. YYY) ou (Angrist & Pischke, Cap. X, p. YYY)."""


# ── Sinônimos por tópico ──────────────────────────────────────────────────────
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "brant":             ["brant", "proportional odds", "parallel regression", "ordinal"],
    "vif":               ["variance inflation", "multicollinearity", "collinear"],
    "bootstrap":         ["bootstrap", "resampling", "bias-corrected", "percentile"],
    "ame":               ["marginal effect", "average marginal", "partial effect"],
    "ordered logit":     ["ordered logit", "proportional odds", "ordinal regression", "cumulative"],
    "ridge":             ["ridge", "penalized", "regularization", "firth", "separation"],
    "permutation":       ["permutation", "randomization", "non-parametric", "mann-whitney"],
    "power":             ["power analysis", "sample size", "statistical power", "type ii"],
    "lpm":               ["linear probability", "ordinary least squares", "ols", "robust"],
    "interaction":       ["interaction", "moderation", "heterogeneous", "slope"],
    "multiple testing":  ["bonferroni", "holm", "benjamini", "false discovery", "fwer", "fdr"],
    "separation":        ["complete separation", "perfect separation", "quasi-perfect", "sparse"],
    "goodness of fit":   ["mcfadden", "nagelkerke", "aic", "bic", "log-likelihood", "pseudo r"],
    "counterfactual":    ["counterfactual", "potential outcome", "treatment effect", "causal"],
}


def _get_pdf_files() -> list[Path]:
    if not _MAT_DIR.exists():
        log.warning("Professor: pasta material_referencia não encontrada em %s", _MAT_DIR)
        return []
    pdfs = [p for p in _MAT_DIR.glob("*.pdf") if "Zone.Identifier" not in p.name]
    log.info("Professor: %d PDF(s) encontrados em %s", len(pdfs), _MAT_DIR)
    return pdfs


def _cache_path(pdf_path: Path) -> Path:
    safe_stem = re.sub(r"[^\w]", "_", pdf_path.stem)[:60]
    return _CACHE_DIR / f"{safe_stem}.txt"


def _extract_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        log.error("Professor: pdfplumber não instalado. Execute: pip install pdfplumber")
        return ""

    log.info("Professor: extraindo texto de '%s' (pode demorar na 1ª vez)...", pdf_path.name)
    parts: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(f"[pag:{i + 1}] {text}")
                if i > 0 and i % 200 == 0:
                    log.info("Professor: %d páginas extraídas de '%s'...", i + 1, pdf_path.name)
    except Exception as exc:
        log.warning("Professor: falha ao extrair '%s': %s", pdf_path.name, exc)
        return ""

    return "\n".join(parts)


def _load_or_extract(pdf_path: Path) -> str:
    cache = _cache_path(pdf_path)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text = _extract_text(pdf_path)
    if text:
        cache.write_text(text, encoding="utf-8")
        log.info("Professor: cache salvo → %s", cache.name)
    return text


def _load_all_pdfs() -> dict[str, str]:
    return {p.name: _load_or_extract(p) for p in _get_pdf_files()}


def _build_keywords(topic: str) -> list[str]:
    topic_lower = topic.lower()
    kws: list[str] = []

    # Palavras do próprio tópico (≥3 chars)
    for word in re.split(r"\W+", topic_lower):
        if len(word) >= 3:
            kws.append(word)

    # Sinônimos mapeados
    for key, syns in _TOPIC_KEYWORDS.items():
        if key in topic_lower:
            kws.extend(syns)

    return list(dict.fromkeys(kws))  # mantém ordem, remove duplicatas


def _find_relevant_excerpts(
    texts: dict[str, str],
    keywords: list[str],
    context_lines: int = 35,
    max_blocks_per_pdf: int = 5,
    max_total_chars: int = 45_000,
) -> str:
    excerpts: list[str] = []

    for pdf_name, text in texts.items():
        if not text:
            continue
        lines = text.split("\n")
        hit_idx: set[int] = set()

        for i, line in enumerate(lines):
            lower = line.lower()
            if any(kw in lower for kw in keywords):
                lo = max(0, i - context_lines)
                hi = min(len(lines), i + context_lines)
                hit_idx.update(range(lo, hi))

        if not hit_idx:
            continue

        # Agrupa índices consecutivos em blocos
        sorted_idx = sorted(hit_idx)
        blocks: list[list[int]] = []
        blk = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if idx - blk[-1] <= 3:
                blk.append(idx)
            else:
                blocks.append(blk)
                blk = [idx]
        blocks.append(blk)

        block_texts: list[str] = []
        for blk in blocks[:max_blocks_per_pdf]:
            block_texts.append("\n".join(lines[i] for i in blk))

        short_name = pdf_name[:50]
        excerpts.append(f"\n\n=== {short_name} ===\n" + "\n---\n".join(block_texts))

    combined = "".join(excerpts)
    if len(combined) > max_total_chars:
        combined = combined[:max_total_chars] + "\n...[truncado para caber no contexto]"
    return combined


def _save_insight(topic: str, commentary: str, context: Optional[dict]) -> None:
    _INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"\W+", "_", topic)[:40].strip("_")
    fpath = _INSIGHTS_DIR / f"{ts}_{slug}.md"

    lines = [f"# Professor: {topic}\n"]
    if context:
        ctx_str = json.dumps(context, indent=2, default=str)
        lines.append(f"## Contexto\n```json\n{ctx_str}\n```\n")
    lines.append(f"## Análise\n{commentary}\n")

    fpath.write_text("\n".join(lines), encoding="utf-8")
    log.info("Professor: insight salvo → %s", fpath.name)


def consult_professor(
    topic: str,
    context: Optional[dict] = None,
    force: bool = False,
    save: bool = True,
) -> str:
    """
    Consulta o professor pesquisador de econometria.

    Args:
        topic:   Tópico ou questão (ex: "Brant test violado — p=0.03").
        context: Dicionário com resultados numéricos para contextualizar.
        force:   Se True, ignora deduplicação por sessão.
        save:    Se True, salva commentary em outputs/professor_insights/.

    Returns:
        String com commentary acadêmica, ou "" em caso de falha/desativado.
    """
    # ── Guarda de sessão ──────────────────────────────────────────────────────
    if not force and topic in _called_topics:
        log.debug("Professor: tópico '%s' já consultado nesta sessão — ignorando.", topic)
        return ""
    _called_topics.add(topic)

    # ── Verificações de ambiente ──────────────────────────────────────────────
    if os.environ.get("PROFESSOR_ENABLED", "1") == "0":
        log.info("Professor: desativado via PROFESSOR_ENABLED=0.")
        return ""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning(
            "Professor: ANTHROPIC_API_KEY não definida — consulta ignorada.\n"
            "  Defina com: export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return ""

    model = os.environ.get("PROFESSOR_MODEL", "claude-haiku-4-5-20251001")

    try:
        import anthropic
    except ImportError:
        log.error("Professor: pacote 'anthropic' não instalado. Execute: pip install anthropic")
        return ""

    try:
        # 1. Carrega textos de todos os PDFs
        pdf_texts = _load_all_pdfs()
        if not pdf_texts:
            log.warning("Professor: nenhum PDF carregado — consultando sem trechos.")

        # 2. Encontra trechos relevantes
        keywords = _build_keywords(topic)
        excerpts = _find_relevant_excerpts(pdf_texts, keywords)

        # 3. Monta mensagem
        ctx_str = ""
        if context:
            ctx_str = "\n\n**Contexto numérico:**\n```json\n"
            ctx_str += json.dumps(context, indent=2, default=str)
            ctx_str += "\n```"

        user_msg = (
            f"**Tópico:** {topic}"
            f"{ctx_str}\n\n"
            f"**Trechos relevantes dos livros de referência:**\n"
            f"{excerpts if excerpts else '(nenhum trecho encontrado para este tópico)'}\n\n"
            "Forneça sua análise completa seguindo a estrutura solicitada, "
            "citando livro, capítulo e página para cada afirmação central."
        )

        # 4. Chama API
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1800,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        commentary: str = response.content[0].text

        # 5. Loga com destaque
        sep = "=" * 64
        log.info("\n%s\nPROFESSOR — %s\n%s\n%s\n%s", sep, topic, sep, commentary, sep)

        if save:
            _save_insight(topic, commentary, context)

        return commentary

    except Exception as exc:
        log.warning("Professor: erro na consulta '%s': %s", topic, exc)
        return ""
