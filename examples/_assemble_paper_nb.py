# -*- coding: utf-8 -*-
"""Monta paper_explorations.ipynb a partir dos scripts jupytext dos agentes."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = ["_paper_exp123.py", "_paper_exp78.py", "_paper_exp469.py", "_paper_exp5.py"]


def parse_percent(path):
    """Converte um script em formato percent numa lista de células."""
    text = path.read_text(encoding="utf-8")
    cells = []
    cur_type, cur_lines = None, []

    def flush():
        nonlocal cur_lines, cur_type
        if cur_type is None:
            cur_lines = []
            return
        src = "\n".join(cur_lines).strip("\n")
        # remove config de backend (o notebook usa inline)
        src = "\n".join(l for l in src.splitlines()
                        if "matplotlib.use(" not in l)
        src = src.strip("\n")
        if src:
            if cur_type == "markdown":
                # tira o prefixo de comentario "# " das linhas
                src = "\n".join(re.sub(r"^# ?", "", l) for l in src.splitlines())
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": src.splitlines(keepends=True)})
            else:
                cells.append({"cell_type": "code", "metadata": {},
                              "execution_count": None, "outputs": [],
                              "source": src.splitlines(keepends=True)})
        cur_lines = []

    for line in text.splitlines():
        m = re.match(r"^# %%\s*(\[markdown\])?\s*$", line)
        if m:
            flush()
            cur_type = "markdown" if m.group(1) else "code"
        else:
            if cur_type is None and line.strip() in ("", "# -*- coding: utf-8 -*-"):
                continue
            if cur_type is None:
                cur_type = "code"
            cur_lines.append(line)
    flush()
    return cells


intro = """# Paper explorations — a geometria das direções compartilhadas

Este notebook reúne os **9 experimentos** propostos para o paper sobre o
espectro θ da GSVD em espaços de word embeddings multilíngues (PT-BR × EN,
MUSE/fastText alinhados, 300d). O frame conjunto é o mesmo do
`demo_word_embeddings.ipynb`, encapsulado em `paper_common.py`: cada direção
da matriz `H` tem um ângulo θ ∈ [0°, 90°] — θ baixo = direção específica do
português, θ ≈ 45° = compartilhada, θ alto = específica do inglês.

Organização (os números dos experimentos seguem a proposta original):

| Seção | Experimentos | Pergunta |
|---|---|---|
| A | 1, 2, 3 | O que cada banda do espectro *sabe fazer*? (analogias, band-pass, retrieval) |
| B | 7, 8 | O que as direções *significam*? (coerência de tradução, entidades) |
| C | 4, 6, 9 | Generalização e controles (outras línguas, não-alinhado, diminutivos informais) |
| D | 5 | mBERT camada a camada |

Pré-requisitos: caches em `~/.gsvdlib/` (criados por `_fetch_muse.py` e
`_fetch_paper_data.py`) e o cache do mBERT (criado na primeira execução do
experimento 5). Resultados numéricos também ficam em `paper_results/*.json`.
"""

cells = [{"cell_type": "markdown", "metadata": {},
          "source": intro.splitlines(keepends=True)}]
section_titles = {
    "_paper_exp123.py": "# Seção A — Experimentos 1–3",
    "_paper_exp78.py": "# Seção B — Experimentos 7–8",
    "_paper_exp469.py": "# Seção C — Experimentos 4, 6 e 9",
    "_paper_exp5.py": "# Seção D — Experimento 5 (mBERT)",
}
for name in SCRIPTS:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [section_titles[name]]})
    cells.extend(parse_percent(HERE / name))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"p{i}"

# valida sintaxe de todas as celulas de codigo
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")

out = HERE / "paper_explorations.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
ncode = sum(1 for c in cells if c["cell_type"] == "code")
print(f"ok: {len(cells)} celulas ({ncode} de codigo) -> {out}")
