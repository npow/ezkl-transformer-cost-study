"""Convert paper/main.tex to a publish.fun package (Markdown body + metadata).

usage:  uv run --with pypandoc_binary python paper/make_publishfun.py [out_dir]

Writes body.md, meta.json, keywords.json.  Sends nothing.  publish.fun accepts
Markdown only ($...$ math, GFM tables), abstract <= 3000 chars, body <= 150k.
"""
import json
import pathlib
import re
import sys

import pypandoc

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "publishfun"
OUT.mkdir(parents=True, exist_ok=True)
src = (HERE / "main.tex").read_text()

# ---- bibliography: \bibitem[Label(year)]{key} -> author-year text ----------
bib = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", src, re.S)
items = re.findall(r"\\bibitem\[([^\]]*)\]\{([^}]+)\}(.*?)(?=\\bibitem|\Z)", bib.group(1), re.S)
label = {k: l for l, k, _ in items}


def cite(m):
    cmd, keys = m.group(1), [k.strip() for k in m.group(3).split(",")]
    ls = []
    for k in keys:
        l = re.sub(r"\((\d{4})\)", r" \1", label.get(k, k)).replace("\\&", "&").replace("~", " ")
        ls.append(l)
    if cmd in ("citep", "cite"):
        return "(" + "; ".join(ls) + ")"
    # narrative: "Author et al. 2024" -> "Author et al. (2024)"
    return ", ".join(re.sub(r" (\d{4})$", r" (\1)", l) for l in ls)


src = re.sub(r"\\(citep|citet|cite)(\[[^\]]*\])?\{([^}]+)\}", cite, src)
m = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", src, re.S)
refs = "\\section*{References}\n\\begin{itemize}\n" + "".join(
    "\\item " + re.sub(r"\\newblock", "", b).strip() + "\n" for _, _, b in items) + "\\end{itemize}\n"
src = src[:m.start()] + refs + src[m.end():]
src = re.sub(r"\\bibliographystyle\{[^}]*\}", "", src)

# ---- cross references: \ref{sec:x} -> "N", \ref{tab:x} -> "N" --------------
nums, sec, tab = {}, 0, 0
for t in re.finditer(r"\\section\{[^}]*\}\s*\\label\{([^}]+)\}|\\begin\{table\}.*?\\label\{([^}]+)\}", src, re.S):
    if t.group(1):
        sec += 1
        nums[t.group(1)] = str(sec)
    else:
        tab += 1
        nums[t.group(2)] = str(tab)
src = re.sub(r"\\ref\{([^}]+)\}", lambda x: nums.get(x.group(1), "??"), src)
src = src.replace("~", " ")  # non-breaking spaces from LaTeX
# custom math macros -> plain LaTeX that markdown renderers understand
for macro, rep in {r"\dmodel": r"d_{\mathrm{model}}", r"\nlayers": r"n_{\mathrm{layers}}",
                   r"\ZKcost": r"\mathrm{ZK\_cost}"}.items():
    src = re.sub(re.escape(macro) + r"(?![A-Za-z])", lambda _m, r=rep: r, src)
# \path{..} (breakable file names) -> verbatim; drop \resizebox wrappers pandoc cannot parse
src = re.sub(r"\\path\{([^}]*)\}", lambda m: "\\verb|" + m.group(1) + "|", src)
src = re.sub(r"\\texttt\{([^}]*)\}", lambda m: "\\verb|" + m.group(1).replace(r"\_", "_") + "|", src)
src = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}\{%?\s*", "", src)
src = re.sub(r"(\\end\{tabular\})\}", r"\1", src)
src = src.replace("\\footnotesize", "").replace("\\small", "")

# ---- title / abstract ------------------------------------------------------
title = re.sub(r"\\\\\s*", " ", re.search(r"\\title\{(.*?)\}\s*\\author", src, re.S).group(1))
title = re.sub(r"\s+", " ", title).strip()
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
abstract = re.sub(r"^\s*%.*$", "", abstract, flags=re.M).strip()


def md(tex):
    converted = pypandoc.convert_text(tex, "gfm+tex_math_dollars", format="latex",
                                      extra_args=["--wrap=none"]).strip()
    # Pandoc GFM emits GitHub's $`...`$ syntax; publish.fun expects $...$.
    converted = re.sub(r"\$`(.*?)`\$", lambda m: "$" + m.group(1) + "$",
                       converted, flags=re.S)
    # \verb is not supported by KaTeX inside math; identifiers use \text.
    converted = re.sub(r"\\verb\|([^|]*)\|",
                       lambda m: r"\text{" + m.group(1).replace("_", r"\_") + "}",
                       converted)
    return converted


abstract_md = md(abstract)
body_tex = src[src.index("\\section{Introduction}"):src.index("\\end{document}")]
body = md(body_tex)

body = re.sub(r"\\begin\{equation\}\s*(.*?)\s*\\end\{equation\}", lambda m: m.group(1), body, flags=re.S)
body = re.sub(r"\\label\{[^}]*\}", "", body)
body = re.sub(r"\\\((\d+|i|ii|iii|iv|v)\\\)", r"(\1)", body)
def _tab(m):
    return m.group(2).strip() + "\n\n**Table " + nums[m.group(1)] + ".** " + m.group(3).strip()
body = re.sub(r'<div id="(tab:[^"]+)">\s*(\|.*?\|)\s*\n\n(.*?)\s*</div>', _tab, body, flags=re.S)
# unwrap display-math fences, number headings
body = re.sub(r"``` math\n(.*?)\n```", lambda m: "$$\n" + m.group(1).strip() + "\n$$", body, flags=re.S)
s = ss = 0
out = []
for line in body.split("\n"):
    h = re.match(r"(#{1,2}) (.*)", line)
    if h and h.group(2) != "References":
        if len(h.group(1)) == 1:
            s += 1; ss = 0; line = f"# {s}. {h.group(2)}"
        else:
            ss += 1; line = f"## {s}.{ss} {h.group(2)}"
    out.append(line)
body = "\n".join(out)

assert len(abstract_md) <= 3000, f"abstract too long: {len(abstract_md)}"
assert len(body) <= 150_000, len(body)
bad = re.findall(r"\\begin\{|\\ref\{|\\cite|\?\?|\\path|\\texttt", body)
assert not bad, f"unconverted LaTeX left in body: {bad}"

(OUT / "body.md").write_text(body + "\n")
(OUT / "meta.json").write_text(json.dumps({"title": title, "abstract": abstract_md}, indent=1))
(OUT / "keywords.json").write_text(json.dumps(
    ["zero-knowledge machine learning", "EZKL", "transformers", "cost models", "circuit size", "empirical study"]))
print(f"title: {title}\nabstract chars: {len(abstract_md)}  body chars: {len(body)}  -> {OUT}")
