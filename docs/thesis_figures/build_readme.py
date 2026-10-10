#!/usr/bin/env python3
# rebuild thesis_figures_readme.md: insert each figure's image and its verbatim code
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
T = PUB / "thesis_ch3"
MD = HERE / "THESIS_FIGURES_README.md"
CODE_DIR = HERE / "code"

REGROUP = {
    "3.3": "fig3_03_splice_environment",
    "3.4": "fig3_04_donor_scores",
    "3.6": "fig3_05_splice_model",
    "3.7": "fig3_06_splice_model_performance",
    "3.8": "fig3_07_architecture",
    "3.9": "fig3_08_cnn_vs_splice",
    "3.11": "fig3_12_interpretation",
}

FIGS = {
    "2.1": ("fig2_1.png", ["sfig1.R"]),
    "2.2": (["fig2_2a.png", "fig2_2b.png"], []),
    "2.3": ("fig2_3.png", ["scripts/make_figure1.R"]),
    "2.4": ("fig2_4.png", ["make_analysis_set_funnel.R"]),
    "2.5": ("fig2_5.png", ["scripts/make_figure2.R"]),
    "2.6": (["fig2_6a.png", "fig2_6b.png"], []),
    "2.7": (["fig2_7_top.png", "fig2_7_bottom.png"], []),
    "2.8": (["fig2_8a.png", "fig2_8b.png"], []),
    "2.9": ("fig2_9.png", ["make_figure_overlap_shared.R"]),
    "3.1": ("fig3_1.png", ["thesis_ch3/R/fig3_01_hypothesis.R"]),
    "3.2": ("fig3_2.png", ["thesis_ch3/R/fig3_02_genesets.R"]),
    "3.3": ("fig3_3.png", [("regroup", "3.3")]),
    "3.4": ("fig3_4.png", [("regroup", "3.4")]),
    "3.5": ("fig3_5.png", ["thesis_ch3/R/fig3_hypothesis_dose.R"]),
    "3.6": ("fig3_6.png", [("regroup", "3.6")]),
    "3.7": ("fig3_7.png", [("regroup", "3.7")]),
    "3.8": ("fig3_8.png", [("regroup", "3.8")]),
    "3.9": ("fig3_9.png", [("regroup", "3.9")]),
    "3.10": ("fig3_10.png", ["thesis_ch3/R/fig3_09_robustness.R"]),
    "3.11": ("fig3_11.png", [("regroup", "3.11")]),
    "3.12": ("fig3_12.png", ["thesis_ch3/R/fig3_13_mechanism.R"]),
    "3.13": ("fig3_13.png", ["thesis_ch3/R/fig3_implant_panel.R"]),
    "3.14": ("fig3_14.png", ["thesis_ch3/R/fig3_14_recovery.R"]),
}

APPENDIX = [
    ("thesis_ch3/R/fig3_regroup.R", "Assembles Figures 3.3, 3.4, 3.6, 3.7, 3.8, 3.9 and 3.11 from harvested panels."),
    ("thesis_ch3/R/_harvest.R", "Runs a pub/ script for its panel objects without letting it write; also `retitle` and `edit_text`."),
    ("make_splicing_environment.R", "Panels harvested for Figures 3.3 (a, b) and 3.4 (a, b)."),
    ("make_splice_model_final.R", "Panels harvested for Figures 3.6 and 3.7."),
    ("make_figure_architecture.R", "Panel b of Figure 3.8."),
    ("make_figure_rescue.R", "Panel a of Figure 3.8."),
    ("make_figure_loco.R", "Panels harvested for Figure 3.9."),
    ("make_figure_cnn_interpretability.R", "Panels harvested for Figure 3.11."),
    ("nature_figs/make_nature_figs.R", "Re-renders all 19 coded figures in Nature style; the images in this document."),
]

LANG = {".R": "r", ".py": "python", ".sh": "bash"}


def sha12(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def read(rel: str) -> str:
    p = PUB / rel
    if not p.exists():
        sys.exit(f"ERROR: missing code file {p}")
    return p.read_text()


def regroup_block(num: str):
    rel = "thesis_ch3/R/fig3_regroup.R"
    lines = read(rel).splitlines()
    banners = [i for i, l in enumerate(lines) if l.startswith("# ------")]
    want = REGROUP[num]
    for bi, start in enumerate(banners):
        end = banners[bi + 1] if bi + 1 < len(banners) else len(lines)
        block = lines[start:end]
        saved = [s for s in re.findall(r'"([^"]+)"', "\n".join(block))
                 if re.fullmatch(r"fig3_[0-9a-z_]+", s)]
        if want in saved:
            also = [n for n in saved if n != want]
            return "\n".join(block).rstrip(), start + 1, end, also
    sys.exit(f"ERROR: no fig3_regroup.R block saves {want}")


def code_section(specs, key):
    if not specs:
        return ("*No code: this figure is a drawing, a photograph of a gel, or a sequencing "
                "trace panel. See the figure's entry above for what is known about its origin.*")
    out = []
    for spec in specs:
        if isinstance(spec, tuple):
            num = spec[1]
            text, lo, hi, also = regroup_block(num)
            rel = "thesis_ch3/R/fig3_regroup.R"
            whole = read(rel)
            note = (f"`{rel}`, lines {lo} to {hi} (the block that saves "
                    f"`{REGROUP[num]}`), plus the shared preamble. ")
            if also:
                note += ("This block also builds " +
                         ", ".join("`%s`" % a for a in also) + ". ")
            note += (f"The whole file ({len(whole.splitlines())} lines, sha256 {sha12(whole.encode())}) "
                     "is in the appendix and in `code/`.")
            pre = "\n".join(whole.splitlines()[:23]).rstrip()
            out.append(f"**Code.** {note}\n\nShared preamble:\n\n```r\n{pre}\n```\n\n"
                       f"Block for this figure:\n\n```r\n{text}\n```")
        else:
            txt = read(spec)
            n = len(txt.splitlines())
            lang = LANG.get(Path(spec).suffix, "")
            out.append(f"**Code.** `{spec}`, {n} lines, sha256 `{sha12(txt.encode())}`. "
                       f"Reproduced verbatim; also in `code/{Path(spec).name}`.\n\n"
                       f"```{lang}\n{txt.rstrip()}\n```")
    return "\n\n".join(out)


def image_section(img, key):
    imgs = img if isinstance(img, list) else [img]
    for i in imgs:
        if not (HERE / "figures" / i).exists():
            sys.exit(f"ERROR: missing image figures/{i}")
    return "\n\n".join(f"![Figure {key}](figures/{i})" for i in imgs)


def mask(md: str) -> str:
    out = list(md)
    spans = [m.span() for m in re.finditer(
        r"<!-- BEGIN AUTO:.*?-->.*?<!-- END AUTO:.*?-->|^```.*?^```", md, re.S | re.M)]
    for a, b in spans:
        for i in range(a, b):
            if out[i] != "\n":
                out[i] = "x"
    return "".join(out)


def splice(md: str) -> str:
    for key, (img, specs) in FIGS.items():
        mk = mask(md)
        head = re.search(rf"^### (?:Figure|Figures) {re.escape(key)}[:,]", mk, re.M)
        if not head:
            sys.exit(f"ERROR: no '### Figure {key}:' heading in the markdown")
        nxt = re.search(r"^#{1,3} ", mk[head.end():], re.M)
        sec_end = head.end() + (nxt.start() if nxt else len(md) - head.end())
        eol = md.index("\n", head.start()) + 1
        body = md[eol:sec_end]
        for kind, payload in (("image", image_section(img, key)),
                              ("code", code_section(specs, key))):
            blk = (f"<!-- BEGIN AUTO:{key}:{kind} -->\n{payload}\n"
                   f"<!-- END AUTO:{key}:{kind} -->")
            pat = re.compile(rf"<!-- BEGIN AUTO:{re.escape(key)}:{kind} -->.*?"
                             rf"<!-- END AUTO:{re.escape(key)}:{kind} -->", re.S)
            if pat.search(body):
                body = pat.sub(lambda _m: blk, body)
            elif kind == "image":
                body = "\n" + blk + "\n" + body.lstrip("\n")
            else:
                body = body.rstrip("\n") + "\n\n" + blk + "\n"
        md = md[:eol] + body + md[sec_end:]
    return md


def appendix(md: str) -> str:
    rows = []
    for rel, why in APPENDIX:
        txt = read(rel)
        lang = LANG.get(Path(rel).suffix, "")
        rows.append(f"### `{rel}`\n\n{why} {len(txt.splitlines())} lines, "
                    f"sha256 `{sha12(txt.encode())}`.\n\n```{lang}\n{txt.rstrip()}\n```")
    payload = ("Reproduced once each, because several figures draw panels from them. "
               "Copies are in `code/`.\n\n" + "\n\n".join(rows))
    blk = f"<!-- BEGIN AUTO:appendix:code -->\n{payload}\n<!-- END AUTO:appendix:code -->"
    pat = re.compile(r"<!-- BEGIN AUTO:appendix:code -->.*?<!-- END AUTO:appendix:code -->", re.S)
    if not pat.search(md):
        sys.exit("ERROR: appendix marker block not found; add the 'Appendix' heading first")
    return pat.sub(lambda _m: blk, md)


def manifest():
    CODE_DIR.mkdir(exist_ok=True)
    rows = ["file\tlines\tsha256_12\tfigures"]
    used = {}
    for key, (_img, specs) in FIGS.items():
        for spec in specs:
            rel = "thesis_ch3/R/fig3_regroup.R" if isinstance(spec, tuple) else spec
            used.setdefault(rel, []).append(key)
    for rel, _why in APPENDIX:
        used.setdefault(rel, [])
    for rel in sorted(used):
        txt = read(rel)
        (CODE_DIR / Path(rel).name).write_text(txt)
        figs = ", ".join(used[rel]) or "appendix"
        rows.append(f"{rel}\t{len(txt.splitlines())}\t{sha12(txt.encode())}\t{figs}")
    (HERE / "code_manifest.tsv").write_text("\n".join(rows) + "\n")
    return len(used)


def normalise(md: str) -> str:
    md = re.sub(r"\n*(<!-- BEGIN AUTO:)", r"\n\n\1", md)
    md = re.sub(r"(<!-- END AUTO:[^>]*-->)\n*", r"\1\n\n", md)
    return md.rstrip("\n") + "\n"


def main():
    check = "--check" in sys.argv
    old = MD.read_text()
    new = normalise(appendix(splice(old)))
    if check:
        print("UP TO DATE" if new == old else "STALE: re-run build_readme.py")
        sys.exit(0 if new == old else 1)
    MD.write_text(new)
    n = manifest()
    print(f"wrote {MD.name}: {len(new)//1024} KB, {len(FIGS)} figures, "
          f"{n} code files copied to code/")


if __name__ == "__main__":
    main()
