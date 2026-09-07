"""Render the method description to the 2-3 page PDF the challenge requires.

Markdown -> HTML -> PDF via WeasyPrint. No LaTeX on this machine, and the document
is prose with a couple of tables, so a print stylesheet is enough.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from weasyprint import CSS, HTML

HERE = Path(__file__).parent

CSS_TEXT = """
@page { size: A4; margin: 16mm 17mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 9.4pt; line-height: 1.42;
       text-align: justify; hyphens: auto; color: #111; }
h1 { font-size: 15pt; line-height: 1.25; margin: 0 0 2mm; text-align: left; }
h2 { font-size: 10.6pt; margin: 4.2mm 0 1.4mm; text-align: left;
     border-bottom: 0.4pt solid #bbb; padding-bottom: 0.6mm; }
p { margin: 0 0 1.9mm; }
strong { font-weight: 700; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.3pt;
       background: #f4f4f4; padding: 0 0.6mm; }
/* byline: the single line under the title */
h1 + p { font-size: 8.8pt; color: #444; margin-bottom: 3mm; }
a { color: #111; text-decoration: none; }
"""


def main() -> None:
    src = HERE / (sys.argv[1] if len(sys.argv) > 1 else "method.md")
    html = markdown.markdown(src.read_text(), extensions=["tables", "smarty"])
    out = src.with_suffix(".pdf")
    HTML(string=html, base_url=str(HERE)).write_pdf(out, stylesheets=[CSS(string=CSS_TEXT)])

    from pypdf import PdfReader

    pages = len(PdfReader(str(out)).pages)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {pages} page(s))")
    if not 2 <= pages <= 3:
        print(f"WARNING: the challenge asks for 2-3 pages, this is {pages}")


if __name__ == "__main__":
    main()
