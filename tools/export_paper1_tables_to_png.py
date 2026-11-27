# -*- coding: utf-8 -*-
"""
Parse all tables directly from manuscript/paper1_discover_ai/overleaf_ready/main.tex
and render each one as a crisp 300-DPI publication-grade PNG (Table_1.png .. Table_7.png)
using headless Chrome + PIL auto-crop.
"""
import os
import re
import html
import tempfile
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = REPO_ROOT / "manuscript" / "paper1_discover_ai" / "overleaf_ready" / "main.tex"
OUT_DIR = REPO_ROOT / "manuscript" / "paper1_discover_ai" / "overleaf_ready" / "tables"
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Chrome or Edge executable not found.")


def build_cite_map(tex: str) -> dict[str, int]:
    bibs = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    return {key: idx for idx, key in enumerate(bibs, 1)}


def split_unescaped(text: str, sep: str) -> list[str]:
    parts, cur = [], []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == sep:
            cur.append(text[i : i + 2])
            i += 2
        elif text[i] == sep:
            parts.append("".join(cur))
            cur = []
            i += 1
        else:
            cur.append(text[i])
            i += 1
    parts.append("".join(cur))
    return parts


def extract_braced_arg(s: str, start_brace: int) -> tuple[str, int]:
    assert s[start_brace] == "{", f"Expected '{{' at {start_brace}, got {s[start_brace:start_brace+20]!r}"
    depth = 0
    for i in range(start_brace, len(s)):
        if s[i] == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif s[i] == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return s[start_brace + 1 : i], i + 1
    raise ValueError("Unbalanced braces in LaTeX.")


def replace_cmd(s: str, cmd: str, fn) -> str:
    token = "\\" + cmd + "{"
    out = []
    i = 0
    while i < len(s):
        idx = s.find(token, i)
        if idx == -1:
            out.append(s[i:])
            break
        out.append(s[i:idx])
        inner, nxt = extract_braced_arg(s, idx + len(token) - 1)
        out.append(fn(inner))
        i = nxt
    return "".join(out)


def latex_to_html(text: str, cite_map: dict[str, int]) -> str:
    s = text.strip()
    tokens: list[str] = []

    def tok(html_frag: str) -> str:
        # Token uses ONLY digits and control sentinels (no underscores or letters!)
        key = f"\x01{len(tokens)}\x02"
        tokens.append(html_frag)
        return key

    # Citations -> superscript [n]
    def fmt_cite(keys_str: str) -> str:
        nums = [str(cite_map.get(k.strip(), "?")) for k in keys_str.split(",")]
        return tok(f"<sup>[{','.join(nums)}]</sup>")

    s = replace_cmd(s, "cite", fmt_cite)

    # Convert math expressions ($...$) cleanly without any underscore-containing sentinels
    def conv_math_expr(expr: str) -> str:
        e = expr
        # Protect escaped underscores first
        e = e.replace(r"\_", "\x03UND\x04")
        e = e.replace(r"\times", "×")
        e = e.replace(r"\to", "→")
        e = e.replace(r"\pm", "±")
        e = e.replace(r"\varepsilon", "ε")
        e = e.replace(r"\sigma", "σ")
        e = e.replace(r"\|", "‖")
        e = e.replace(r"\le", "≤")
        e = e.replace(r"\ge", "≥")
        e = e.replace(r"\approx", "≈")
        e = e.replace(r"\in", "∈")

        # Handle subscripts _{...} and _x BEFORE converting \text / \mathbf so nested \text in _{...} works
        # First convert \mathbf{...} and \text{...} using underscore-free sentinels
        def conv_nested(x: str) -> str:
            x = replace_cmd(x, "mathbf", lambda v: f"\x05B\x06{v}\x05/B\x06")
            x = replace_cmd(x, "text", lambda v: f"\x05RM\x06{v}\x05/RM\x06")
            return x

        e = conv_nested(e)
        # Brace-aware subscripts _{...}
        while "_{" in e:
            idx = e.find("_{")
            inner, nxt = extract_braced_arg(e, idx + 1)
            inner = conv_nested(inner)
            e = e[:idx] + f"\x05SUB\x06{inner}\x05/SUB\x06" + e[nxt:]
        e = re.sub(r"_([A-Za-z0-9])", "\x05SUB\x06\\1\x05/SUB\x06", e)

        # Brace-aware superscripts ^{...}
        while "^{" in e:
            idx = e.find("^{")
            inner, nxt = extract_braced_arg(e, idx + 1)
            inner = conv_nested(inner)
            e = e[:idx] + f"\x05SUP\x06{inner}\x05/SUP\x06" + e[nxt:]
        e = re.sub(r"\^([A-Za-z0-9])", "\x05SUP\x06\\1\x05/SUP\x06", e)

        # Italicize isolated single-letter math variables outside tags
        e = re.sub(r"(?<![A-Za-z\x05/])([QnLND])(?![A-Za-z\x06])", "\x05I\x06\\1\x05/I\x06", e)
        e = e.replace("\x03UND\x04", "_")
        return e

    s = re.sub(r"\$([^$]+)\$", lambda m: conv_math_expr(m.group(1)), s)

    # Convert text-mode formatting commands iteratively
    for _ in range(4):
        s = replace_cmd(s, "textbf", lambda x: f"\x05B\x06{x}\x05/B\x06")
        s = replace_cmd(s, "textit", lambda x: f"\x05I\x06{x}\x05/I\x06")
        s = replace_cmd(s, "emph", lambda x: f"\x05I\x06{x}\x05/I\x06")
        s = replace_cmd(s, "texttt", lambda x: f"\x05CODE\x06{x}\x05/CODE\x06")

    # Text-mode LaTeX escapes
    s = s.replace(r"\ref{tab:stats_appendix}", "8")
    s = s.replace(r"\{", "{")
    s = s.replace(r"\}", "}")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\_", "_")
    s = s.replace(r"\&", "&")
    s = s.replace(r"\ ", " ")
    s = s.replace("---", "—")
    s = s.replace("--", "–")
    s = s.replace("~", " ")

    # Escape HTML special chars
    s = html.escape(s)

    # Restore formatting tags
    tag_map = {
        "\x05B\x06": "<strong>",
        "\x05/B\x06": "</strong>",
        "\x05I\x06": "<em>",
        "\x05/I\x06": "</em>",
        "\x05CODE\x06": '<span class="code">',
        "\x05/CODE\x06": "</span>",
        "\x05SUB\x06": "<sub>",
        "\x05/SUB\x06": "</sub>",
        "\x05SUP\x06": "<sup>",
        "\x05/SUP\x06": "</sup>",
        "\x05RM\x06": '<span class="rm">',
        "\x05/RM\x06": "</span>",
    }
    for k, v in tag_map.items():
        s = s.replace(k, v)
    for idx, frag in enumerate(tokens):
        s = s.replace(f"\x01{idx}\x02", frag)
    return s


def parse_multicolumn(row_str: str):
    if not row_str.startswith(r"\multicolumn{"):
        return None
    pos = len(r"\multicolumn")
    span_s, pos = extract_braced_arg(row_str, pos)
    _align_s, pos = extract_braced_arg(row_str, pos)
    content_s, pos = extract_braced_arg(row_str, pos)
    return int(span_s), content_s


def parse_tabular(tab_block: str, cite_map: dict[str, int]):
    cap_idx = tab_block.find(r"\caption{")
    caption_tex, _ = extract_braced_arg(tab_block, cap_idx + len(r"\caption"))
    caption_html = latex_to_html(caption_tex, cite_map)

    tab_start = tab_block.find(r"\begin{tabular}")
    assert tab_start != -1, "tabular start not found"
    col_spec_raw, body_start = extract_braced_arg(tab_block, tab_start + len(r"\begin{tabular}"))
    tab_end = tab_block.find(r"\end{tabular}", body_start)
    body = tab_block[body_start:tab_end]

    # Strip @{...} cleanly from column spec
    col_spec = re.sub(r"@\{[^}]*\}", "", col_spec_raw)
    col_aligns = [{"l": "left", "c": "center", "r": "right"}[ch] for ch in col_spec if ch in "lcr"]

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    header_rows = []
    data_rows = []
    seen_first_mid = False
    pending_section_sep = False

    for ln in lines:
        if ln.startswith(r"\toprule") or ln.startswith(r"\bottomrule"):
            continue
        if ln.startswith(r"\midrule"):
            if not seen_first_mid:
                seen_first_mid = True
            else:
                pending_section_sep = True
            continue
        row_str = re.sub(r"\\\\\s*$", "", ln).strip()
        if not row_str:
            continue

        mc = parse_multicolumn(row_str)
        if mc is not None:
            span, content_tex = mc
            data_rows.append(("subhead", span, latex_to_html(content_tex, cite_map), pending_section_sep))
            pending_section_sep = False
            continue

        cells = [latex_to_html(c, cite_map) for c in split_unescaped(row_str, "&")]
        if not seen_first_mid:
            header_rows.append(cells)
        else:
            data_rows.append(("row", cells, pending_section_sep))
            pending_section_sep = False

    return caption_html, col_aligns, header_rows, data_rows


def build_html(num: int, caption_html: str, col_aligns: list[str], header_rows, data_rows) -> str:
    ncols = len(col_aligns)
    thead_html = []
    for row in header_rows:
        tds = []
        for idx, cell in enumerate(row):
            al = col_aligns[idx] if idx < ncols else "left"
            tds.append(f'<th style="text-align:{al};">{cell}</th>')
        thead_html.append("<tr>" + "".join(tds) + "</tr>")

    tbody_html = []
    for item in data_rows:
        if item[0] == "subhead":
            _, span, content, sep = item
            cls = "subhead sep-top" if sep else "subhead"
            tbody_html.append(f'<tr class="{cls}"><td colspan="{span}">{content}</td></tr>')
        else:
            _, cells, sep = item
            cls_list = []
            if sep:
                cls_list.append("sep-top")
            first_plain = re.sub(r"<[^>]+>", "", cells[0]).strip()
            if first_plain.startswith(("Subtotal", "Total ", "Total Trainable")):
                cls_list.append("total-row")
            cls_attr = f' class="{" ".join(cls_list)}"' if cls_list else ""
            tds = []
            for idx, cell in enumerate(cells):
                al = col_aligns[idx] if idx < ncols else "left"
                # Keep short metric cells on one line in Tables 3, 5, 6, 7
                nowrap = ' class="nw"' if (num in (3, 5, 6, 7, 8) and idx > 0) else ""
                tds.append(f'<td{nowrap} style="text-align:{al};">{cell}</td>')
            tbody_html.append(f"<tr{cls_attr}>" + "".join(tds) + "</tr>")

    # Comfortable width per table so nothing wraps awkwardly
    width_map = {1: 1120, 2: 1180, 3: 1320, 4: 1260, 5: 1340, 6: 1260, 7: 1420, 8: 1340}
    target_w = width_map.get(num, 1240)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    padding: 0;
    background: #ffffff;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrapper {{
    display: inline-block;
    background: #ffffff;
    padding: 28px 32px;
    width: {target_w}px;
  }}
  .caption {{
    font-size: 14px;
    line-height: 1.48;
    color: #111827;
    margin-bottom: 14px;
    text-align: left;
  }}
  .caption .tab-label {{
    font-weight: 700;
    color: #0f172a;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12.6px;
    color: #1f2937;
    font-variant-numeric: tabular-nums;
  }}
  thead tr th {{
    border-top: 2.2px solid #111827;
    border-bottom: 1.5px solid #111827;
    background: #f8fafc;
    color: #0f172a;
    font-weight: 700;
    padding: 9px 10px;
    line-height: 1.32;
    vertical-align: bottom;
    white-space: nowrap;
  }}
  tbody td {{
    padding: 6.5px 10px;
    border-bottom: 1px solid #e5e7eb;
    line-height: 1.38;
    vertical-align: middle;
  }}
  tbody td.nw {{
    white-space: nowrap;
  }}
  tbody tr:last-child td {{
    border-bottom: 2.2px solid #111827;
  }}
  tbody tr.sep-top td {{
    border-top: 1.4px solid #64748b;
  }}
  tbody tr.subhead td {{
    background: #f1f5f9;
    color: #1e293b;
    font-weight: 600;
    padding: 7px 10px;
    border-bottom: 1px solid #cbd5e1;
  }}
  tbody tr.total-row td {{
    background: #f8fafc;
  }}
  .code {{
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11.9px;
    color: #0f172a;
  }}
  .rm {{ font-style: normal; }}
  sup {{ font-size: 0.72em; line-height: 0; }}
  sub {{ font-size: 0.74em; line-height: 0; }}
</style>
</head>
<body>
  <div class="wrapper" id="capture">
    <div class="caption"><span class="tab-label">Table {num}.</span> {caption_html}</div>
    <table>
      <thead>{"".join(thead_html)}</thead>
      <tbody>{"".join(tbody_html)}</tbody>
    </table>
  </div>
</body>
</html>"""


def render_html_to_png(browser: str, html_str: str, out_png: Path, win_w: int = 1600, win_h: int = 2400):
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        html_path = td_p / "table.html"
        raw_png = td_p / "raw.png"
        html_path.write_text(html_str, encoding="utf-8")

        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size={win_w},{win_h}",
            "--force-device-scale-factor=3",
            "--default-background-color=FFFFFFFF",
            f"--screenshot={raw_png}",
            html_path.as_uri(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        im = Image.open(raw_png).convert("RGB")
        arr = np.array(im)
        nonwhite = (arr < 252).any(axis=2)
        ys = np.where(nonwhite.any(axis=1))[0]
        xs = np.where(nonwhite.any(axis=0))[0]
        pad = 60
        y0 = max(0, int(ys[0]) - pad)
        y1 = min(arr.shape[0], int(ys[-1]) + pad)
        x0 = max(0, int(xs[0]) - pad)
        x1 = min(arr.shape[1], int(xs[-1]) + pad)
        cropped = Image.fromarray(arr[y0:y1, x0:x1, :])
        cropped.save(out_png, dpi=(300, 300))
        return cropped.size


def main():
    browser = find_browser()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex = MAIN_TEX.read_text(encoding="utf-8")
    cite_map = build_cite_map(tex)
    tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, flags=re.S)
    print(f"Found {len(tables)} tables in {MAIN_TEX.name}")

    for idx, tab_block in enumerate(tables, 1):
        cap_html, aligns, head_rows, data_rows = parse_tabular(tab_block, cite_map)
        doc_html = build_html(idx, cap_html, aligns, head_rows, data_rows)
        # Sanity check: ensure zero unparsed LaTeX commands or sentinels remain
        bad = re.findall(r"\\multicolumn|@@|\x01|\x05", doc_html)
        assert not bad, f"Table {idx} has unparsed tokens: {bad}"
        out_png = OUT_DIR / f"Table_{idx}.png"
        sz = render_html_to_png(browser, doc_html, out_png)
        print(f"Rendered {out_png.name}: {sz[0]}x{sz[1]} px (300 DPI)")


if __name__ == "__main__":
    main()
