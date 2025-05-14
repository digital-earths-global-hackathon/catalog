from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import mistune
from markupsafe import Markup
from textwrap import dedent
from pathlib import Path
import shutil
import argparse
import json
from urllib.parse import quote as urlquote

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

def generate_highlight_css(style="default"):
    # Generate the CSS using Pygments
    css = HtmlFormatter(style=style).get_style_defs('.source')

    # Return the CSS as a string
    return Markup(css)

def highlight_code(code, lexer_name="python"):
    lexer = get_lexer_by_name(lexer_name)
    formatter = HtmlFormatter(cssclass="source")
    return Markup(highlight(dedent(code), lexer, formatter))

def format_default_params(mlds):
    if descr := mlds.get("parameter_descriptions", {}):
        return "(" + ", ".join(f"{name}={d['default']!r}" for name, d in descr.items()) + ")"
    else:
        return ""

def format_raw_list(l):
    return ", ".join(f"{e!r}" for e in l)

def max_allowed_params(mlds):
    return {
        key: list(sorted(set(a for r in mlds["raw"].values() for a in r["allowed_parameters"][key])))
        for key in mlds.get("parameter_descriptions", {})
    }

def gridlook_url(mlds):
    if not "online" in mlds["raw"]:
        return None
    raw = mlds["raw"]["online"]
    if raw["driver"] != "zarr":
        return None
    if mlds.get("metadata", {}).get("region", "global") != "global":
        return None
    url = raw["args"]["urlpath"]
    if isinstance(url, list):
        url = url[0]
    if not isinstance(url, str):
        return None

    defaults = {k: v["default"] for k, v in mlds["parameter_descriptions"].items()}
    if "zoom" in defaults:
        defaults["zoom"] = list(sorted(raw["allowed_parameters"]["zoom"], key=lambda z: abs(int(z) - 7)))[0]

    return Template(url).render(**defaults)

def render_markdown(markdown):
    return Markup(mistune.html(markdown))

def parse_url(u: str):
    u = u.strip()
    ul = u.lower()
    if ul.startswith("http://") or ul.startswith("https://"):
        url = u
    elif ul.startswith("doi:"):
        url = "https://doi.org/" + urlquote(u[4:])
    else:
        url = None
    return {"url": url, "text": u}

def split_url_list(l):
    if isinstance(l, str):
        l = l.split(",")
    return [parse_url(u) for u in l]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mlds", type=Path, metavar="mlds.json"
    )
    parser.add_argument("outdir", type=Path, help="output folder")
    args = parser.parse_args()

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(),
    )
    env.filters["highlight_css"] = generate_highlight_css
    env.filters["highlight"] = highlight_code
    env.filters["default_params"] = format_default_params
    env.filters["raw_list"] = format_raw_list
    env.filters["markdown"] = render_markdown
    env.filters["gridlook_url"] = gridlook_url

    mldss = json.load(open(args.mlds))

    template = env.get_template("index.html")

    with open(args.outdir / "index.html", "w") as outfile:
        outfile.write(template.render(mldss=mldss, sorted=sorted, max_allowed_params=max_allowed_params, split_url_list=split_url_list, gridlook_url=gridlook_url))

    shutil.copytree(Path(__file__).parent / "static", args.outdir, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
