"""Build a self-contained interactive page from a template and the results a notebook wrote.

    python _build/build_web.py data_splits                      # -> web/data_splits.html
    python _build/build_web.py data_splits --data other.json --out test.html

The page is one HTML file: the shared house style (_build/web/house.css), the chart engine
(_build/web/chart.js) and the results JSON of the experiment (experiments/<topic>/<topic>_results.json,
written by its notebook) are inlined into the template _build/web/<topic>.html, so it also works offline.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "_build" / "web"


def build(topic, data_path=None, out_path=None):
    template = (WEB / f"{topic}.html").read_text(encoding="utf-8")
    data_path = Path(data_path) if data_path else ROOT / "experiments" / topic / f"{topic}_results.json"
    out_path = Path(out_path) if out_path else ROOT / "web" / f"{topic}.html"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    # "</" would end the <script> element early
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    parts = {
        "/*@HOUSE_CSS@*/": (WEB / "house.css").read_text(encoding="utf-8"),
        "/*@CHART_JS@*/": (WEB / "chart.js").read_text(encoding="utf-8"),
        "@DATA_JSON@": payload,
    }
    for key, value in parts.items():
        if template.count(key) != 1:
            raise ValueError(f"{key} must occur exactly once in the template")
        template = template.replace(key, value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template, encoding="utf-8", newline="\n")
    return out_path, len(template.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topic")
    parser.add_argument("--data", help="results JSON (default: experiments/<topic>/<topic>_results.json)")
    parser.add_argument("--out", help="output page (default: web/<topic>.html)")
    args = parser.parse_args()
    path, size = build(args.topic, args.data, args.out)
    print(f"wrote {path} ({size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
