"""Every page in web/ is the build of its template with the results its notebook wrote (python _build/build_web.py
<topic>); a page left behind after the results or the template changed fails here."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p.stem for p in (ROOT / "web").glob("*.html"))


def build_web():
    spec = importlib.util.spec_from_file_location("build_web", ROOT / "_build" / "build_web.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("topic", PAGES)
def test_page_is_the_build_of_its_template_and_results(topic, tmp_path):
    out, _ = build_web().build(topic, out_path=tmp_path / f"{topic}.html")
    # read_text translates line endings, so a CRLF checkout compares equal
    committed = (ROOT / "web" / f"{topic}.html").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == committed, \
        f"web/{topic}.html is out of date: python _build/build_web.py {topic}"
