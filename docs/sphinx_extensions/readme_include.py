import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx

README_PATH = Path(__file__).resolve().parents[2] / "README.md"


# ----------------------------
# Regex
# ----------------------------

UL_PATTERN = re.compile(r"<ul[^>]*>(.*?)</ul>", re.DOTALL | re.IGNORECASE)
LI_PATTERN = re.compile(r"<li[^>]*>(.*?)</li>", re.DOTALL | re.IGNORECASE)

LINK_PATTERN = re.compile(
    r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)

BR_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"</?\w+[^>]*>")


# ----------------------------
# IO
# ----------------------------


def _read_readme() -> str:
    if not README_PATH.exists():
        return ""
    return README_PATH.read_text(encoding="utf-8")


def _extract_section(text: str, header: str) -> str:
    m = re.search(
        rf"<h2>\s*{re.escape(header)}\s*</h2>(.*?)(?=<h2>|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


# ----------------------------
# SINGLE SAFE CONVERTER
# ----------------------------


def _html_to_rst(text: str) -> str:
    """
    Convert HTML fragment → RST safely (ONE PASS ONLY).
    """

    text = BR_PATTERN.sub("\n", text)

    # Protect converted links so HTML cleanup cannot strip their <URL> targets.
    links = []

    def _link_replace(m):
        url = m.group(1)
        label = TAG_PATTERN.sub("", m.group(2)).strip()
        links.append(f"`{label} <{url}>`_")
        return f"\x00READMELINK{len(links) - 1}\x00"

    text = LINK_PATTERN.sub(_link_replace, text)

    # bold
    text = re.sub(
        r"<(?:b|strong)>(.*?)</(?:b|strong)>",
        r"**\1**",
        text,
        flags=re.I | re.S,
    )

    # italic
    text = re.sub(
        r"<(?:i|em)>(.*?)</(?:i|em)>",
        r"*\1*",
        text,
        flags=re.I | re.S,
    )

    # remove remaining HTML
    text = TAG_PATTERN.sub("", text)

    # Restore links only after all HTML tags have been removed.
    for index, link in enumerate(links):
        text = text.replace(f"\x00READMELINK{index}\x00", link)

    # normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------
# EXTRACTION (NO DOUBLE PROCESSING)
# ----------------------------


def _extract_news(text: str) -> List[str]:
    section = _extract_section(text, "News & Announcements")
    if not section:
        return []

    items = []

    ul = UL_PATTERN.search(section)
    if ul:
        for li in LI_PATTERN.findall(ul.group(1)):
            items.append(_html_to_rst(li))

    return items


def _build_rst(items: List[str]) -> List[str]:
    return [f"- {x}" for x in items]


# ----------------------------
# SPHINX
# ----------------------------


class ReadmeNews(Directive):
    has_content = False

    def run(self):
        readme = _read_readme()
        news = _extract_news(readme)
        rst_lines = _build_rst(news)

        container = nodes.container()

        if rst_lines:
            self.state.nested_parse(StringList(rst_lines), 0, container)

        return container.children


def setup(app: Sphinx):
    app.add_directive("readme-news", ReadmeNews)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
