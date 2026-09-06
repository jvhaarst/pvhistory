"""The site must publish what the analysis produced, and say what is stale."""

import re

import pytest

import build_site


def test_every_report_is_described_and_linked():
    """A card with no link, or a link with no card, is a page nobody reads."""
    html = build_site.build_index(build_site.REPORTS, "2026-01-01")
    linked = set(re.findall(r'href="([^"]+\.html)"', html))
    assert linked == {r["file"] for r in build_site.REPORTS}
    for r in build_site.REPORTS:
        assert r["title"] in html
        assert len(r["blurb"]) > 80, f"{r['file']} needs a real description"


def test_superseded_reports_are_marked_as_such():
    """Phase 2's figures were invalidated by a hardware fault found later.

    A reader arriving from a search engine has no other way to know that, so
    the badge is not decoration.
    """
    html = build_site.build_index(build_site.REPORTS, "2026-01-01")
    assert 'class="tag superseded"' in html
    stale = [r for r in build_site.REPORTS if r["status"] == "superseded"]
    assert stale, "at least phase 2 is superseded"
    assert html.count('class="tag superseded"') == len(stale)


def test_the_index_marks_the_right_report_stale():
    html = build_site.build_index(build_site.REPORTS, "2026-01-01")
    # The badge must sit in phase 2's card, not merely somewhere on the page.
    card = html.split('href="night_report.html"')[1].split("</a>")[0]
    assert "superseded" in card
    other = html.split('href="meter_report.html"')[1].split("</a>")[0]
    assert "superseded" not in other


def test_building_without_the_reports_fails_loudly(tmp_path):
    """Publishing a stale or partial site silently is the failure to avoid."""
    (tmp_path / "out").mkdir()
    with pytest.raises(FileNotFoundError, match="missing reports"):
        build_site.run(tmp_path)
