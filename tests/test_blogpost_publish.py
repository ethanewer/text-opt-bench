from pathlib import Path

from tools.make_blogpost import build, build_official


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = (
    "llm_routing",
    "optimizer_generalization",
    "slm_compression_3_5bpw",
    "slm_compression_4_5bpw",
)
LEGACY_MARKERS = (
    '<span class="pname">mem_index</span>',
    '<span class="pname">mem_infer</span>',
    '<span class="pname">tag_seq</span>',
    '<span class="pname">compress_heldout</span>',
    'id="experiment-1a"',
    'id="experiment-1b"',
    'id="experiment-2"',
    'id="experiment-3"',
)


def test_official_blog_has_only_official_task_results():
    html = build_official()
    assert all(f'<span class="pname">{task}</span>' in html
               for task in OFFICIAL)
    assert not any(marker in html for marker in LEGACY_MARKERS)
    assert "blogpost-all.html" in html
    assert "<h2>Official task results</h2>" in html
    assert "The four official task protocols" not in html
    assert "Fixed-method SLM reference studies" not in html


def test_complete_blog_labels_and_contains_both_scopes():
    html = build(include_legacy=True)
    assert all(f'<span class="pname">{task}</span>' in html
               for task in OFFICIAL)
    assert all(marker in html for marker in LEGACY_MARKERS)
    assert "official · generalization" in html
    assert "legacy ·" in html
    assert "Fixed-method SLM reference studies" not in html


def test_slm_overfitting_audit_is_on_task_back_only():
    html = build_official()
    for task in OFFICIAL[-2:]:
        panel_start = html.index(f'<span class="pname">{task}</span>')
        panel_end = html.index("</figure>", panel_start)
        panel_html = html[panel_start:panel_end]
        back_start = panel_html.index('<div class="face back">')
        audit_start = panel_html.index("All-submission overfitting audit:")
        assert audit_start > back_start


def test_generated_files_match_generator():
    assert (ROOT / "docs/blogpost.html").read_text() == build_official()
    assert (ROOT / "docs/blogpost-all.html").read_text() == build(
        include_legacy=True)


if __name__ == "__main__":
    test_official_blog_has_only_official_task_results()
    test_complete_blog_labels_and_contains_both_scopes()
    test_generated_files_match_generator()
    print("blogpost publish checks passed")
