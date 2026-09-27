from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.errors import InputValidationError
from scribe.outputs import plan_outputs, sibling


def _input(tmp_path: Path) -> Path:
    source = tmp_path / "in.json"
    source.write_text("{}\n", encoding="utf-8")
    return source


def test_distinct_outputs_are_planned_as_given(tmp_path: Path) -> None:
    source = _input(tmp_path)
    (tmp_path / "old.md").write_text("replaced by the run\n", encoding="utf-8")
    outputs = {"--out": tmp_path / "old.md", "sidecar": tmp_path / "new.json"}

    plan = plan_outputs(outputs, {"the input": source})

    assert dict(plan.paths) == outputs
    assert plan["sidecar"] == tmp_path / "new.json"


def test_an_output_that_is_a_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "out.md").mkdir()

    with pytest.raises(InputValidationError, match="is a directory"):
        plan_outputs({"--out": tmp_path / "out.md"}, {"the input": _input(tmp_path)})


def test_an_output_that_is_an_input_is_refused(tmp_path: Path) -> None:
    source = _input(tmp_path)
    (tmp_path / "link.md").hardlink_to(source)

    with pytest.raises(InputValidationError, match="is the input"):
        plan_outputs({"--out": tmp_path / "link.md"}, {"the input": source})


def test_an_output_named_through_its_input_directory_is_refused(tmp_path: Path) -> None:
    # Neither `absent` nor the path through it exists, so only the resolved
    # path can tell that it names the input.
    source = _input(tmp_path)

    with pytest.raises(InputValidationError, match="is the input"):
        plan_outputs(
            {"--out": tmp_path / "absent" / ".." / "in.json"},
            {"the input": source},
            missing_parent_ok=True,
        )


def test_two_outputs_that_are_one_file_are_refused(tmp_path: Path) -> None:
    (tmp_path / "out.md").write_text("earlier output\n", encoding="utf-8")
    (tmp_path / "out.json").symlink_to(tmp_path / "out.md")

    with pytest.raises(InputValidationError, match=r"sidecar .* is the same file as --out"):
        plan_outputs(
            {"--out": tmp_path / "out.md", "sidecar": tmp_path / "out.json"},
            {"the input": _input(tmp_path)},
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param("Talk.clean.md", "talk.clean.md", id="case"),
        pytest.param(
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}.md",
            "cafe\N{COMBINING ACUTE ACCENT}.md",
            id="normalization",
        ),
    ],
)
def test_two_new_outputs_named_alike_but_for_case_or_form_are_refused(
    tmp_path: Path, first: str, second: str
) -> None:
    # A case- and normalization-insensitive volume, the macOS default, holds
    # these as one file, so the later write would replace the earlier.
    with pytest.raises(InputValidationError, match=r"--curated .* is the same file as --out"):
        plan_outputs(
            {"--out": tmp_path / first, "--curated": tmp_path / second},
            {"the input": _input(tmp_path)},
        )


def test_new_outputs_named_alike_in_two_directories_are_planned(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    outputs = {
        "--out": tmp_path / "a" / "Talk.clean.md",
        "--curated": tmp_path / "b" / "talk.clean.md",
    }

    plan = plan_outputs(outputs, {"the input": _input(tmp_path)})

    assert dict(plan.paths) == outputs


def test_an_output_in_a_missing_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="not a directory"):
        plan_outputs({"--out": tmp_path / "typo" / "out.md"}, {"the input": _input(tmp_path)})


def test_a_missing_directory_is_accepted_when_the_caller_creates_it(tmp_path: Path) -> None:
    plan = plan_outputs(
        {"md": tmp_path / "new" / "out.md"},
        {"the input": _input(tmp_path)},
        missing_parent_ok=True,
    )

    assert plan["md"] == tmp_path / "new" / "out.md"
    assert not (tmp_path / "new").exists()


def test_a_parent_that_is_a_file_is_refused_even_when_missing_ones_are_accepted(
    tmp_path: Path,
) -> None:
    (tmp_path / "blocker").write_text("a file where a directory must go\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="not a directory"):
        plan_outputs(
            {"md": tmp_path / "blocker" / "out.md"},
            {"the input": _input(tmp_path)},
            missing_parent_ok=True,
        )


@given(st.lists(st.from_regex(r"[a-z]{1,8}", fullmatch=True), min_size=1, max_size=6))
def test_a_plan_refuses_exactly_the_repeated_names(names: list[str]) -> None:
    # Paths that do not exist compare by resolved path; the parent is a real
    # directory so only the collision rule can refuse.
    root = Path(__file__).parent
    outputs = {f"out{index}": root / f"absent-{name}.md" for index, name in enumerate(names)}

    if len(set(names)) == len(names):
        assert dict(plan_outputs(outputs, {}).paths) == outputs
    else:
        with pytest.raises(InputValidationError, match="is the same file as"):
            plan_outputs(outputs, {})


@pytest.mark.parametrize(
    ("name", "named"),
    [
        ("talk.transcript.json", "talk.xai.json"),
        ("talk.json", "talk.xai.json"),
        ("talk", "talk.xai.json"),
        ("talk.transcript.json.bak", "talk.transcript.json.bak.xai.json"),
    ],
)
def test_a_sibling_replaces_a_trailing_transcript_suffix(name: str, named: str) -> None:
    assert sibling(Path("/d") / name, ".xai.json") == Path("/d") / named
