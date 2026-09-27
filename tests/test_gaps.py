from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from scribe import gaps
from scribe.cli import app
from scribe.errors import ExternalServiceError, InputValidationError, ToolMissingError
from scribe.gaps import Gap, find_gaps, format_gap, frame_levels
from scribe.schema import Engine, Source, Transcript, Turn, Word

if TYPE_CHECKING:
    from collections.abc import Callable

    Runner = Callable[..., subprocess.CompletedProcess[str]]

KEY = "lavfi.astats.Overall.RMS_level"
SPEECH = -20.0
runner = CliRunner()


def _words(*spans: tuple[float, float]) -> list[Word]:
    return [Word(text="w", start=start, end=end) for start, end in spans]


def _printed(*levels: str) -> str:
    return "".join(f"frame:{i}    pts:{i * 800}\n{KEY}={level}\n" for i, level in enumerate(levels))


def _runner(stdout: str, *, code: int = 0, calls: list[list[str]] | None = None) -> Runner:
    def run(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        (calls if calls is not None else []).append(argv)
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="Invalid data found")

    return run


def test_frame_levels_reads_one_level_per_frame_and_silence_as_minus_120(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.chdir(tmp_path)
    # Relative, with a colon ffmpeg would read as a protocol unless it is made absolute.
    audio = Path("meeting:1.mp3")

    levels = frame_levels(audio, _runner(_printed("-21.5", "-inf", "-80.25"), calls=calls))

    assert levels == [-21.5, -120.0, -80.25]
    assert calls == [
        [
            *["ffmpeg", "-nostdin", "-v", "error", "-i", str(Path.cwd() / audio), "-vn", "-af"],
            "aresample=16000,aformat=channel_layouts=mono,asetnsamples=n=800:p=0,"
            "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
            f"ametadata=mode=print:key={KEY}:file=-",
            *["-f", "null", "-"],
        ]
    ]


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("", "no audio frames"),
        ("frame:0    pts:0\n", "frame 0 without its level"),
        (_printed("-20.0") + "frame:1    pts:800\n", "frame 1 without its level"),
        (_printed("-20.0", "-30.0").replace(f"{KEY}=-20.0\n", ""), "unexpected line: 'frame:1"),
        (_printed("-20.0").replace("frame:0", "frame:1"), "unexpected line: 'frame:1"),
        (_printed("-20.0", "loud"), "not a number: 'loud'"),
        (_printed("nan"), "not a number"),
        (_printed("inf"), "not a number"),
        (_printed("-20.0") + "Overall.Peak_level=-3.0\n", "unexpected line"),
    ],
)
def test_frame_levels_refuses_output_that_is_not_levels(stdout: str, message: str) -> None:
    with pytest.raises(ExternalServiceError, match=message):
        frame_levels(Path("a.mp3"), _runner(stdout))


def _unrunnable(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
    raise FileNotFoundError(2, "No such file or directory")


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (_runner("", code=1), "ffmpeg exited 1: Invalid data found"),
        (_unrunnable, "cannot run ffmpeg: No such file or directory"),
    ],
)
def test_a_failed_ffmpeg_is_an_error(run: Runner, message: str) -> None:
    with pytest.raises(ExternalServiceError, match=message):
        frame_levels(Path("a.mp3"), run)


def _with(levels: list[float], first: int, stop: int, level: float) -> list[float]:
    return [*levels[:first], *[level] * (stop - first), *levels[stop:]]


@pytest.mark.parametrize(
    ("second", "expected"), [(7.0, [Gap(1.0, 7.0, 0.0)]), (6.95, [])], ids=["6.0-s", "5.95-s"]
)
def test_a_hole_is_a_candidate_from_exactly_six_seconds(second: float, expected: list[Gap]) -> None:
    assert find_gaps(_words((0.0, 1.0), (second, 8.0)), [SPEECH] * 160) == expected


def test_a_six_second_hole_between_decimal_times_is_a_candidate() -> None:
    # 8.2 - 2.2 is 5.999999999999999 in binary floating point.
    flagged = find_gaps(_words((1.0, 2.2), (8.2, 9.0)), [SPEECH] * 200)

    assert flagged == [Gap(2.2, 8.2, 0.0)]


@pytest.mark.parametrize(
    ("relative", "expected"), [(-25.0, [Gap(1.0, 7.0, -25.0)]), (-25.1, [])], ids=["-25.0", "-25.1"]
)
def test_a_hole_is_flagged_from_exactly_25_db_below_speech(
    relative: float, expected: list[Gap]
) -> None:
    levels = _with([SPEECH] * 160, 20, 140, SPEECH + relative)

    assert find_gaps(_words((0.0, 1.0), (7.0, 8.0)), levels) == expected


def test_a_hole_is_judged_by_the_inclusive_90th_percentile_of_its_frames() -> None:
    # 12 of the 120 frames at speech level put it a tenth of the way from -40 to -20 dB.
    # They sit mid-hole, so a window one frame off takes in a word's frame and counts 13.
    levels = _with(_with([SPEECH] * 160, 20, 140, -40.0), 60, 72, SPEECH)

    assert find_gaps(_words((0.0, 1.0), (7.0, 8.0)), levels) == [Gap(1.0, 7.0, -18.0)]


def test_a_stretched_word_covers_only_its_first_two_seconds() -> None:
    assert find_gaps(_words((0.0, 11.1), (12.0, 13.0)), [SPEECH] * 260) == [Gap(2.0, 12.0, 0.0)]


def test_a_word_that_ends_before_it_starts_still_closes_the_hole_at_its_start() -> None:
    flagged = find_gaps(_words((0.0, 1.0), (3.0, 2.5), (20.0, 21.0)), [SPEECH] * 440)

    assert flagged == [Gap(3.0, 20.0, 0.0)]


def test_a_word_starting_before_the_audio_covers_no_frame_from_its_end() -> None:
    # Read from the end, the last second's 0 dB would lift speech to 0 dB and hide the hole.
    levels = [*[-20.0] * 10, *[-40.0] * 130, -20.0, *[-40.0] * 39, *[0.0] * 20]

    assert find_gaps(_words((-1.0, 0.5), (7.0, 7.0)), levels) == [Gap(0.5, 7.0, -20.0)]


def test_a_frame_two_words_cover_counts_once_in_the_speech_level() -> None:
    # Counted twice, the loud first half second would lift speech to -25 dB and
    # put the -60 dB hole out of reach.
    levels = [*[-10.0] * 10, *[-40.0] * 10, *[-60.0] * 130, *[-40.0] * 10]
    words = _words((0.0, 0.5), (0.0, 0.5), (0.5, 1.0), (7.5, 8.0))

    assert find_gaps(words, levels) == [Gap(1.0, 7.5, -20.0)]


def test_the_start_and_end_of_the_audio_bound_holes_too() -> None:
    # The second word starts past the 16 s of audio, so it closes nothing.
    flagged = find_gaps(_words((7.0, 8.0), (30.0, 31.0)), [SPEECH] * 320)

    assert flagged == [Gap(0.0, 7.0, 0.0), Gap(8.0, 16.0, 0.0)]


@pytest.mark.parametrize(("frames", "expected"), [(120, [Gap(0.0, 6.0, None)]), (119, [])])
def test_no_words_makes_the_whole_audio_one_hole_with_no_level(
    frames: int, expected: list[Gap]
) -> None:
    assert find_gaps([], [-120.0] * frames) == expected


def test_words_outside_the_audio_are_refused() -> None:
    with pytest.raises(InputValidationError, match="not of this audio"):
        find_gaps(_words((10.0, 11.0)), [SPEECH] * 20)


# Quarter seconds are exact in binary, so no hole length rounds across a threshold.
# Negative starts and ends before starts are valid in the schema, so they are drawn too.
_times = st.integers(-20, 400).map(lambda quarters: quarters / 4)
_lengths = st.integers(-40, 24).map(lambda quarters: quarters / 4)
_spans = st.lists(st.tuples(_times, _lengths), min_size=1, max_size=30).filter(
    lambda spans: any(start >= 0 for start, _ in spans)
)


@st.composite
def _cases(draw: st.DrawFn) -> tuple[list[Word], list[float]]:
    spans = draw(_spans)
    # Words that share a start, whose order among themselves must not matter either.
    twins = st.tuples(st.sampled_from([start for start, _ in spans]), _lengths)
    spans += draw(st.lists(twins, max_size=5))
    words = [Word(text="w", start=start, end=start + length) for start, length in spans]
    frames = int(max(word.start for word in words) * 20) + draw(st.integers(1, 2400))
    # One level per second, repeated: loud and quiet holes both occur.
    per_second = draw(st.lists(st.floats(-120.0, 0.0), min_size=1, max_size=40))
    return words, [per_second[(i // 20) % len(per_second)] for i in range(frames)]


@given(_cases())
def test_every_flag_is_at_least_six_seconds_and_holds_no_word_start(
    case: tuple[list[Word], list[float]],
) -> None:
    words, levels = case
    for gap in find_gaps(words, levels):
        assert gap.end - gap.start >= 6.0
        assert not any(gap.start < word.start < gap.end for word in words)


@given(_cases(), st.data())
def test_the_flags_do_not_depend_on_word_order(
    case: tuple[list[Word], list[float]], data: st.DataObject
) -> None:
    words, levels = case

    assert find_gaps(data.draw(st.permutations(words)), levels) == find_gaps(words, levels)


@st.composite
def _around_a_hole(draw: st.DrawFn) -> tuple[list[Word], float, float]:
    """Words of at most 2 s, a hole of at least 8 s from the latest end among them, more words."""
    short = st.integers(0, 8).map(lambda quarters: quarters / 4)
    offsets = st.integers(0, 100).map(lambda quarters: quarters / 4)
    before = [
        Word(text="w", start=2.0 + t, end=2.0 + t + draw(short))
        for t in draw(st.lists(offsets, min_size=1, max_size=10))
    ]
    edge = max(word.end for word in before)
    stop = edge + draw(st.integers(32, 240)) / 4
    after = [
        Word(text="w", start=stop + t, end=stop + t + draw(short))
        for t in [0.0, *draw(st.lists(offsets, max_size=4))]
    ]
    return [*before, *after], edge, stop


def _frames_to(words: list[Word]) -> int:
    return int((max(word.end for word in words) + 10.0) * 20)


# Whole decibels, so a uniform level's percentile comes back exact.
@given(_around_a_hole(), st.integers(-60, 0).map(float))
def test_a_hole_at_speech_level_is_flagged_and_a_silent_one_is_not(
    case: tuple[list[Word], float, float], speech: float
) -> None:
    words, edge, stop = case
    level = [speech] * _frames_to(words)
    silent = _with(level, int(edge * 20), int(stop * 20), -120.0)

    assert Gap(edge, stop, 0.0) in find_gaps(words, level)
    assert all(gap.end <= edge or gap.start >= stop for gap in find_gaps(words, silent))


@given(_around_a_hole(), st.integers(1, 7), st.integers(9, 48))
def test_an_earlier_long_word_over_the_hole_start_shortens_it(
    case: tuple[list[Word], float, float], lead: int, length: int
) -> None:
    words, edge, stop = case
    long_word = Word(text="w", start=edge - lead / 4, end=edge - lead / 4 + length / 4)
    level = [SPEECH] * _frames_to(words)

    flagged = find_gaps([*words, long_word], level)

    assert Gap(edge - lead / 4 + 2.0, stop, 0.0) in flagged
    assert all(gap.start != edge for gap in flagged)


@pytest.mark.parametrize(
    ("gap", "line"),
    [
        (Gap(59.96, 3725.04, -3.14159), "00:01:00.0-01:02:05.0\t3665.0 s\t-3.1 dB"),
        (Gap(1.04, 7.06, 0.0), "00:00:01.0-00:00:07.1\t6.1 s\t0.0 dB"),
        (Gap(0.0, 6.0, None), "00:00:00.0-00:00:06.0\t6.0 s\t- dB"),
    ],
)
def test_a_flag_prints_its_span_rounded_to_tenths_first(gap: Gap, line: str) -> None:
    assert format_gap(gap) == line


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    *,
    words: list[Word],
    turns: list[Turn] | None = None,
    found: str | None = "/usr/bin/ffmpeg",
    code: int = 0,
) -> tuple[int, str, str, list[list[str]]]:
    calls: list[list[str]] = []
    run = _runner(stdout, code=code, calls=calls)
    checked = partial(gaps.check_gaps, run=run, which=lambda _name: found)
    monkeypatch.setattr("scribe.cli.check_gaps", checked)
    source = Source(kind="audio", ref="a.mp3")
    Transcript(
        source=source, engine=Engine(name="xai-stt"), text="", words=words, turns=turns or []
    ).dump(tmp_path / "t.json")
    (tmp_path / "a.mp3").write_bytes(b"audio")
    result = runner.invoke(app, ["gaps", str(tmp_path / "t.json"), str(tmp_path / "a.mp3")])
    return result.exit_code, result.stdout, result.stderr, calls


@pytest.mark.parametrize(
    ("hole", "lines", "count"),
    [("-30.0", "00:00:01.0-00:00:07.0\t6.0 s\t-10.0 dB\n", "1 hole"), ("-inf", "", "0 holes")],
)
def test_the_command_prints_one_line_per_flag_exits_0_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hole: str, lines: str, count: str
) -> None:
    printed = _printed(*["-20.0"] * 20, *[hole] * 120, *["-20.0"] * 20)

    code, stdout, stderr, _ = _invoke(
        tmp_path, monkeypatch, printed, words=_words((0.0, 1.0), (7.0, 8.0))
    )

    assert (code, stdout) == (0, lines)
    assert stderr == f"scribe: {count} flagged; speech level -20.0 dB\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.mp3", "t.json"]


def test_a_transcript_without_words_or_turns_is_one_hole_with_no_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, stdout, stderr, _ = _invoke(tmp_path, monkeypatch, _printed(*["-inf"] * 140), words=[])

    assert (code, stdout) == (0, "00:00:00.0-00:00:07.0\t7.0 s\t- dB\n")
    assert stderr == "scribe: 1 hole flagged; no words, so no speech level\n"


def test_ffmpeg_missing_from_path_is_its_own_error_and_nothing_is_spawned() -> None:
    calls: list[list[str]] = []
    transcript = Transcript(
        source=Source(kind="audio", ref="a.mp3"), engine=Engine(name="x"), text=""
    )

    with pytest.raises(ToolMissingError, match="ffmpeg is not on PATH"):
        gaps.check_gaps(
            transcript, Path("a.mp3"), run=_runner("", calls=calls), which=lambda _: None
        )

    assert calls == []


_TURN = Turn(speaker="Speaker 1", start=0.0, end=1.0, text="hi")


@pytest.mark.parametrize(
    ("words", "turns", "found", "code", "spawned", "message"),
    [
        ([], [_TURN], "/usr/bin/ffmpeg", 0, 0, "has turns but no words"),
        (_words((0.0, 1.0)), None, None, 0, 0, "ffmpeg is not on PATH"),
        (_words((0.0, 1.0)), None, "/usr/bin/ffmpeg", 1, 1, "ffmpeg exited 1: Invalid data"),
        (_words((9.0, 10.0)), None, "/usr/bin/ffmpeg", 0, 1, "not of this audio"),
    ],
    ids=["turns-without-words", "no-ffmpeg", "ffmpeg-fails", "other-audio"],
)
def test_a_check_that_cannot_run_exits_2_with_one_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    words: list[Word],
    turns: list[Turn] | None,
    found: str | None,
    code: int,
    spawned: int,
    message: str,
) -> None:
    exit_code, stdout, stderr, calls = _invoke(
        tmp_path, monkeypatch, _printed("-20.0"), words=words, turns=turns, found=found, code=code
    )

    assert (exit_code, stdout) == (2, "")
    assert stderr.startswith("scribe: ")
    assert message in stderr
    assert len(stderr.splitlines()) == 1
    assert len(calls) == spawned


@pytest.mark.parametrize("missing", ["t.json", "a.mp3"])
def test_a_missing_input_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr("scribe.cli.check_gaps", partial(gaps.check_gaps, run=_runner("")))
    Transcript(source=Source(kind="audio", ref="a.mp3"), engine=Engine(name="x"), text="").dump(
        tmp_path / "t.json"
    )
    (tmp_path / "a.mp3").write_bytes(b"audio")
    (tmp_path / missing).unlink()

    result = runner.invoke(app, ["gaps", str(tmp_path / "t.json"), str(tmp_path / "a.mp3")])

    assert result.exit_code == 2
    assert result.stderr.startswith("scribe: cannot read")
    assert len(result.stderr.splitlines()) == 1


def test_the_help_says_it_needs_ffmpeg_and_writes_and_sends_nothing() -> None:
    result = runner.invoke(app, ["gaps", "--help"], terminal_width=200)
    text = " ".join(result.stdout.split())

    assert result.exit_code == 0
    assert "Needs ffmpeg on PATH" in text
    assert "Writes no file and sends nothing over the network" in text
