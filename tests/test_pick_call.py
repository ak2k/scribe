"""The pick's call: what the model is shown, how its reply is read, and what it changes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from scribe.claude_cli import Completion
from scribe.pick import (
    PICK_PROMPT_VERSION,
    ChunkPick,
    Spot,
    chunk_spans,
    find_spots,
    pick_readings,
    read_reply,
    reference_first,
    system_prompt,
)
from scribe.schema import Engine, Transcript, Word
from scribe.speakers import render
from scribe.vote import norm_tokens
from tests.pick_fakes import (
    answering,
    choosing,
    heard_early_at_an_edge,
    heard_late,
    marks,
    numbered,
    transcript,
    unsure,
)
from tests.speakers_fakes import FakeSpeakerBackend, target_of

if TYPE_CHECKING:
    import random
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from scribe.pick import Picking

CAUSES = {
    "max_tokens",
    "no_out_block",
    "unclosed_out_block",
    "bad_json",
    "foreign_id",
    "repeated_id",
    "missing_id",
}


def _said(*said: tuple[str, int | None]) -> list[Word]:
    """One word a second, each 0.4 s long, with its speaker."""
    return [
        Word(text=text, start=float(index), end=index + 0.4, speaker=speaker)
        for index, (text, speaker) in enumerate(said)
    ]


def _cat_and_mat() -> tuple[Transcript, Transcript]:
    """Two spots, cat/hat shown transcript first and mat/bat shown reference first."""
    return (
        transcript(_said(("the", 0), ("cat", 0), ("sat", 0), ("on", 1), ("the", 1), ("mat", 1))),
        transcript(
            _said(
                ("the", None),
                ("hat", None),
                ("sat", None),
                ("on", None),
                ("the", None),
                ("bat", None),
            ),
            engine="parakeet-mlx",
            model="tdt-0.6b",
        ),
    )


def _completion(text: str, stop_reason: str | None = "end_turn") -> Completion:
    return Completion(text=text, model="fake-model", stop_reason=stop_reason)


def _texts(result: Transcript) -> list[str]:
    return [word.text for word in result.words]


def _spoken(words: Sequence[Word], held: range) -> int:
    """Count the words held that were said: a word of bare punctuation was not."""
    return sum(1 for index in held if norm_tokens(words[index].text))


def _record(result: Transcript) -> list[list[object]]:
    raw = result.engine.params["pick_record"]
    assert isinstance(raw, str)
    return cast("list[list[object]]", json.loads(raw))


def test_the_order_of_a_spots_readings_is_drawn_from_the_spot_and_pinned() -> None:
    # Pinned values: the draw must be the same in every process.
    assert reference_first(1, "cat", "hat") is False
    assert reference_first(2, "mat", "bat") is True


def test_the_model_sees_each_spot_marked_in_place_in_its_order() -> None:
    said, heard = _cat_and_mat()
    backend = FakeSpeakerBackend(reply=answering(unsure))

    pick_readings(said, heard, backend)

    assert backend.calls == [
        (
            system_prompt(),
            "<before>\n(start of recording)\n</before>\n\n"
            "<target>\n<spk:0> the [#1 A: cat | B: hat] sat\n"
            "<spk:1> on the [#2 A: bat | B: mat]\n</target>\n\n"
            "<after>\n(end of recording)\n</after>\n",
        )
    ]


def test_the_reference_reading_is_applied_whichever_label_shows_it() -> None:
    said, heard = _cat_and_mat()
    backend = FakeSpeakerBackend(reply=answering(choosing({"hat", "bat"})))

    picking = pick_readings(said, heard, backend)

    # The reference's reading is B at the first spot and A at the second.
    assert marks(target_of(backend.calls[0][1])) == [(1, "cat", "hat"), (2, "bat", "mat")]
    assert _texts(picking.transcript) == ["the", "hat", "sat", "on", "the", "bat"]
    assert picking.picked == ("reference", "reference")


@pytest.mark.parametrize("unsure", [False, True])
def test_picks_naming_the_transcripts_reading_or_unsure_keep_every_word(unsure: bool) -> None:
    said, heard = _cat_and_mat()

    def keep(number: int, a: str, _b: str) -> str:
        return "unsure" if unsure and number == 2 else "A" if a in {"cat", "mat"} else "B"

    backend = FakeSpeakerBackend(reply=answering(keep))

    picking = pick_readings(said, heard, backend)

    assert picking.transcript.words == said.words
    assert picking.picked == (("transcript", "unsure") if unsure else ("transcript", "transcript"))


def test_a_reading_put_in_is_held_between_the_kept_words_and_takes_the_nearest_speaker() -> None:
    said = transcript(
        [
            Word(text="a", start=1.2, end=1.4, speaker=0),
            Word(text="big", start=1.5, end=1.7, speaker=1),
            Word(text="cat", start=1.8, end=1.9, speaker=2),
            Word(text="sat", start=2.0, end=2.4, speaker=0),
        ]
    )
    heard = transcript(
        [
            Word(text="a", start=1.0, end=1.1),
            Word(text="Pig,", start=1.1, end=1.7),
            Word(text="hat", start=2.1, end=2.2),
            Word(text="sat", start=2.2, end=2.6),
        ]
    )
    backend = FakeSpeakerBackend(reply=answering(choosing({"Pig, hat"})))

    picking = pick_readings(said, heard, backend)

    assert picking.transcript.words == [
        said.words[0],
        Word(text="Pig,", start=1.2, end=1.7, speaker=1),
        Word(text="hat", start=2.0, end=2.2, speaker=2),
        said.words[3],
    ]


def test_a_reading_put_in_between_two_equally_near_words_takes_the_earlier_speaker() -> None:
    said = transcript(
        [
            Word(text="so", start=0.0, end=0.4, speaker=0),
            Word(text="big", start=1.0, end=1.25, speaker=1),
            Word(text="cat", start=1.75, end=2.0, speaker=2),
            Word(text="now", start=3.0, end=3.4, speaker=0),
        ]
    )
    heard = transcript(
        [
            Word(text="so", start=0.0, end=0.4),
            Word(text="hip", start=1.5, end=1.6),
            Word(text="now", start=3.0, end=3.4),
        ]
    )
    backend = FakeSpeakerBackend(reply=answering(choosing({"hip"})))

    picking = pick_readings(said, heard, backend)

    assert picking.transcript.words[1] == Word(text="hip", start=1.5, end=1.6, speaker=1)


def test_a_word_only_the_transcript_heard_survives_a_reference_pick_one_word_away() -> None:
    said = transcript(_said(("please", 0), ("lovely", 0), ("cat", 0), ("sat", 0)))
    heard = transcript(
        [
            Word(text="please", start=0.0, end=0.4),
            Word(text="cat", start=2.0, end=2.4),
            Word(text="slept", start=3.0, end=3.4),
        ],
        engine="parakeet-mlx",
    )
    backend = FakeSpeakerBackend(reply=answering(choosing({"slept"})))

    picking = pick_readings(said, heard, backend)

    assert _texts(picking.transcript) == ["please", "lovely", "cat", "slept"]


@pytest.mark.parametrize(
    ("case", "changed"),
    [(heard_late(0.8), 6), (heard_early_at_an_edge(), 10)],
    ids=["late", "early-at-an-edge"],
)
def test_a_reference_pick_keeps_every_word_both_heard_alike_as_it_was(
    case: tuple[list[Word], list[Word]], changed: int
) -> None:
    said, heard = case
    backend = FakeSpeakerBackend(reply=answering(choosing({heard[changed].text})))

    picking = pick_readings(transcript(said), transcript(heard, engine="parakeet-mlx"), backend)

    words = picking.transcript.words
    assert [word.text for word in words] == [word.text for word in heard]
    assert words[:changed] + words[changed + 1 :] == said[:changed] + said[changed + 1 :]
    assert words[changed].speaker == said[changed].speaker


_COLORS = ("red", "green", "blue", "pink", "gray", "brown", "black", "white", "gold", "teal")


@pytest.mark.parametrize(
    ("said_as", "heard_as", "side"),
    [
        (_COLORS[:10], ("purple",), "guarded"),
        (_COLORS[:6], ("purple",), "guarded"),
        (_COLORS[:6], ("purple", "violet"), "reference"),
        (_COLORS[:1], ("purple", "violet", "lilac", "plum", "mauve", "puce"), "reference"),
        (_COLORS[:7], ("purple", ",", "violet"), "guarded"),
        ((*_COLORS[:2], ".", *_COLORS[2:5]), ("purple",), "reference"),
    ],
    ids=["10-to-1", "6-to-1", "6-to-2", "1-to-6", "7-to-2-and-a-comma", "5-and-a-period-to-1"],
)
def test_a_reference_pick_dropping_five_words_or_more_keeps_the_transcripts(
    said_as: tuple[str, ...], heard_as: tuple[str, ...], side: str
) -> None:
    # Both sides' readings span the same seconds, so "at" is heard when it was said.
    span = max(len(said_as), len(heard_as))

    def spoken(texts: Sequence[str], speaker: int | None) -> list[Word]:
        starts = [2 + index * span / len(texts) for index in range(len(texts))]
        return [
            *_said(("we", speaker), ("saw", speaker)),
            *(
                Word(text=text, start=start, end=start + 0.4, speaker=speaker)
                for text, start in zip(texts, starts, strict=True)
            ),
            Word(text="at", start=2 + span, end=2.4 + span, speaker=speaker),
        ]

    said = transcript(spoken(said_as, 0))
    heard = transcript(spoken(heard_as, None), engine="parakeet-mlx")
    backend = FakeSpeakerBackend(reply=answering(choosing({" ".join(heard_as)})))

    picking = pick_readings(said, heard, backend)

    assert picking.spots == (Spot(range(2, 2 + len(said_as)), range(2, 2 + len(heard_as))),)
    assert picking.picked == (side,)
    guarded = side == "guarded"
    assert _texts(picking.transcript) == _texts(said if guarded else heard)
    if guarded:
        assert picking.transcript.words == said.words
    params = picking.transcript.engine.params
    assert (params["pick_guarded"], params["pick_to_reference"]) == (int(guarded), 1 - guarded)
    assert _record(picking.transcript)[0][4] == side


_BIRDS = ("robin", "wren", "finch", "crow", "owl", "hawk", "swan", "duck", "dove", "lark", "jay")


@pytest.mark.parametrize(
    ("heard_as", "answer", "side"),
    [
        (_BIRDS[:11], "transcript", "restored"),
        (_BIRDS[:10], "transcript", "transcript"),
        (_BIRDS[:11], "unsure", "restored"),
        (_BIRDS[:11], "reference", "reference"),
        ((*_BIRDS[:5], ",", *_BIRDS[5:10]), "transcript", "transcript"),
        (_BIRDS[:11], "fails", "failed"),
    ],
    ids=[
        "10-more-picked-transcript",
        "9-more-picked-transcript",
        "10-more-unsure",
        "10-more-picked-reference",
        "9-more-and-a-comma-picked-transcript",
        "10-more-in-a-failed-chunk",
    ],
)
def test_a_reference_reading_ten_words_longer_is_put_in_whatever_was_answered(
    heard_as: tuple[str, ...], answer: str, side: str
) -> None:
    at = Word(text="at", start=13.0, end=13.4, speaker=0)
    said = transcript([*_said(("we", 0), ("saw", 0), ("red", 0)), at])
    heard = transcript(
        _said(("we", None), ("saw", None), *((text, None) for text in heard_as), ("at", None)),
        engine="parakeet-mlx",
    )
    readings = {"transcript": "red", "reference": " ".join(heard_as)}
    backend = FakeSpeakerBackend(
        reply=answering(choosing({readings[answer]}) if answer in readings else unsure),
        fail_when=lambda _target: answer == "fails",
    )

    picking = pick_readings(said, heard, backend)

    assert picking.spots == (Spot(range(2, 3), range(2, 2 + len(heard_as))),)
    assert picking.picked == (side,)
    params = picking.transcript.engine.params
    assert (
        params["pick_to_reference"],
        params["pick_unsure"],
        params.get("pick_restored", 0),
    ) == (int(side == "reference"), int(side == "unsure"), int(side == "restored"))
    assert _record(picking.transcript)[0][4] == side
    put_in = side in {"restored", "reference"}
    assert _texts(picking.transcript) == _texts(heard if put_in else said)
    if not put_in:
        assert picking.transcript.words == said.words


def test_the_result_records_the_pick_in_its_params() -> None:
    said, heard = _cat_and_mat()
    said = said.model_copy(
        update={"engine": Engine(name="xai-stt", params={"fill_spans": 0}), "text": "as sent"}
    )
    backend = FakeSpeakerBackend(model="opus", reply=answering(choosing({"hat", "mat"})))

    result = pick_readings(said, heard, backend, context="  Ann Lee \n").transcript

    assert result.engine.params == {
        "fill_spans": 0,
        "pick_model": "opus",
        "pick_prompt_version": PICK_PROMPT_VERSION,
        "pick_reference": "parakeet-mlx tdt-0.6b",
        "pick_context_chars": 7,
        "pick_spots": 2,
        "pick_to_reference": 1,
        "pick_guarded": 0,
        "pick_restored": 0,
        "pick_unsure": 0,
        "pick_failed": 0,
        "pick_chunks": 1,
        "pick_chunks_failed": 0,
        "pick_record": json.dumps(
            [[1.0, 1.4, "cat", "hat", "reference"], [5.0, 5.4, "mat", "bat", "transcript"]]
        ),
    }
    assert result.text == "the hat sat on the mat"
    assert result.turns == []
    assert (result.source, result.engine.name) == (said.source, "xai-stt")


def test_no_spot_asks_nothing_and_records_none() -> None:
    said = transcript(_said(("the", 0), ("cat", 0)))
    backend = FakeSpeakerBackend()

    picking = pick_readings(said, said, backend)

    assert backend.calls == []
    assert picking.transcript.words == said.words
    assert (picking.picked, picking.chunks) == ((), ())
    assert picking.transcript.engine.params["pick_spots"] == 0
    assert picking.transcript.engine.params["pick_chunks"] == 0
    assert _record(picking.transcript) == []


def test_a_transcript_whose_starts_step_back_is_refused_before_any_call() -> None:
    said, heard = _cat_and_mat()
    backwards = said.model_copy(update={"words": [said.words[1], said.words[0], *said.words[2:]]})
    backend = FakeSpeakerBackend()

    with pytest.raises(ValueError, match="transcript word 1 starts before word 0"):
        pick_readings(backwards, heard, backend)
    assert backend.calls == []


def test_no_context_and_blank_context_give_the_same_system_prompt() -> None:
    assert system_prompt() == system_prompt(None) == system_prompt("") == system_prompt(" \n\t ")
    assert "Background" not in system_prompt()


def test_context_adds_one_section_holding_its_stripped_text() -> None:
    prompt = system_prompt("\n  People at this meeting: Ann Lee; Bob Roe  \n")

    added = prompt.removeprefix(system_prompt())
    assert added != prompt
    assert added.count("Background on this recording") == 1
    assert added.endswith(":\nPeople at this meeting: Ann Lee; Bob Roe\n")


@pytest.mark.parametrize(("context", "stripped"), [(" Ann Lee ", "Ann Lee"), ("   ", "")])
def test_context_reaches_every_call_and_its_length_the_params(context: str, stripped: str) -> None:
    said, heard = _cat_and_mat()
    backend = FakeSpeakerBackend(reply=answering(unsure))

    picking = pick_readings(said, heard, backend, context=context)

    # Blank, it is the prompt with no context at all.
    assert [system for system, _ in backend.calls] == [system_prompt(stripped)]
    assert picking.transcript.engine.params["pick_context_chars"] == len(stripped)


def test_a_cut_inside_a_spot_moves_to_its_first_word() -> None:
    texts = [f"w{index}." for index in range(10)]
    # Unmoved, the chunks are (0, 3), (3, 6) and (6, 10).
    assert chunk_spans(texts, [], target=3, max_words=4) == [(0, 3), (3, 6), (6, 10)]
    assert chunk_spans(texts, [Spot(range(2, 5), range(2, 5))], target=3, max_words=4) == [
        (0, 2),
        (2, 6),
        (6, 10),
    ]
    assert chunk_spans(texts, [Spot(range(3, 5), range(3, 5))], target=3, max_words=4) == [
        (0, 3),
        (3, 6),
        (6, 10),
    ]


def test_a_cut_whose_move_would_empty_a_chunk_is_dropped() -> None:
    texts = [f"w{index}." for index in range(10)]

    assert chunk_spans(texts, [Spot(range(4), range(4))], target=3, max_words=4) == [
        (0, 6),
        (6, 10),
    ]
    assert chunk_spans(texts, [Spot(range(2, 7), range(2, 7))], target=3, max_words=4) == [
        (0, 2),
        (2, 10),
    ]


def _three_chunks() -> tuple[Transcript, Transcript]:
    """4,500 words cut at 1,500 and 3,000, with a spot in the first and last chunk only."""
    return (
        transcript(numbered(4500)),
        transcript(numbered(4500, {100: "x100", 4000: "x4000"}), engine="parakeet-mlx"),
    )


def test_a_chunk_whose_call_raises_keeps_its_words_and_the_others_apply() -> None:
    said, heard = _three_chunks()
    backend = FakeSpeakerBackend(
        reply=answering(choosing({"x100", "x4000"})), fail_when=lambda target: "[#1 " in target
    )

    picking = pick_readings(said, heard, backend)

    assert picking.picked == ("failed", "reference")
    assert _texts(picking.transcript)[100] == "w100"
    assert _texts(picking.transcript)[4000] == "x4000"
    # The middle chunk holds no spot, so it is not asked.
    assert picking.chunks == (
        ChunkPick(0, (1,), "ExternalServiceError"),
        ChunkPick(1, (2,)),
    )
    params = picking.transcript.engine.params
    assert (params["pick_chunks"], params["pick_chunks_failed"], params["pick_failed"]) == (2, 1, 1)


def test_a_chunk_is_shown_the_words_around_it() -> None:
    said, heard = _three_chunks()
    backend = FakeSpeakerBackend(reply=answering(unsure))

    pick_readings(said, heard, backend)

    last = next(user for _, user in backend.calls if "[#2 " in user)
    before = said.words[2800:3000]
    rendered = render([word.text for word in before], [word.speaker for word in before])
    assert last.startswith(f"<before>\n{rendered}\n</before>\n\n<target>\n")
    assert last.endswith("</target>\n\n<after>\n(end of recording)\n</after>\n")


def test_an_unexpected_error_is_named_by_its_class() -> None:
    said, heard = _cat_and_mat()
    backend = FakeSpeakerBackend(
        fail_when=lambda _target: True, fail_with=lambda: RuntimeError("secret words")
    )

    picking = pick_readings(said, heard, backend)

    assert picking.chunks == (ChunkPick(0, (1, 2), "error: RuntimeError"),)


def test_an_unusable_reply_fails_its_chunk() -> None:
    said = transcript(_said(("the", 0), ("cat", 0), ("sat", 0)))
    heard = transcript(_said(("the", 0), ("hat", 0), ("sat", 0)))
    backend = FakeSpeakerBackend(
        reply=lambda _target: '{"picks": [{"id": 1, "pick": "B", "reason": "r"}]}',
        stop_reason="max_tokens",
    )

    picking = pick_readings(said, heard, backend)

    assert picking.chunks == (ChunkPick(0, (1,), "max_tokens"),)
    assert picking.transcript.words == said.words


@pytest.mark.parametrize(
    ("text", "stop_reason", "cause"),
    [
        ("<out>{}</out>", "max_tokens", "max_tokens"),
        ('{"picks": []}', "end_turn", "no_out_block"),
        ('<out>{"picks": []}', "end_turn", "unclosed_out_block"),
        ("<out>not json</out>", "end_turn", "bad_json"),
        ('<out>{"picks": [{"id": 1, "pick": "A"}]}</out>', "end_turn", "bad_json"),
        ('<out>{"picks": [{"id": 1, "pick": "a", "reason": ""}]}</out>', "end_turn", "bad_json"),
        (
            '<out>{"picks": [{"id": 1, "pick": "A", "reason": "", "sure": true}, '
            '{"id": 2, "pick": "A", "reason": ""}]}</out>',
            "end_turn",
            "bad_json",
        ),
        (
            '<out>{"picks": [{"id": 1, "pick": "A", "reason": ""}, '
            '{"id": 2, "pick": "A", "reason": ""}], "note": ""}</out>',
            "end_turn",
            "bad_json",
        ),
        # An id of another type is not read as the number it could be taken for.
        (
            '<out>{"picks": [{"id": true, "pick": "A", "reason": ""}, '
            '{"id": 2, "pick": "A", "reason": ""}]}</out>',
            "end_turn",
            "bad_json",
        ),
        (
            '<out>{"picks": [{"id": "1", "pick": "A", "reason": ""}, '
            '{"id": 2, "pick": "A", "reason": ""}]}</out>',
            "end_turn",
            "bad_json",
        ),
        (
            '<out>```\n```\n{"picks": [{"id": 1, "pick": "A", "reason": ""}]}\n```\n```</out>',
            "end_turn",
            "bad_json",
        ),
        (
            '<out>{"picks": [{"id": 3, "pick": "A", "reason": ""}, {"id": 1, "pick": "A", '
            '"reason": ""}, {"id": 1, "pick": "A", "reason": ""}]}</out>',
            "end_turn",
            "foreign_id",
        ),
        (
            '<out>{"picks": [{"id": 1, "pick": "A", "reason": ""}, '
            '{"id": 1, "pick": "B", "reason": ""}]}</out>',
            "end_turn",
            "repeated_id",
        ),
        ('<out>{"picks": [{"id": 2, "pick": "A", "reason": ""}]}</out>', "end_turn", "missing_id"),
    ],
)
def test_the_first_cause_that_holds_names_the_failure(
    text: str, stop_reason: str, cause: str
) -> None:
    assert read_reply(_completion(text, stop_reason), {1, 2}) == cause


def _picks_object(picks: Sequence[Mapping[str, object]]) -> str:
    return json.dumps({"picks": picks})


def _out_block(body: str, closed: bool) -> str:
    return f"<out>{body}</out>" if closed else f"<out>{body}"


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=4),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=4), inner),
    max_leaves=6,
)
_PICKS = st.lists(
    st.fixed_dictionaries(
        {
            "id": st.integers(0, 9),
            "pick": st.sampled_from(["A", "B", "unsure"]),
            "reason": st.text(max_size=4),
        }
    ),
    max_size=6,
).map(_picks_object)
# Blocks reach the JSON and id checks, which bare text almost never does.
_REPLIES = st.text() | st.builds(
    _out_block, st.text() | _JSON.map(json.dumps) | _PICKS, st.booleans()
)


@given(
    _REPLIES,
    st.sampled_from([None, "end_turn", "max_tokens"]),
    st.frozensets(st.integers(0, 9)),
)
def test_any_reply_gives_picks_or_a_cause(
    text: str, stop_reason: str | None, asked: frozenset[int]
) -> None:
    read = read_reply(_completion(text, stop_reason), asked)

    event(read if isinstance(read, str) else "picks")
    assert read in CAUSES if isinstance(read, str) else set(read) == asked


@given(
    st.dictionaries(st.integers(0, 99), st.sampled_from(["A", "B", "unsure"]), min_size=1),
    st.sampled_from(["", "```\n", "```json\n"]),
    st.randoms(use_true_random=False),
)
def test_picks_read_back_as_sent_fenced_or_not_in_any_order(
    sent: dict[int, str], fence: str, shuffler: random.Random
) -> None:
    picks = [{"id": number, "pick": label, "reason": "why"} for number, label in sent.items()]
    shuffler.shuffle(picks)
    body = json.dumps({"picks": picks}, indent=2)
    text = f"<out>\n{fence}{body}\n{'```' if fence else ''}\n</out>"

    assert read_reply(_completion(text), set(sent)) == sent


_VOCAB = ["the", "cat", "hat", "sat", "on", "bat", "um", "twenty", "20", "e-mail", "Cat,", ""]


@st.composite
def _cases(draw: st.DrawFn) -> tuple[Transcript, Transcript, list[str]]:
    def words(offset: float) -> list[Word]:
        texts = draw(st.lists(st.sampled_from(_VOCAB), max_size=14))
        return [
            Word(
                text=text,
                start=index * 0.5 + offset,
                end=index * 0.5 + offset + draw(st.sampled_from([0.0, 0.3, 0.9])),
                speaker=draw(st.sampled_from([0, 1, None])),
            )
            for index, text in enumerate(texts)
        ]

    return (
        transcript(words(0.0)),
        transcript(words(draw(st.sampled_from([0.0, 0.2]))), engine="parakeet-mlx"),
        draw(st.lists(st.sampled_from(["A", "B", "unsure"]), min_size=1)),
    )


def _expected(said: Transcript, heard: Transcript, picking: Picking) -> list[Word | str]:
    """The transcript's words, each spot applied replaced by the text of the reference's."""
    expected: list[Word | str] = []
    chosen = {
        spot.transcript.start: spot
        for spot, side in zip(picking.spots, picking.picked, strict=True)
        if side in {"reference", "restored"}
    }
    index = 0
    while index < len(said.words):
        spot = chosen.get(index)
        if spot is None:
            expected.append(said.words[index])
            index += 1
        else:
            expected += [
                word.text for word in heard.words[spot.reference.start : spot.reference.stop]
            ]
            index = spot.transcript.stop
    return expected


@settings(deadline=None)
@given(case=_cases())
def test_only_the_spots_picked_for_the_reference_change_and_starts_stay_in_order(
    case: tuple[Transcript, Transcript, list[str]], tmp_path_factory: pytest.TempPathFactory
) -> None:
    said, heard, labels = case
    backend = FakeSpeakerBackend(
        reply=answering(lambda number, _a, _b: labels[number % len(labels)])
    )

    picking = pick_readings(said, heard, backend)
    written: Path = tmp_path_factory.mktemp("pick") / "picked.json"
    picking.transcript.dump(written)
    loaded = Transcript.load(written)

    expected = _expected(said, heard, picking)
    assert len(loaded.words) == len(expected)
    for word, wanted in zip(loaded.words, expected, strict=True):
        assert word == wanted if isinstance(wanted, Word) else word.text == wanted
    starts = [word.start for word in loaded.words]
    assert starts == sorted(starts)
    assert all(word.end >= word.start for word in loaded.words if word not in said.words)
    assert len(marks("".join(target_of(user) for _, user in backend.calls))) == len(picking.spots)


_KEPT = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
_PUT = ["pig", "fox", "owl"]


@st.composite
def _drops(draw: st.DrawFn) -> tuple[Transcript, Transcript]:
    """Words two a second, heard with one run of up to ten of them as up to three others."""
    before, run, after = (
        draw(st.lists(st.sampled_from(_KEPT), min_size=low, max_size=high))
        for low, high in ((0, 4), (1, 10), (0, 4))
    )
    texts = [*before, *run, *after]
    start, stop = len(before), len(before) + len(run)
    put = draw(st.lists(st.sampled_from(_PUT), max_size=3))
    said = [
        Word(text=text, start=index * 0.5, end=index * 0.5 + 0.3, speaker=0)
        for index, text in enumerate(texts)
    ]
    step = (stop - start) * 0.5 / max(len(put), 1)
    heard = [
        *(word.model_copy(update={"speaker": None}) for word in said[:start]),
        *(
            Word(text=text, start=start * 0.5 + index * step, end=start * 0.5 + index * step + 0.3)
            for index, text in enumerate(put)
        ),
        *(word.model_copy(update={"speaker": None}) for word in said[stop:]),
    ]
    return transcript(said), transcript(heard, engine="parakeet-mlx")


@settings(deadline=None)
@given(case=_drops())
def test_no_reference_pick_applied_drops_five_words_and_the_rest_still_hold(
    case: tuple[Transcript, Transcript],
) -> None:
    said, heard = case
    theirs = {
        " ".join(word.text for word in heard.words[spot.reference.start : spot.reference.stop])
        for spot in find_spots(said.words, heard.words)
    }
    backend = FakeSpeakerBackend(reply=answering(choosing(theirs)))

    picking = pick_readings(said, heard, backend)

    for spot, side in zip(picking.spots, picking.picked, strict=True):
        dropped = _spoken(said.words, spot.transcript) - _spoken(heard.words, spot.reference)
        event(f"{side}, {'5 or more' if dropped >= 5 else 'fewer'} dropped")
        assert side != "reference" or dropped < 5
        assert side != "guarded" or dropped >= 5
    expected = _expected(said, heard, picking)
    words = picking.transcript.words
    assert len(words) == len(expected)
    for word, wanted in zip(words, expected, strict=True):
        assert word == wanted if isinstance(wanted, Word) else word.text == wanted
    if "reference" not in picking.picked:
        assert words == said.words
    starts = [word.start for word in words]
    assert starts == sorted(starts)
