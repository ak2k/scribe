from __future__ import annotations

import hashlib
import json
import random
import re
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from structlog.testing import capture_logs

from scribe.errors import AppError
from scribe.schema import Word
from scribe.speakers import (
    CONTEXT_WORDS,
    align_labels,
    cut_points,
    needs_relabeling,
    parse_reply,
    prompts,
    relabel,
    render,
)
from scribe.turns import turns_from_speakers
from tests.speakers_fakes import FakeSpeakerBackend, target_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from scribe.speakers import Relabeling


_TAGS = re.compile(r"<spk:[^>]*>\s*")


def _texts(count: int, *, ends: set[int] | None = None) -> list[str]:
    """Words named w0, w1, ...; the ones at `ends` close a sentence."""
    return [f"w{index}." if index in (ends or set()) else f"w{index}" for index in range(count)]


def test_a_short_transcript_is_one_chunk() -> None:
    assert cut_points(_texts(900)) == [(0, 900)]
    assert cut_points([]) == []


def test_a_chunk_ends_at_the_first_sentence_end_from_its_target_word_on() -> None:
    # Word 698 is the 699th: one short of the target, so it cannot end a chunk.
    texts = _texts(2000, ends={698, 749, 760})

    assert cut_points(texts)[0] == (0, 750)


def test_a_chunk_without_a_sentence_end_stops_at_the_hard_cap() -> None:
    assert cut_points(_texts(2000)) == [(0, 900), (900, 1800), (1800, 2000)]


def test_the_cut_accepts_every_sentence_end_mark() -> None:
    for mark in ".?!":
        texts = _texts(1000)
        texts[799] += mark
        assert cut_points(texts)[0] == (0, 800)


def test_render_opens_each_run_with_its_tag_on_a_new_line() -> None:
    assert render(["Hi.", "Hello", "there.", "Bye."], [0, 1, 1, 0]) == (
        "<spk:0> Hi.\n<spk:1> Hello there.\n<spk:0> Bye."
    )


def test_each_prompt_carries_the_words_before_it_as_context() -> None:
    texts = _texts(1000, ends={749})
    ids = [0] * 500 + [1] * 500

    (system, first), (same_system, second) = prompts(texts, ids, cut_points(texts))

    assert system == same_system
    assert "Speaker ids in this call: 0, 1." in system
    assert system.rstrip().endswith("between <out> and </out>, and nothing else.")
    assert first.startswith("<context>\n(start of call)\n</context>")
    assert target_of(first) == render(texts[:750], ids[:750])
    assert second.startswith(
        f"<context>\n{render(texts[750 - CONTEXT_WORDS : 750], ids[750 - CONTEXT_WORDS : 750])}\n"
    )
    assert target_of(second) == render(texts[750:], ids[750:])


def test_a_prompt_lists_every_id_in_the_recording_not_just_its_chunks() -> None:
    texts = _texts(1000, ends={749})
    ids = [2] * 10 + [0] * 990

    asked = prompts(texts, ids, cut_points(texts))

    assert all("Speaker ids in this call: 0, 2." in system for system, _ in asked)


def test_parse_reply_reads_labels_inside_the_out_block() -> None:
    reply = "Sure.\n<out>\n<spk:0> Hi there.\n<spk:1>Hello.\n</out>\ntrailing"

    assert parse_reply(reply) == [(0, "Hi"), (0, "there."), (1, "Hello.")]


def test_parse_reply_without_an_out_block_reads_the_whole_reply() -> None:
    assert parse_reply("orphan <spk:3> one two") == [(None, "orphan"), (3, "one"), (3, "two")]


def test_a_reply_that_moves_a_sentence_moves_only_its_labels() -> None:
    texts = ["So", "what", "now?", "We", "ship", "it.", "Okay."]
    labels = [0, 0, 0, 0, 0, 0, 1]
    reply = parse_reply("<out><spk:0> So what now?\n<spk:1> We ship it. Okay.</out>")

    moved, aligned = align_labels(texts, labels, reply, {0, 1})

    assert moved == [0, 0, 0, 1, 1, 1, 1]
    assert aligned == 7


def test_matching_ignores_case_punctuation_and_curly_quotes() -> None:
    texts = ["It's", "done,", "Bob."]
    reply = parse_reply("<spk:1> it\N{RIGHT SINGLE QUOTATION MARK}s DONE bob")

    assert align_labels(texts, [0, 0, 0], reply, {0, 1}) == ([1, 1, 1], 3)


def test_an_unknown_id_and_an_unmatched_word_keep_the_input_label() -> None:
    texts = ["one", "two", "three", "four"]
    reply = parse_reply("<spk:9> one <spk:1> two <spk:1> tree four")

    labels, aligned = align_labels(texts, [0, 0, 0, 0], reply, {0, 1})

    assert labels == [0, 1, 0, 1]
    assert aligned == 3


_REPLY_WORDS = st.sampled_from(["w0", "w1", "w2", "W3.", "junk", "w5", "", "<spk:1>"])


@given(
    labels=st.lists(st.none() | st.integers(min_value=0, max_value=2), min_size=0, max_size=30),
    reply=st.lists(
        st.tuples(st.none() | st.integers(min_value=0, max_value=50), _REPLY_WORDS), max_size=40
    ),
)
def test_no_reply_changes_the_word_count_or_invents_a_label(
    labels: list[int | None], reply: list[tuple[int | None, str]]
) -> None:
    texts = [f"w{index % 6}" for index in range(len(labels))]
    allowed = {0, 1, 2}

    out, aligned = align_labels(texts, labels, reply, allowed)

    assert len(out) == len(texts)
    assert all(new in allowed or new == old for old, new in zip(labels, out, strict=True))
    assert [old is None for old in labels] == [new is None for new in out]
    assert aligned <= min(len(texts), len(reply))


def _said(*runs: tuple[int | None, int]) -> list[Word]:
    """(speaker, word count) runs, one word per second, every tenth word ending a sentence."""
    said = [speaker for speaker, count in runs for _ in range(count)]
    return [
        Word(
            text=f"w{index}." if index % 10 == 9 else f"w{index}",
            start=float(index),
            end=index + 0.9,
            speaker=speaker,
        )
        for index, speaker in enumerate(said)
    ]


def _swap_ranks(target: str) -> str:
    return (
        target.replace("<spk:0>", "<spk:x>")
        .replace("<spk:1>", "<spk:0>")
        .replace("<spk:x>", "<spk:1>")
    )


def test_ranks_in_the_reply_map_back_to_diarization_ids() -> None:
    words = _said((7, 5), (4, 5))
    backend = FakeSpeakerBackend(reply=_swap_ranks)

    result = relabel([word.text for word in words], [7] * 5 + [4] * 5, backend)

    assert result.speakers == (4,) * 5 + (7,) * 5
    assert result.relabeled == 10
    assert result.failed == ()
    assert "<spk:0> w0" in backend.calls[0][1]


def test_an_echoed_reply_changes_nothing() -> None:
    words = _said((1, 800), (2, 800))
    speakers = [word.speaker for word in words]
    backend = FakeSpeakerBackend()

    result = relabel([word.text for word in words], speakers, backend)

    assert result.speakers == tuple(speakers)
    assert [chunk.status for chunk in result.chunks] == ["ok", "ok"]
    assert result.relabeled == 0
    assert all(chunk.reply_words == chunk.aligned_words for chunk in result.chunks)


def test_one_speaker_makes_no_call() -> None:
    backend = FakeSpeakerBackend()

    result = relabel(["a", "b."], [None, None], backend)

    assert not needs_relabeling([None, None])
    assert result.speakers == (None, None)
    assert result.chunks == ()
    assert backend.calls == []


def _fails_on_w900(target: str) -> bool:
    return " w900 " in f" {target} "


def test_a_failed_chunk_keeps_its_labels_and_logs_a_warning() -> None:
    words = _said((1, 1000), (2, 1000))
    speakers = [word.speaker for word in words]
    backend = FakeSpeakerBackend(reply=_swap_ranks, fail_when=_fails_on_w900)

    with capture_logs() as logs:
        result = relabel([word.text for word in words], speakers, backend)

    assert [(chunk.start, chunk.end) for chunk in result.chunks] == [
        (0, 700),
        (700, 1400),
        (1400, 2000),
    ]
    assert result.failed == (1,)
    assert result.chunks[1].reason == "ExternalServiceError"
    assert result.speakers[700:1400] == tuple(speakers[700:1400])
    assert result.speakers[:700] == (2,) * 700
    warnings = [entry for entry in logs if entry["event"] == "speakers.chunk_failed"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert warnings[0]["chunk"] == 1


def _always(_target: str) -> bool:
    return True


def test_every_chunk_failing_leaves_every_label() -> None:
    words = _said((1, 1000), (2, 1000))
    speakers = [word.speaker for word in words]

    result = relabel([word.text for word in words], speakers, FakeSpeakerBackend(fail_when=_always))

    assert result.speakers == tuple(speakers)
    assert result.failed == (0, 1, 2)


def test_calls_run_at_once_up_to_the_bound() -> None:
    words = _said((1, 2500), (2, 2500))
    backend = FakeSpeakerBackend(delay_s=0.05)

    relabel([word.text for word in words], [word.speaker for word in words], backend, concurrency=3)

    assert len(backend.calls) == len(cut_points([word.text for word in words]))
    assert backend.peak == 3


def _garbled(seed: int) -> Callable[[str], str]:
    def reply(target: str) -> str:
        rng = random.Random(f"{seed}:{target[:40]}")  # noqa: S311  # test data, not security
        tokens = target.split()
        rng.shuffle(tokens)
        kept = [token for token in tokens if rng.random() > 0.2]
        kept.insert(rng.randrange(len(kept) + 1), f"<spk:{rng.randrange(9)}> added")
        return " ".join(kept)

    return reply


@settings(max_examples=25)
@given(
    runs=st.lists(
        st.tuples(st.sampled_from([None, 0, 3, 8]), st.integers(min_value=1, max_value=400)),
        min_size=1,
        max_size=8,
    ),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_no_reply_changes_the_words_turns_are_built_from(
    runs: list[tuple[int | None, int]], seed: int
) -> None:
    words = _said(*runs)
    speakers = [word.speaker for word in words]
    backend = FakeSpeakerBackend(reply=_garbled(seed))

    result = relabel([word.text for word in words], speakers, backend)

    assert len(result.speakers) == len(words)
    assert set(result.speakers) <= set(speakers)
    assert [new is None for new in result.speakers] == [old is None for old in speakers]
    turns = turns_from_speakers(words, result.speakers)
    assert " ".join(turn.text for turn in turns) == " ".join(word.text for word in words)


def _replying(text: str) -> Callable[[str], str]:
    def reply(_target: str) -> str:
        return text

    return reply


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("", "empty_reply"),
        ("  \n", "empty_reply"),
        ("I can't help with relabeling this transcript.", "no_out_block"),
        ("<spk:1> w0 w1 w2", "no_out_block"),
        ("<out></out>", "no_speaker_tags"),
        ("<out>w0 w1 w2</out>", "no_speaker_tags"),
        ("<out><spk:1> nothing like the target</out>", "no_words_aligned"),
    ],
)
def test_an_unusable_reply_fails_its_chunk(reply: str, reason: str) -> None:
    words = _said((1, 5), (2, 5))
    speakers = [word.speaker for word in words]
    backend = FakeSpeakerBackend(reply=_replying(reply), wrap=False)

    with capture_logs() as logs:
        result = relabel([word.text for word in words], speakers, backend)

    assert result.speakers == tuple(speakers)
    assert result.failed == (0,)
    assert result.chunks[0].reason == reason
    warnings = [entry for entry in logs if entry["event"] == "speakers.chunk_failed"]
    assert [(entry["log_level"], entry["reason"]) for entry in warnings] == [("warning", reason)]


def _undecodable() -> BaseException:
    return UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


def test_an_unexpected_error_fails_only_its_chunk_and_logs_no_prompt() -> None:
    words = _said((1, 1000), (2, 1000))
    speakers = [word.speaker for word in words]
    backend = FakeSpeakerBackend(
        reply=_swap_ranks, fail_when=_fails_on_w900, fail_with=_undecodable
    )

    with capture_logs() as logs:
        result = relabel([word.text for word in words], speakers, backend)

    assert result.failed == (1,)
    assert result.chunks[1].reason == "error: UnicodeDecodeError"
    assert result.speakers[700:1400] == tuple(speakers[700:1400])
    assert result.speakers[:700] == (2,) * 700
    warnings = [entry for entry in logs if entry["event"] == "speakers.chunk_failed"]
    assert len(warnings) == 1
    assert "w900" not in repr(warnings[0])


def _interrupted() -> BaseException:
    return KeyboardInterrupt()


def test_an_interrupt_still_stops_the_pass() -> None:
    words = _said((1, 1000), (2, 1000))
    backend = FakeSpeakerBackend(fail_when=_fails_on_w900, fail_with=_interrupted)

    with pytest.raises(KeyboardInterrupt):
        relabel([word.text for word in words], [word.speaker for word in words], backend)


def test_unattributed_words_are_never_offered_or_given_an_id() -> None:
    words = _said((3, 10), (None, 10), (8, 10))
    speakers = [word.speaker for word in words]
    # Every word, unattributed ones included, claimed by the first speaker.
    backend = FakeSpeakerBackend(reply=lambda target: "<spk:0> " + _TAGS.sub("", target))

    result = relabel([word.text for word in words], speakers, backend)

    system, user = backend.calls[0]
    assert "Speaker ids in this call: 0, 1." in system
    assert "<spk:?> marks words" in system
    assert target_of(user).splitlines()[1].startswith("<spk:?> w10 ")
    assert result.speakers == (3,) * 10 + (None,) * 10 + (3,) * 10
    assert result.relabeled == 10


def test_words_the_reply_leaves_unattributed_keep_their_speaker() -> None:
    texts = ["one", "two", "three", "four"]
    reply = parse_reply("<spk:1> one <spk:?> two three <spk:1> four")

    assert reply == [(1, "one"), (None, "two"), (None, "three"), (1, "four")]
    assert align_labels(texts, [0, 0, None, 0], reply, {0, 1}) == ([1, 0, None, 1], 4)


def test_one_speaker_beside_unattributed_words_makes_no_call() -> None:
    backend = FakeSpeakerBackend()

    result = relabel(["a", "b", "c."], [None, 4, None], backend)

    assert not needs_relabeling([None, 4, None])
    assert needs_relabeling([None, 4, 5])
    assert result.speakers == (None, 4, None)
    assert backend.calls == []


def test_render_tags_a_leading_unattributed_run() -> None:
    assert render(["Hi.", "Yes."], [None, 0]) == "<spk:?> Hi.\n<spk:0> Yes."


def _talk(*runs: tuple[int, str]) -> tuple[list[str], list[int | None]]:
    """(speaker, sentence) runs as per-word texts and speaker ids."""
    texts = [word for _, sentence in runs for word in sentence.split()]
    ids: list[int | None] = [speaker for speaker, sentence in runs for _ in sentence.split()]
    return texts, ids


@pytest.mark.parametrize(
    "reply",
    [
        # The model kept one copy of a line said twice; matched to the first,
        # its label would land on the other speaker's words.
        pytest.param("<spk:1> We ship it.", id="dropped-repeat"),
        pytest.param("<spk:0> We ship it.\n<spk:1> We ship it.", id="dropped-word"),
        pytest.param("<spk:0> We ship it.\n<spk:1> We really ship it. Okay.", id="added-word"),
        pytest.param("<spk:0> We ship it.\n<spk:1> Okay. We ship it.", id="reordered"),
        pytest.param("<spk:0> We ship it.\n<spk:1> We send it. Okay.", id="reworded"),
    ],
)
def test_a_reply_whose_words_differ_from_the_chunk_fails_it(reply: str) -> None:
    texts, speakers = _talk((5, "We ship it."), (6, "We ship it. Okay."))
    backend = FakeSpeakerBackend(reply=_replying(reply))

    with capture_logs() as logs:
        result = relabel(texts, speakers, backend)

    assert result.speakers == tuple(speakers)
    assert result.failed == (0,)
    assert result.chunks[0].reason == "words_changed"
    warnings = [entry for entry in logs if entry["event"] == "speakers.chunk_failed"]
    assert [(entry["log_level"], entry["reason"]) for entry in warnings] == [
        ("warning", "words_changed")
    ]


def test_words_before_the_first_tag_count_toward_the_word_check() -> None:
    # The tagged words alone match the chunk; the untagged copy in front is the
    # extra line that makes the swapped labels wrong.
    texts, speakers = _talk((5, "We ship it."), (6, "We ship it."))
    reply = "<out>We ship it. <spk:1> We ship it. <spk:0> We ship it.</out>"
    backend = FakeSpeakerBackend(reply=_replying(reply), wrap=False)

    result = relabel(texts, speakers, backend)

    assert result.speakers == tuple(speakers)
    assert result.failed == (0,)
    assert result.chunks[0].reason == "words_changed"


_PREFIX_WORDS = st.sampled_from(["w0", "w1", "w2", "junk"])


@given(
    count=st.integers(min_value=2, max_value=12),
    cut=st.integers(min_value=0, max_value=12),
    prefix=st.lists(_PREFIX_WORDS, max_size=6),
)
def test_an_untagged_prefix_passes_only_when_every_word_is_the_chunks(
    count: int, cut: int, prefix: list[str]
) -> None:
    texts = [f"w{index % 3}" for index in range(count)]
    half = count // 2
    speakers: list[int | None] = [5 if index < half else 6 for index in range(count)]
    cut = min(cut, count)
    reply = f"<out>{' '.join(prefix)} <spk:1> {' '.join(texts[cut:])}</out>"
    backend = FakeSpeakerBackend(reply=_replying(reply), wrap=False)

    result = relabel(texts, speakers, backend)

    assert (result.chunks[0].status == "ok") == (prefix == texts[:cut])
    if result.chunks[0].status == "ok":
        # Untagged words keep their labels; the tagged rest all goes to rank 1.
        assert result.speakers == tuple(speakers[:cut]) + (6,) * (count - cut)
    else:
        assert result.speakers == tuple(speakers)


def test_a_word_holding_a_space_takes_the_label_of_its_first_token() -> None:
    # The prompt joins raw word texts, so the model sees such a word as two.
    texts = ["We", "fly", "to", "New York", "", "today."]
    speakers: list[int | None] = [4, 4, 4, 4, 4, 9]
    backend = FakeSpeakerBackend(reply=_replying("<spk:1> We fly to New York today."))

    result = relabel(texts, speakers, backend)

    assert result.failed == ()
    assert result.speakers == (9, 9, 9, 9, 4, 9)
    assert result.chunks[0].reply_words == 6
    assert result.chunks[0].aligned_words == 6


@pytest.mark.parametrize(
    ("wrap", "stop_reason", "reason"),
    [
        pytest.param(False, "end_turn", "unclosed_out_block", id="no-closing-tag"),
        pytest.param(True, "max_tokens", "truncated_reply", id="max-tokens"),
    ],
)
def test_a_truncated_reply_fails_its_chunk(wrap: bool, stop_reason: str, reason: str) -> None:
    words = _said((1, 5), (2, 5))
    speakers = [word.speaker for word in words]

    def reply(target: str) -> str:
        swapped = _swap_ranks(target)
        return swapped if wrap else f"<out>\n{swapped}"

    backend = FakeSpeakerBackend(reply=reply, wrap=wrap, stop_reason=stop_reason)

    result = relabel([word.text for word in words], speakers, backend)

    assert result.speakers == tuple(speakers)
    assert result.failed == (0,)
    assert result.chunks[0].reason == reason
    assert result.chunks[0].stop_reason == stop_reason


_OLD_RULE = "- Reply with the corrected TARGET text between <out> and </out>, and nothing else.\n"
_NAMES_RULE = (
    "- Reply with the corrected TARGET text between <out> and </out>, "
    "then the <names> block described below, and nothing else.\n"
)
_NAMES_SECTION = """
People at this meeting: Alice, Bru{n}o.
Naming them does not change the rules for <spk:N> tags above.
After </out>, list each place in TARGET where one of these people is named, one line per \
place, between <names> and </names>:
NAME | SAID | KIND | QUOTE
- NAME: the person, written exactly as in the list above.
- SAID: the word or words in TARGET that name the person, copied as written there; speech \
recognition may have misspelled the name.
- KIND, from what the words show:
  next: the speaker addresses the person, who is expected to speak next;
  previous: the speaker addresses the person who spoke just before;
  self: the speaker names themself;
  about: any other mention.
- QUOTE: 4 to 12 consecutive words copied exactly from TARGET, including SAID.
List only people on the list, and only names said in TARGET, not in CONTEXT. When no one on \
the list is named, write <names></names>.
"""


def test_with_attendees_the_prompt_asks_for_a_names_block_after_the_out_block() -> None:
    texts = _texts(1000, ends={749})
    ids = [0] * 500 + [1] * 500
    plain = prompts(texts, ids, cut_points(texts))

    # A brace in a name reaches the prompt as written.
    asked = prompts(texts, ids, cut_points(texts), attendees=("Alice", "Bru{n}o"))

    assert len(asked) == len(plain) == 2
    for (system, user), (plain_system, plain_user) in zip(asked, plain, strict=True):
        assert user == plain_user
        assert _OLD_RULE not in system
        assert system == plain_system.replace(_OLD_RULE, _NAMES_RULE) + _NAMES_SECTION


def test_without_attendees_the_prompts_are_the_ones_the_pass_was_measured_with() -> None:
    texts = _texts(1000, ends=set(range(9, 1000, 10)))
    ids = [0] * 300 + [None] * 50 + [1] * 650

    asked = json.dumps(prompts(texts, ids, cut_points(texts)))

    # tpst-2's prompts for this input, byte for byte: a result is comparable
    # only to one made with the same prompt.
    assert hashlib.sha256(asked.encode()).hexdigest() == (
        "d537cbd6bb3e789c661a8e8fe6fe637885127ec9ee500f465ccee2cff0c5271a"
    )


def test_a_prompt_without_the_rule_the_names_request_replaces_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scribe.speakers._SYSTEM", "Ids: {ids}.{unattributed}\nReply in <out>.\n")

    assert prompts(["w0", "w1."], [0, 1], [(0, 2)])
    with pytest.raises(AppError, match="names"):
        prompts(["w0", "w1."], [0, 1], [(0, 2)], attendees=("Alice",))


_ATTENDEES = ("Alice", "Bruno")


def _naming_first_word(target: str) -> str:
    first = target.split()[1]
    return f"<names>\nAlice | {first} | next | {first} and | more\nnot a line\n\n</names>"


def _claims(result: Relabeling) -> list[tuple[int, int, int, str, str, str, str]]:
    return [
        (claim.chunk, claim.start, claim.end, claim.name, claim.said, claim.kind, claim.quote)
        for claim in result.claims
    ]


def test_each_names_line_comes_back_as_a_claim_with_its_chunks_span() -> None:
    words = _said((1, 1000), (2, 1000))
    backend = FakeSpeakerBackend(trailer=_naming_first_word)

    result = relabel(
        [word.text for word in words],
        [word.speaker for word in words],
        backend,
        attendees=_ATTENDEES,
    )

    assert "People at this meeting: Alice, Bruno." in backend.calls[0][0]
    # A line short of four fields leaves the missing ones empty; a quote may hold a `|`.
    assert _claims(result) == [
        (0, 0, 700, "Alice", "w0", "next", "w0 and | more"),
        (0, 0, 700, "not a line", "", "", ""),
        (1, 700, 1400, "Alice", "w700", "next", "w700 and | more"),
        (1, 700, 1400, "not a line", "", "", ""),
        (2, 1400, 2000, "Alice", "w1400", "next", "w1400 and | more"),
        (2, 1400, 2000, "not a line", "", "", ""),
    ]
    assert result.names_blocks_missing == ()


def _rewording_w900(target: str) -> str:
    return target.replace(" w900 ", " w9000 ")


def test_a_failed_chunks_claims_are_discarded() -> None:
    words = _said((1, 1000), (2, 1000))
    backend = FakeSpeakerBackend(reply=_rewording_w900, trailer=_naming_first_word)

    result = relabel(
        [word.text for word in words],
        [word.speaker for word in words],
        backend,
        attendees=_ATTENDEES,
    )

    assert result.failed == (1,)
    assert [claim.chunk for claim in result.claims] == [0, 0, 2, 2]
    assert result.names_blocks_missing == ()


def _block_in_the_first_chunk_only(target: str) -> str:
    first = target.split()[1]
    if first == "w0":
        return "<names></names>"
    if first == "w700":
        return "No one on the list is named."
    return f"<names>\nAlice | {first} | next | {first} and more"


def test_a_usable_reply_without_a_names_block_is_listed() -> None:
    words = _said((1, 1000), (2, 1000))
    backend = FakeSpeakerBackend(trailer=_block_in_the_first_chunk_only)

    result = relabel(
        [word.text for word in words],
        [word.speaker for word in words],
        backend,
        attendees=_ATTENDEES,
    )

    assert result.failed == ()
    assert result.claims == ()
    assert result.names_blocks_missing == (1, 2)


def _quoting_the_closing_tag(_target: str) -> str:
    return "<names>\nAlice | Alice | next | Alice, is </out> the closing tag?\n</names>"


def test_a_names_line_quoting_the_closing_tag_costs_no_correction() -> None:
    texts, speakers = _talk((5, "Alice, is </out> the closing tag?"), (6, "Yes, it ends a reply."))

    plain = relabel(texts, speakers, FakeSpeakerBackend(reply=_swap_ranks))
    named = relabel(
        texts,
        speakers,
        FakeSpeakerBackend(reply=_swap_ranks, trailer=_quoting_the_closing_tag),
        attendees=_ATTENDEES,
    )

    assert named.failed == ()
    assert named.speakers == plain.speakers != tuple(speakers)
    assert _claims(named) == [
        (0, 0, len(texts), "Alice", "Alice", "next", "Alice, is </out> the closing tag?")
    ]


def test_without_attendees_a_names_block_is_ignored() -> None:
    words = _said((1, 1000), (2, 1000))
    texts, speakers = [word.text for word in words], [word.speaker for word in words]

    plain = relabel(texts, speakers, FakeSpeakerBackend(reply=_swap_ranks))
    trailed = relabel(
        texts, speakers, FakeSpeakerBackend(reply=_swap_ranks, trailer=_naming_first_word)
    )

    assert trailed == plain
    assert (trailed.claims, trailed.names_blocks_missing) == ((), ())
