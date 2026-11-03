"""Timelines of who holds the microphone in, starts and stops out."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import pytest
from hypothesis import given
from hypothesis import strategies as st
from structlog.testing import capture_logs
from typer.testing import CliRunner

from scribe.call_watch import (
    CONFIRM_SECONDS,
    DEFAULT_APPS,
    GRACE_SECONDS,
    HIJACK_BUNDLE_ID,
    OUTAGE_SECONDS,
    READBACK_SECONDS,
    RESEND_SECONDS,
    Action,
    Idle,
    Marker,
    Note,
    Read,
    Recording,
    Starting,
    State,
    Stopping,
    Timings,
    VanishedProcessError,
    Verb,
    command_text,
    decide,
    input_holders,
    is_call_app,
    marker_path,
    needs_running,
    require_macos,
    restored_state,
    watch,
)
from scribe.cli import app
from scribe.errors import AppError, ExternalServiceError, UnsupportedPlatformError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

TIMINGS = Timings(
    confirm_seconds=3.0,
    grace_seconds=30.0,
    readback_seconds=5.0,
    resend_seconds=30.0,
    outage_seconds=30.0,
)
ZOOM = frozenset({"us.zoom.xos"})
CHROME = frozenset({"com.google.Chrome.helper"})
QUIET: frozenset[str] = frozenset()
HIJACK = frozenset({HIJACK_BUNDLE_ID})
NEVER_CALLS = (
    HIJACK_BUNDLE_ID,
    "ai.krisp.krispMac",
    "com.prakashjoshipax.VoiceInk",
    "com.p5sys.jump.mac.viewer",
)
REMINDER = "call_watch.recording_without_call"
TROUBLE = frozenset(
    {
        "call_watch.start_unconfirmed",
        "call_watch.stop_unconfirmed",
        "call_watch.recording_released",
        "call_watch.reads_failing",
    }
)

type Span = tuple[int, frozenset[str] | Read]


@dataclass
class Trace:
    actions: list[tuple[float, Action]] = field(default_factory=list[tuple[float, Action]])
    notes: list[tuple[float, Note]] = field(default_factory=list[tuple[float, Note]])
    state: State = field(default_factory=State)

    def verbs(self) -> list[Action]:
        return [action for _, action in self.actions]

    def events(self) -> list[str]:
        return [note.event for _, note in self.notes]

    def notified(self) -> list[str]:
        return [note.notify for _, note in self.notes if note.notify]

    def banners(self) -> list[tuple[float, str]]:
        return [(when, note.event) for when, note in self.notes if note.notify]


def replay(
    spans: Sequence[Span],
    *,
    state: State | None = None,
    answers: bool = True,
    hijack_running: bool = True,
    session_running: bool = False,
    start: float = 0.0,
    apps: Sequence[str] = DEFAULT_APPS,
    timings: Timings = TIMINGS,
) -> Trace:
    """Decide once a second over `spans`, with Audio Hijack answering the actions it is sent."""
    trace = Trace(state=state or State())
    now = start
    running = session_running
    for seconds, holding in spans:
        for _ in range(seconds):
            read = (
                holding if isinstance(holding, Read) else holding | (HIJACK if running else QUIET)
            )
            step = decide(
                trace.state, now, read, apps=apps, hijack_running=hijack_running, timings=timings
            )
            trace.state = step.state
            trace.notes.extend((now, note) for note in step.notes)
            if step.action is not None:
                trace.actions.append((now, step.action))
                if answers:
                    running = step.action in {"start", "resend-start"}
            now += 1.0
    return trace


def resume(state: State, spans: Sequence[Span], *, hijack_running: bool = True) -> Trace:
    """Replay on from a state an earlier replay left, later than any time it holds."""
    return replay(spans, state=state, start=10_000.0, hijack_running=hijack_running)


def test_the_named_constants_hold_the_measured_values() -> None:
    assert (CONFIRM_SECONDS, GRACE_SECONDS, READBACK_SECONDS, RESEND_SECONDS, OUTAGE_SECONDS) == (
        3.0,
        30.0,
        5.0,
        30.0,
        30.0,
    )
    assert Timings() == TIMINGS


def test_a_hold_shorter_than_the_confirmation_starts_nothing() -> None:
    assert replay([(3, ZOOM), (60, QUIET)]).verbs() == []


def test_a_hold_past_the_confirmation_starts_once() -> None:
    trace = replay([(120, ZOOM)])
    assert trace.actions == [(3.0, "start")]


def test_a_release_shorter_than_the_grace_stops_nothing() -> None:
    assert replay([(20, ZOOM), (10, QUIET), (20, ZOOM)]).verbs() == ["start"]


def test_a_release_past_the_grace_stops_once() -> None:
    trace = replay([(20, ZOOM), (120, QUIET)])
    assert trace.verbs() == ["start", "stop"]
    assert trace.actions[1][0] - 20.0 >= 30.0


@pytest.mark.parametrize("bundle", NEVER_CALLS)
def test_an_app_off_the_list_never_starts(bundle: str) -> None:
    assert replay([(120, frozenset({bundle}))]).verbs() == []


def test_an_app_off_the_list_keeps_nothing_alive() -> None:
    krisp = frozenset({"ai.krisp.krispMac"})
    assert replay([(20, ZOOM), (120, krisp)]).verbs() == ["start", "stop"]


@pytest.mark.parametrize(
    ("short", "gap"),
    [(6, 22), (3, 1), (4, 1)],
    ids=["six-then-gap", "three-ticks-observes-two-seconds", "observes-three-seconds"],
)
def test_the_real_openings_give_one_recording(short: int, gap: int) -> None:
    trace = replay([(short, CHROME), (gap, QUIET), (3000, CHROME), (60, QUIET)])
    assert trace.verbs() == ["start", "stop"]


def test_the_opening_that_confirms_nothing_starts_within_the_next_holds_confirmation() -> None:
    trace = replay([(3, CHROME), (1, QUIET), (100, CHROME)])
    assert trace.actions == [(7.0, "start")]


@pytest.mark.parametrize(
    "bundle",
    [
        "com.google.Chrome",
        "com.google.Chrome.helper",
        "COM.GOOGLE.CHROME.HELPER",
        "us.zoom.xos",
        "com.microsoft.teams2",
        "com.apple.avconferenced",
        "com.apple.TelephonyUtilities",
        # A plain prefix, because a boundary rule would lose the real com.microsoft.teams2.
        "com.google.ChromeX",
    ],
)
def test_a_bundle_id_on_the_list_matches(bundle: str) -> None:
    assert is_call_app(bundle, DEFAULT_APPS)


@pytest.mark.parametrize(
    "bundle", ["com.googlex.Chrome", "us.zoomer", "org.mozilla.firefox", "", *NEVER_CALLS]
)
def test_a_bundle_id_off_the_list_does_not_match(bundle: str) -> None:
    assert not is_call_app(bundle, DEFAULT_APPS)


def test_audio_hijack_never_counts_even_when_a_prefix_covers_it() -> None:
    assert not is_call_app(HIJACK_BUNDLE_ID, ("com.rogueamoeba.",))


def test_a_start_that_does_not_read_back_warns_once_and_resends() -> None:
    trace = replay([(100, ZOOM)], answers=False)
    assert trace.actions == [
        (3.0, "start"),
        (33.0, "resend-start"),
        (63.0, "resend-start"),
        (93.0, "resend-start"),
    ]
    assert trace.events().count("call_watch.start_unconfirmed") == 1
    assert [when for when, note in trace.notes if note.notify] == [8.0]


def test_a_start_is_resent_only_while_the_call_holds_input() -> None:
    trace = replay([(10, ZOOM), (25, QUIET), (2, ZOOM)], answers=False)
    assert trace.actions == [(3.0, "start"), (35.0, "resend-start")]


def test_a_late_confirmation_ends_the_resending() -> None:
    silent = replay([(40, ZOOM)], answers=False)
    assert silent.verbs() == ["start", "resend-start"]
    step = decide(silent.state, 40.0, ZOOM | HIJACK, timings=TIMINGS)
    assert isinstance(step.state.session, Recording)
    state = step.state
    for second in range(41, 200):
        step = decide(state, float(second), ZOOM | HIJACK, timings=TIMINGS)
        state = step.state
        assert step.action is None
        assert not step.notes


def test_a_second_call_warns_again_after_the_first_ended() -> None:
    trace = replay([(20, ZOOM), (40, QUIET), (20, ZOOM)], answers=False)
    assert len(trace.notified()) >= 2
    assert trace.events().count("call_watch.start_unconfirmed") == 2


def test_a_stop_that_does_not_read_back_warns_once_and_resends() -> None:
    state = replay([(10, ZOOM | HIJACK), (31, HIJACK)]).state
    assert isinstance(state.session, Stopping)
    trace = resume(state, [(100, HIJACK)])
    assert trace.verbs() == ["stop", "resend-stop", "resend-stop", "resend-stop"]
    assert len(trace.notified()) == 1
    assert trace.events().count("call_watch.stop_unconfirmed") == 1


def test_a_late_stop_confirmation_ends_the_resending_and_clears_the_marker() -> None:
    state = replay([(10, ZOOM | HIJACK), (31, HIJACK)]).state
    trace = resume(state, [(40, HIJACK), (60, QUIET)])
    assert trace.verbs() == ["stop", "resend-stop"]
    assert trace.events().count("call_watch.stop_done") == 1
    assert trace.state.marker is None


def against_a_slow_stop(
    state: State, spans: Sequence[tuple[int, frozenset[str]]], *, lands_after: float
) -> Trace:
    """Decide once a second from a running session that a stop ends `lands_after` seconds
    after it is sent. A start lands by the next poll and changes nothing while one runs."""
    trace = Trace(state=state)
    running = True
    landings: list[tuple[float, bool]] = []
    now = 10_000.0
    for seconds, holding in spans:
        for _ in range(seconds):
            for _when, after in sorted(landing for landing in landings if landing[0] <= now):
                running = after
            landings = [landing for landing in landings if landing[0] > now]
            read = holding | (HIJACK if running else QUIET)
            step = decide(trace.state, now, read, timings=TIMINGS)
            trace.state = step.state
            trace.notes.extend((now, note) for note in step.notes)
            if step.action is not None:
                trace.actions.append((now, step.action))
                starts = step.action in {"start", "resend-start"}
                landings.append((now + (0.25 if starts else lands_after), starts))
            now += 1.0
    return trace


def test_a_call_confirmed_while_a_stop_lands_starts_after_it_lands() -> None:
    # The stop goes out at 10 000 s, and the read at 10 004 s is the last to show it running.
    state = replay([(10, ZOOM), (31, QUIET)]).state
    trace = against_a_slow_stop(state, [(1, QUIET), (60, ZOOM)], lands_after=4.5)
    assert trace.verbs() == ["stop", "start"]
    assert trace.actions[1][0] > 10_004.5
    assert isinstance(trace.state.session, Recording)
    assert trace.state.session.holding
    assert "call_watch.start_confirmed" in trace.events()
    assert "call_watch.recording_released" not in trace.events()


def test_a_call_confirmed_while_another_session_holds_input_starts_after_the_window() -> None:
    state = replay([(10, ZOOM | HIJACK), (31, HIJACK)]).state
    trace = resume(state, [(1, HIJACK), (100, ZOOM | HIJACK)])
    assert trace.actions == [(10_000.0, "stop"), (10_005.0, "start")]
    assert isinstance(trace.state.session, Recording)
    assert trace.state.marker == "confirmed"


def test_a_hold_too_short_to_confirm_does_not_interrupt_a_stop_being_resent() -> None:
    state = replay([(10, ZOOM | HIJACK), (31, HIJACK)]).state
    trace = resume(state, [(29, HIJACK), (2, ZOOM | HIJACK), (5, HIJACK)])
    assert trace.verbs() == ["stop", "resend-stop"]


def test_a_call_back_on_the_poll_after_the_grace_keeps_the_recording() -> None:
    # The grace ends on the poll at 40 s, and the call is back on the poll at 41 s.
    returned = replay([(10, ZOOM), (31, QUIET), (1, ZOOM)])
    assert returned.actions == [(3.0, "start")]
    assert isinstance(returned.state.session, Recording)
    trace = replay([(10, ZOOM), (31, QUIET), (20, ZOOM), (40, QUIET)])
    assert trace.actions == [(3.0, "start"), (92.0, "stop")]
    assert "call_watch.recording_released" not in trace.events()
    assert trace.state.marker is None


def test_a_call_back_on_the_poll_after_the_grace_resumes_an_unread_start() -> None:
    trace = replay([(10, ZOOM), (31, QUIET), (40, ZOOM)], answers=False)
    assert trace.actions == [(3.0, "start"), (71.0, "resend-start")]
    assert isinstance(trace.state.session, Starting)


def test_a_release_during_a_call_notifies_once_and_starts_nothing() -> None:
    state = replay([(10, ZOOM)]).state
    assert isinstance(state.session, Recording)
    trace = resume(state, [(100, ZOOM)])
    assert trace.verbs() == []
    assert trace.events() == ["call_watch.recording_released"]
    assert len(trace.notified()) == 1


def test_after_a_release_the_next_start_needs_the_call_to_end_and_return() -> None:
    state = replay([(10, ZOOM)]).state
    trace = resume(state, [(50, ZOOM), (45, QUIET), (10, ZOOM)])
    assert trace.verbs() == ["stop", "start"]


def test_a_release_as_the_call_ends_is_not_called_a_lost_recording() -> None:
    state = replay([(10, ZOOM)]).state
    trace = resume(state, [(10, QUIET)])
    assert "call_watch.recording_released" not in trace.events()


def test_each_release_of_a_call_notifies() -> None:
    state = replay([(10, ZOOM)]).state
    trace = resume(state, [(5, ZOOM), (5, ZOOM | HIJACK), (5, ZOOM)])
    assert trace.events() == [
        "call_watch.recording_released",
        "call_watch.recording_resumed",
        "call_watch.recording_released",
    ]


def test_a_stop_is_not_sent_when_audio_hijack_is_not_running() -> None:
    state = replay([(10, ZOOM)]).state
    trace = resume(state, [(50, QUIET)], hijack_running=False)
    assert trace.verbs() == []
    assert trace.events() == ["call_watch.stop_not_needed"]
    assert trace.state.marker is None


def test_the_running_check_is_asked_only_before_a_first_stop_with_input_released() -> None:
    assert not needs_running(State(), QUIET)
    assert not needs_running(State(session=Recording()), QUIET)
    pending = State(session=Stopping(marker="confirmed"))
    assert needs_running(pending, QUIET)
    assert not needs_running(pending, HIJACK)
    assert not needs_running(pending, Read.FAILED)
    assert not needs_running(State(session=Stopping("stopping", 1.0, 1.0)), QUIET)
    assert not needs_running(restored_state("stopping"), QUIET)


def test_a_short_outage_stays_quiet() -> None:
    state = replay([(10, ZOOM)]).state
    trace = resume(state, [(29, Read.FAILED), (5, ZOOM | HIJACK)])
    assert trace.verbs() == []
    assert trace.notes == []


def test_a_long_outage_notifies_once_and_its_end_is_logged() -> None:
    state = replay([(10, ZOOM)]).state
    trace = resume(state, [(200, Read.FAILED), (5, ZOOM | HIJACK)])
    assert trace.verbs() == []
    assert trace.events() == ["call_watch.reads_failing", "call_watch.reads_recovered"]
    assert len(trace.notified()) == 1


def test_each_long_outage_notifies() -> None:
    trace = replay([(40, Read.FAILED), (1, QUIET), (40, Read.FAILED), (1, QUIET)])
    assert len(trace.notified()) == 2


def test_an_outage_is_not_a_release() -> None:
    assert replay([(20, ZOOM), (120, Read.FAILED), (20, ZOOM)]).verbs() == ["start"]


def test_an_outage_is_not_a_hold_and_starts_nothing() -> None:
    assert replay([(120, Read.FAILED)]).verbs() == []


def test_time_in_an_outage_does_not_count_toward_the_grace() -> None:
    trace = replay([(20, ZOOM), (5, QUIET), (120, Read.FAILED), (5, QUIET)])
    assert trace.verbs() == ["start"]


def test_time_in_an_outage_does_not_count_toward_the_confirmation() -> None:
    assert replay([(2, ZOOM), (120, Read.FAILED)]).verbs() == []
    assert replay([(2, ZOOM), (20, Read.FAILED), (2, ZOOM)]).actions == [(23.0, "start")]


class TestRestart:
    def test_sent_and_a_call_resumes_the_read_back_without_sending(self) -> None:
        trace = replay([(120, ZOOM)], state=restored_state("sent"), answers=False)
        assert trace.actions == [
            (30.0, "resend-start"),
            (60.0, "resend-start"),
            (90.0, "resend-start"),
        ]
        assert [when for when, note in trace.notes if note.notify] == [5.0]

    def test_sent_and_a_call_is_confirmed_by_the_read_back(self) -> None:
        trace = replay([(5, ZOOM | HIJACK)], state=restored_state("sent"))
        assert trace.verbs() == []
        assert isinstance(trace.state.session, Recording)

    def test_confirmed_and_a_call_sends_nothing(self) -> None:
        trace = replay([(120, ZOOM | HIJACK)], state=restored_state("confirmed"))
        assert trace.verbs() == []
        assert trace.notes == []

    def test_confirmed_and_a_call_leaves_a_persons_stop_standing(self) -> None:
        trace = replay([(120, ZOOM)], state=restored_state("confirmed"))
        assert trace.verbs() == []
        assert trace.events() == ["call_watch.recording_released"]

    @pytest.mark.parametrize("marker", ["sent", "confirmed"])
    def test_a_marker_and_no_call_runs_the_grace_from_zero_then_stops(self, marker: Marker) -> None:
        trace = replay([(70, HIJACK)], state=restored_state(marker))
        assert trace.verbs() == ["stop", "resend-stop"]
        assert trace.actions[0][0] >= 30.0
        assert "call_watch.start_unconfirmed" not in trace.events()

    def test_sent_and_no_call_warns_of_nothing(self) -> None:
        trace = replay([(60, QUIET)], state=restored_state("sent"))
        assert trace.verbs() == ["stop"]
        assert trace.notified() == []

    def test_a_call_returning_in_the_restarted_grace_keeps_the_session(self) -> None:
        trace = replay([(10, HIJACK), (60, ZOOM | HIJACK)], state=restored_state("confirmed"))
        assert trace.verbs() == []

    def test_no_marker_and_no_call_sends_nothing(self) -> None:
        trace = replay([(120, HIJACK)], state=restored_state(None))
        assert trace.verbs() == []
        assert trace.notes == []

    def test_no_marker_and_a_call_runs_the_confirmation(self) -> None:
        trace = replay([(10, ZOOM)], state=restored_state(None))
        assert trace.actions == [(3.0, "start")]

    def test_a_failed_first_read_decides_nothing(self) -> None:
        trace = replay([(5, Read.FAILED)], state=restored_state("sent"))
        assert trace.verbs() == []
        assert trace.state.marker == "sent"


class TestBanners:
    def test_a_call_shows_a_start_on_its_read_back_and_a_stop_on_its_read_back(self) -> None:
        trace = replay([(20, CHROME), (120, QUIET)])
        assert trace.actions == [(3.0, "start"), (51.0, "stop")]
        assert trace.banners() == [
            (4.0, "call_watch.start_confirmed"),
            (52.0, "call_watch.stop_done"),
        ]
        start, stop = trace.notified()
        assert "Recording started" in start
        assert "Google Chrome" in start
        assert "com.google" not in start
        assert "stopped" in stop
        assert "transcript" not in stop

    @pytest.mark.parametrize(
        ("bundle", "name"),
        [
            ("com.google.Chrome.helper", "Google Chrome"),
            ("us.zoom.xos", "Zoom"),
            ("com.microsoft.teams2", "Microsoft Teams"),
            ("com.apple.avconferenced", "FaceTime"),
            ("com.apple.TelephonyUtilities", "Phone"),
        ],
    )
    def test_each_default_app_is_named_by_its_display_name(self, bundle: str, name: str) -> None:
        trace = replay([(10, frozenset({bundle}))])
        assert trace.banners() == [(4.0, "call_watch.start_confirmed")]
        assert name in trace.notified()[0]
        assert bundle not in trace.notified()[0]

    def test_every_default_prefix_has_a_display_name(self) -> None:
        for prefix in DEFAULT_APPS:
            trace = replay([(10, frozenset({prefix + "x"}))])
            assert trace.banners() == [(4.0, "call_watch.start_confirmed")]
            assert prefix not in trace.notified()[0]

    def test_each_app_holding_input_is_named_once(self) -> None:
        holders = frozenset({"com.google.Chrome.helper", "com.google.Chrome.helper.audio"})
        trace = replay([(10, holders | ZOOM)])
        assert trace.banners() == [(4.0, "call_watch.start_confirmed")]
        text = trace.notified()[0]
        assert text.count("Google Chrome") == 1
        assert "Zoom" in text

    def test_two_apps_holding_input_are_both_named_in_bundle_id_order(self) -> None:
        trace = replay([(10, CHROME | ZOOM)])
        assert trace.banners() == [(4.0, "call_watch.start_confirmed")]
        assert trace.notified()[0] == "Recording started for the call in Google Chrome and Zoom."

    def test_an_app_prefix_outside_the_defaults_is_named_by_its_bundle_id(self) -> None:
        trace = replay([(10, frozenset({"org.example.Talk"}))], apps=("org.example.",))
        assert trace.banners() == [(4.0, "call_watch.start_confirmed")]
        assert "org.example.Talk" in trace.notified()[0]

    def test_a_default_app_given_as_a_prefix_keeps_its_display_name(self) -> None:
        trace = replay([(10, ZOOM)], apps=("us.zoom.",))
        assert trace.banners() == [(4.0, "call_watch.start_confirmed")]
        assert "Zoom" in trace.notified()[0]
        assert "us.zoom" not in trace.notified()[0]

    def test_a_start_read_back_with_no_call_app_holding_names_none(self) -> None:
        trace = replay([(5, HIJACK)], state=restored_state("sent"))
        assert trace.banners() == [(0.0, "call_watch.start_confirmed")]
        text = trace.notified()[0]
        assert "Recording started" in text
        assert not any(name in text for name in ("Google Chrome", "Zoom", "Microsoft", "Phone"))

    def test_a_hold_that_started_by_hand_shows_one_start_and_one_stop(self) -> None:
        trace = replay([(10, QUIET), (30, ZOOM), (60, QUIET)], session_running=True)
        assert trace.verbs() == ["start", "stop"]
        assert [event for _, event in trace.banners()] == [
            "call_watch.start_confirmed",
            "call_watch.stop_done",
        ]

    def test_a_start_not_read_back_shows_no_start(self) -> None:
        trace = replay([(100, ZOOM)], answers=False)
        assert [event for _, event in trace.banners()] == ["call_watch.start_unconfirmed"]

    def test_a_release_and_resume_show_no_start_or_stop(self) -> None:
        state = replay([(10, ZOOM)]).state
        trace = resume(state, [(5, ZOOM), (5, ZOOM | HIJACK), (5, ZOOM), (5, ZOOM | HIJACK)])
        assert {event for _, event in trace.banners()} == {"call_watch.recording_released"}

    def test_a_stop_not_needed_shows_no_stop(self) -> None:
        state = replay([(10, ZOOM)]).state
        trace = resume(state, [(50, QUIET)], hijack_running=False)
        assert trace.events() == ["call_watch.stop_not_needed"]
        assert trace.banners() == []

    def test_a_stop_that_does_not_read_back_shows_no_stop(self) -> None:
        state = replay([(10, ZOOM | HIJACK), (31, HIJACK)]).state
        trace = resume(state, [(100, HIJACK)])
        assert [event for _, event in trace.banners()] == ["call_watch.stop_unconfirmed"]

    def test_a_stop_from_an_unread_start_shows_no_stop(self) -> None:
        trace = replay([(10, ZOOM), (60, QUIET)], answers=False)
        assert trace.verbs() == ["start", "stop"]
        assert "call_watch.stop_done" in trace.events()
        assert [event for _, event in trace.banners()] == ["call_watch.start_unconfirmed"]

    def test_a_restart_restoring_stopping_shows_no_stop(self) -> None:
        trace = replay([(5, QUIET)], state=restored_state("stopping"))
        assert trace.events() == ["call_watch.stop_done"]
        assert trace.banners() == []

    def test_a_restart_restoring_confirmed_shows_only_the_stop(self) -> None:
        trace = replay(
            [(10, ZOOM), (60, QUIET)], state=restored_state("confirmed"), session_running=True
        )
        assert trace.verbs() == ["stop"]
        assert [event for _, event in trace.banners()] == ["call_watch.stop_done"]

    def test_a_restart_restoring_sent_then_a_read_back_shows_one_start(self) -> None:
        trace = replay([(60, ZOOM | HIJACK)], state=restored_state("sent"))
        assert trace.verbs() == []
        assert trace.banners() == [(0.0, "call_watch.start_confirmed")]
        assert "Zoom" in trace.notified()[0]

    def test_a_start_read_back_after_the_grace_ran_out_shows_one_start_then_the_stop(
        self,
    ) -> None:
        # The call lets go before the read-back and is back, with Audio Hijack holding, on
        # the poll after the grace runs out.
        trace = replay(
            [(4, ZOOM), (31, QUIET), (10, ZOOM | HIJACK), (32, HIJACK), (5, QUIET)],
            answers=False,
        )
        assert trace.verbs() == ["start", "stop"]
        assert trace.banners() == [
            (8.0, "call_watch.start_unconfirmed"),
            (35.0, "call_watch.start_confirmed"),
            (77.0, "call_watch.stop_done"),
        ]
        assert "Zoom" in trace.notified()[1]


class TestReminder:
    def test_audio_hijack_alone_is_reminded_about_once_and_never_stopped(self) -> None:
        trace = replay([(5000, HIJACK)])
        assert trace.actions == []
        assert trace.banners() == [(1800.0, REMINDER)]
        note = trace.notes[0][1]
        assert note.level == "warning"
        assert note.notify == (
            "Audio Hijack has held the microphone for 30 minutes with no call detected. "
            "Press Stop in Audio Hijack if you do not need this recording."
        )
        assert isinstance(trace.state.session, Idle)

    def test_a_short_call_hold_ends_the_stretch(self) -> None:
        trace = replay([(1000, HIJACK), (2, ZOOM | HIJACK), (3000, HIJACK)])
        assert trace.actions == []
        assert trace.banners() == [(2802.0, REMINDER)]

    def test_audio_hijack_letting_go_ends_the_stretch(self) -> None:
        trace = replay([(1000, HIJACK), (1, QUIET), (3000, HIJACK)])
        assert trace.banners() == [(2801.0, REMINDER)]

    def test_a_read_outage_moves_the_reminder_later_by_its_length(self) -> None:
        trace = replay([(1000, HIJACK), (60, Read.FAILED), (3000, HIJACK)])
        assert trace.actions == []
        assert [(when, event) for when, event in trace.banners() if event == REMINDER] == [
            (1860.0, REMINDER)
        ]

    def test_a_shorter_reminder_time_reminds_sooner(self) -> None:
        trace = replay([(200, HIJACK)], timings=replace(TIMINGS, reminder_seconds=120.0))
        assert trace.banners() == [(120.0, REMINDER)]
        assert "2 minutes" in trace.notified()[0]

    def test_a_reminder_never_comes_while_recording_a_call(self) -> None:
        trace = replay([(5000, ZOOM)])
        assert isinstance(trace.state.session, Recording)
        assert REMINDER not in trace.events()

    def test_a_reminder_never_comes_while_a_start_waits(self) -> None:
        trace = replay([(5000, ZOOM)], answers=False)
        assert isinstance(trace.state.session, Starting)
        assert REMINDER not in trace.events()

    def test_a_reminder_never_comes_while_a_stop_waits(self) -> None:
        state = replay([(10, ZOOM | HIJACK), (31, HIJACK)]).state
        trace = resume(state, [(5000, HIJACK)])
        assert isinstance(trace.state.session, Stopping)
        assert REMINDER not in trace.events()

    def test_a_stretch_begins_only_once_the_session_is_idle(self) -> None:
        state = replay([(10, ZOOM | HIJACK), (31, HIJACK)]).state
        trace = resume(state, [(10, HIJACK), (5, QUIET), (2000, HIJACK)])
        assert trace.actions == [(10_000.0, "stop")]
        assert trace.banners() == [
            (10_005.0, "call_watch.stop_unconfirmed"),
            (10_010.0, "call_watch.stop_done"),
            (11_815.0, REMINDER),
        ]


# A reading of each kind the watcher can see, and the gaps polls can come at.
_reads = st.sampled_from([QUIET, ZOOM, CHROME, HIJACK, ZOOM | HIJACK, HIJACK | CHROME, Read.FAILED])
_timelines = st.lists(
    st.tuples(st.floats(0.1, 45.0), _reads, st.booleans()),
    max_size=120,
)


@given(_timelines)
def test_starts_and_stops_pair(timeline: list[tuple[float, frozenset[str] | Read, bool]]) -> None:
    state = State()
    now = 0.0
    start_open = False
    confirmed = False
    for gap, read, running in timeline:
        now += gap
        was = state.session
        step = decide(state, now, read, hijack_running=running, timings=TIMINGS)
        state = step.state
        events = {note.event for note in step.notes}
        if read is Read.FAILED:
            assert step.action is None
            assert state.session == was
        if step.action == "start":
            assert not confirmed, "a start after a confirmed start with no stop between"
            assert not isinstance(was, Starting), "a repeat start is a resend"
            start_open = True
        elif step.action == "resend-start":
            assert isinstance(was, Starting), "a resend of a start not awaiting confirmation"
            assert not confirmed
        elif step.action == "stop":
            assert start_open, "a stop with no start before it"
            start_open = False
            confirmed = False
        elif step.action == "resend-stop":
            assert isinstance(was, Stopping), "a resend of a stop not awaiting confirmation"
        if "call_watch.stop_not_needed" in events:
            start_open = False
            confirmed = False
        if isinstance(state.session, Recording):
            confirmed = True


type Poll = tuple[float, Action | None, Marker | None, type[object], tuple[Note, ...]]


def _run_timeline(
    timeline: list[tuple[float, frozenset[str] | Read, bool]], timings: Timings
) -> list[Poll]:
    state = State()
    now = 0.0
    polls: list[Poll] = []
    for gap, read, running in timeline:
        now += gap
        step = decide(state, now, read, hijack_running=running, timings=timings)
        state = step.state
        polls.append((now, step.action, state.marker, type(state.session), step.notes))
    return polls


# The shorter times are drawn too: generated timelines almost never reach the default.
@given(_timelines, st.sampled_from([TIMINGS.reminder_seconds, 30.0, 0.0]))
def test_the_reminder_changes_only_its_own_note(
    timeline: list[tuple[float, frozenset[str] | Read, bool]], reminder_seconds: float
) -> None:
    reminded = _run_timeline(timeline, replace(TIMINGS, reminder_seconds=reminder_seconds))
    never = _run_timeline(timeline, replace(TIMINGS, reminder_seconds=float("inf")))
    assert [poll[:4] for poll in reminded] == [poll[:4] for poll in never]
    for with_reminder, without in zip(reminded, never, strict=True):
        assert tuple(note for note in with_reminder[4] if note.event != REMINDER) == without[4]
    assert all(note.event != REMINDER for poll in never for note in poll[4])


@given(_timelines, st.sampled_from([TIMINGS.reminder_seconds, 0.0]))
def test_banners_pair_with_read_backs(
    timeline: list[tuple[float, frozenset[str] | Read, bool]], reminder_seconds: float
) -> None:
    timings = replace(TIMINGS, reminder_seconds=reminder_seconds)
    state = State()
    now = 0.0
    recording = False
    reminded = False
    for gap, read, running in timeline:
        now += gap
        was = state.session
        step = decide(state, now, read, hijack_running=running, timings=timings)
        state = step.state
        if read is not Read.FAILED and read != HIJACK:
            reminded = False
        for note in step.notes:
            if not note.notify:
                continue
            if note.event == "call_watch.start_confirmed":
                # A start whose call let go before its read-back is read back from there.
                assert isinstance(was, Starting) or (
                    isinstance(was, Stopping) and was.marker == "sent"
                )
                assert isinstance(state.session, Recording)
                assert not recording
                recording = True
            elif note.event == "call_watch.stop_done":
                assert isinstance(was, Stopping)
                assert isinstance(state.session, Idle)
                assert recording, "a stop banner with no start banner before it"
                recording = False
            elif note.event == REMINDER:
                assert isinstance(was, Idle)
                assert isinstance(state.session, Idle)
                assert step.action is None
                assert read is not Read.FAILED
                assert not any(is_call_app(bundle, DEFAULT_APPS) for bundle in read)
                assert not reminded, "a second reminder in one stretch"
                reminded = True
        if not isinstance(state.session, (Recording, Stopping)):
            recording = False
        if not isinstance(state.session, Idle):
            reminded = False


@given(st.lists(st.tuples(st.floats(0.1, 45.0), st.sampled_from(NEVER_CALLS)), max_size=60))
def test_no_app_off_the_list_ever_acts(timeline: list[tuple[float, str]]) -> None:
    state = State()
    now = 0.0
    for gap, bundle in timeline:
        now += gap
        step = decide(state, now, frozenset({bundle}), timings=TIMINGS)
        state = step.state
        assert step.action is None


@given(st.text())
def test_a_session_name_stays_a_single_valid_string(name: str) -> None:
    forms: tuple[tuple[Verb, str], ...] = (("start", ").start()"), ("stop", ").stop()"))
    for verb, tail in forms:
        text = command_text(name, verb)
        assert text.startswith("app.sessionWithName(")
        assert text.endswith(tail)
        assert "\n" not in text
        assert json.loads(text.removeprefix("app.sessionWithName(").removesuffix(tail)) == name


@pytest.mark.parametrize("name", ['Call "A"', "back\\slash", "a\nb", "é\u2028"])
def test_a_hard_session_name_is_escaped(name: str) -> None:
    text = command_text(name, "start")
    assert text.isascii()
    assert '"' + json.dumps(name)[1:-1] + '"' in text


class FakeObjects:
    def __init__(self, table: Mapping[int, tuple[str, bool] | None]) -> None:
        self.table = table

    def object_ids(self) -> Sequence[int]:
        return list(self.table)

    def _entry(self, obj: int) -> tuple[str, bool]:
        entry = self.table[obj]
        if entry is None:
            raise VanishedProcessError
        return entry

    def holds_input(self, obj: int) -> bool:
        return self._entry(obj)[1]

    def bundle_id(self, obj: int) -> str:
        return self._entry(obj)[0]


def test_the_walk_returns_only_processes_holding_input() -> None:
    objects = FakeObjects({1: ("us.zoom.xos", True), 2: ("com.apple.Music", False), 3: ("", True)})
    assert input_holders(objects) == frozenset({"us.zoom.xos"})


def test_a_process_object_that_vanishes_is_skipped_not_a_failed_read() -> None:
    objects = FakeObjects({1: None, 2: ("us.zoom.xos", True)})
    assert input_holders(objects) == frozenset({"us.zoom.xos"})


def test_a_process_object_that_vanishes_between_its_two_reads_is_skipped() -> None:
    class Halfway(FakeObjects):
        @override
        def bundle_id(self, obj: int) -> str:
            raise VanishedProcessError

    assert input_holders(Halfway({1: ("us.zoom.xos", True)})) == QUIET


def test_a_failing_read_raises_rather_than_returning_an_empty_set() -> None:
    class Broken(FakeObjects):
        @override
        def object_ids(self) -> Sequence[int]:
            raise ExternalServiceError("CoreAudio status 1")

    with pytest.raises(AppError):
        input_holders(Broken({}))


class _EndError(Exception):
    pass


OPEN_PREFIX = ("/usr/bin/open", "-g", "-b", "com.rogueamoeba.audiohijack")


@dataclass
class FakeMac:
    """A clock, a reader and an argv runner, with Audio Hijack answering the files it is sent."""

    directory: Path
    call: Callable[[float], frozenset[str] | None]
    end: float
    now: float = 1000.0
    began: float = 1000.0
    session_running: bool = False
    app_running: bool = True
    answers: bool = True
    stop_ignored: bool = False
    open_error: Exception | None = None
    open_error_until: float = float("inf")
    check_error: Exception | None = None
    notify_error: Exception | None = None
    marker_seen_at_open: list[str | None] = field(default_factory=list[str | None])
    opens: list[tuple[float, str, str]] = field(default_factory=list[tuple[float, str, str]])
    notifications: list[str] = field(default_factory=list[str])
    notify_argv: list[list[str]] = field(default_factory=list[list[str]])
    running_checks: int = 0

    @property
    def marker_path(self) -> Path:
        return marker_path(self.directory, "Call")

    def elapsed(self) -> float:
        return self.now - self.began

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.elapsed() >= self.end:
            raise _EndError

    def reader(self) -> frozenset[str]:
        held = self.call(self.elapsed())
        if held is None:
            raise ExternalServiceError("CoreAudio status 1")
        return held | (HIJACK if self.session_running else QUIET)

    def runner(self, argv: Sequence[str], timeout: float) -> str:
        assert timeout > 0
        program = argv[0]
        if program == "/usr/bin/open":
            return self._open(argv)
        if program == "/usr/bin/osascript":
            if self.notify_error is not None:
                raise self.notify_error
            self.notifications.append(argv[-1])
            self.notify_argv.append(list(argv))
            return ""
        assert program == "/usr/bin/lsappinfo"
        self.running_checks += 1
        if self.check_error is not None:
            raise self.check_error
        return 'ASN:0x0-0x1-"Audio Hijack":\n' if self.app_running else "\n"

    def _open(self, argv: Sequence[str]) -> str:
        assert tuple(argv[:4]) == OPEN_PREFIX
        path = Path(argv[4])
        assert path.parent == self.directory
        self.marker_seen_at_open.append(
            self.marker_path.read_text() if self.marker_path.is_file() else None
        )
        if self.open_error is not None and self.elapsed() < self.open_error_until:
            raise self.open_error
        text = path.read_text()
        self.opens.append((self.elapsed(), path.name, text))
        if self.answers:
            starts = text.strip().endswith(".start()")
            self.session_running = starts or self.stop_ignored
        return ""

    def run(self, session: str = "Call") -> list[dict[str, Any]]:
        with capture_logs() as logs, pytest.raises(_EndError):
            watch(
                self.directory,
                session=session,
                apps=DEFAULT_APPS,
                reader=self.reader,
                runner=self.runner,
                clock=self.clock,
                sleep=self.sleep,
                poll_seconds=1.0,
                timings=TIMINGS,
            )
        return cast("list[dict[str, Any]]", logs)

    def verbs(self) -> list[str]:
        return [verb for _, verb in self.sent()]

    def sent(self) -> list[tuple[float, str]]:
        return [
            (when, name.removesuffix(".ahcommand").rpartition(".")[2])
            for when, name, _ in self.opens
        ]


def call_between(
    start: float, stop: float, holders: frozenset[str] = ZOOM
) -> Callable[[float], frozenset[str]]:
    return lambda elapsed: holders if start <= elapsed < stop else QUIET


def trouble(logs: Sequence[Mapping[str, Any]]) -> list[str]:
    """The events of the notes that notify of trouble, in order."""
    return [event for log in logs if (event := cast("str", log["event"])) in TROUBLE]


def test_the_loop_runs_a_call_from_start_to_stop(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 60.0), end=120.0)
    logs = mac.run()
    assert mac.verbs() == ["start", "stop"]
    assert mac.opens[0][0] == 3.0
    assert mac.opens[0][2] == 'app.sessionWithName("Call").start()\n'
    assert mac.opens[1][2] == 'app.sessionWithName("Call").stop()\n'
    assert not mac.marker_path.exists()
    assert trouble(logs) == []
    assert len(mac.notifications) == 2
    assert "Recording started" in mac.notifications[0]
    assert "Zoom" in mac.notifications[0]
    assert "stopped" in mac.notifications[1]
    assert mac.running_checks == 0
    assert [log["event"] for log in logs if log["log_level"] == "error"] == []


def test_the_loop_names_chrome_in_the_start_notification(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 60.0, CHROME), end=120.0)
    logs = mac.run()
    assert mac.verbs() == ["start", "stop"]
    assert len(mac.notifications) == 2
    assert "Google Chrome" in mac.notifications[0]
    assert "com.google.Chrome.helper" not in mac.notifications[0]
    assert "stopped" in mac.notifications[1]
    started = next(log for log in logs if log["event"] == "call_watch.start_confirmed")
    assert started["apps"] == ["com.google.Chrome.helper"]


def test_the_loop_reminds_once_about_audio_hijack_alone_and_sends_nothing(
    tmp_path: Path,
) -> None:
    mac = FakeMac(tmp_path, lambda _: QUIET, end=5000.0, session_running=True)
    logs = mac.run()
    assert mac.opens == []
    assert len(mac.notifications) == 1
    assert "30 minutes" in mac.notifications[0]
    reminded = [log for log in logs if log["event"] == REMINDER]
    assert [log["log_level"] for log in reminded] == ["warning"]
    assert not mac.marker_path.exists()


def test_the_marker_is_written_before_open_runs_and_confirmed_after(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=20.0)
    mac.run()
    assert mac.marker_seen_at_open == ["sent"]
    assert mac.marker_path.read_text() == "confirmed"


def test_the_marker_is_cleared_only_when_the_stop_is_done(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 10.0), end=60.0, stop_ignored=True)
    mac.run()
    assert mac.verbs()[:2] == ["start", "stop"]
    assert mac.marker_path.read_text() == "stopping"


def test_a_session_name_with_a_quote_reaches_the_file_as_one_string(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=10.0)
    mac.run(session='My "big" \\ call')
    text = mac.opens[0][2]
    assert text == 'app.sessionWithName("My \\"big\\" \\\\ call").start()\n'


@pytest.mark.parametrize(
    "error", [ExternalServiceError("exit 1: no"), RuntimeError("surprise"), OSError("gone")]
)
def test_a_hook_that_fails_ends_nothing_and_is_retried(tmp_path: Path, error: Exception) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=70.0, open_error=error)
    logs = mac.run()
    assert len(mac.marker_seen_at_open) == 3  # at 3 s, 33 s and 63 s
    assert len(mac.notifications) == 1
    assert [log["event"] for log in logs].count("call_watch.hook_failed") == 3
    assert str(error) in next(
        cast("str", log["error"]) for log in logs if log["event"] == "call_watch.hook_failed"
    )


def test_a_hook_that_recovers_is_confirmed(tmp_path: Path) -> None:
    mac = FakeMac(
        tmp_path,
        call_between(0.0, 3600.0),
        end=70.0,
        open_error=OSError("gone"),
        open_error_until=20.0,
    )
    mac.run()
    assert mac.verbs() == ["start"]
    assert mac.marker_path.read_text() == "confirmed"


def test_the_loop_warns_once_and_resends_when_open_drops_the_command(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=100.0, answers=False)
    logs = mac.run()
    assert [when for when, _, _ in mac.opens] == [3.0, 33.0, 63.0, 93.0]
    assert len(mac.notifications) == 1
    assert [log["log_level"] for log in logs if log["event"] == "call_watch.start_unconfirmed"] == [
        "error"
    ]


def test_a_notification_passes_its_text_as_an_argument(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=20.0, answers=False)
    mac.run()
    assert mac.notify_argv[0][:7] == [
        "/usr/bin/osascript",
        "-e",
        "on run argv",
        "-e",
        'display notification (item 1 of argv) with title "scribe call-watch"',
        "-e",
        "end run",
    ]
    assert mac.notify_argv[0][7] == mac.notifications[0]
    assert mac.notifications[0] not in " ".join(mac.notify_argv[0][:7])


def test_a_failed_notification_is_logged_and_changes_nothing(tmp_path: Path) -> None:
    mac = FakeMac(
        tmp_path,
        call_between(0.0, 3600.0),
        end=70.0,
        answers=False,
        notify_error=ExternalServiceError("exit 1: denied"),
    )
    logs = mac.run()
    assert [when for when, _, _ in mac.opens] == [3.0, 33.0, 63.0]
    assert [log["event"] for log in logs].count("call_watch.notify_failed") == 1


def test_a_stop_is_not_opened_when_the_app_is_not_running(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 10.0), end=80.0)
    mac.marker_path.write_text("confirmed")
    mac.app_running = False
    mac.run()
    assert mac.running_checks == 1
    assert mac.verbs() == []
    assert not mac.marker_path.exists()


def test_a_failed_running_check_sends_the_stop_and_is_logged(tmp_path: Path) -> None:
    mac = FakeMac(
        tmp_path, lambda _: QUIET, end=60.0, check_error=ExternalServiceError("exit 1: no")
    )
    mac.marker_path.write_text("confirmed")
    logs = mac.run()
    assert mac.verbs() == ["stop"]
    assert [log["event"] for log in logs].count("call_watch.running_check_failed") == 1


class TestLoopRestart:
    def test_sent_and_a_call_sends_nothing_until_the_resend(self, tmp_path: Path) -> None:
        mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=20.0, answers=False)
        mac.marker_path.write_text("sent")
        mac.run()
        assert mac.opens == []
        assert len(mac.notifications) == 1
        assert mac.marker_path.read_text() == "sent"

    def test_confirmed_and_a_call_sends_nothing(self, tmp_path: Path) -> None:
        mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=100.0)
        mac.marker_path.write_text("confirmed")
        mac.run()
        assert mac.opens == []
        assert mac.marker_path.read_text() == "confirmed"

    def test_a_marker_and_no_call_stops_after_the_grace(self, tmp_path: Path) -> None:
        mac = FakeMac(tmp_path, lambda _: QUIET, end=60.0, session_running=True)
        mac.marker_path.write_text("confirmed")
        mac.run()
        assert mac.verbs() == ["stop"]
        assert mac.opens[0][0] >= 30.0
        assert not mac.marker_path.exists()

    def test_no_marker_and_no_call_sends_nothing_and_writes_nothing(self, tmp_path: Path) -> None:
        mac = FakeMac(tmp_path, lambda _: QUIET, end=100.0, session_running=True)
        mac.run()
        assert mac.opens == []
        assert not mac.marker_path.exists()

    def test_no_marker_and_a_call_runs_the_confirmation(self, tmp_path: Path) -> None:
        mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=10.0)
        mac.run()
        assert [when for when, _, _ in mac.opens] == [3.0]

    def test_an_unreadable_marker_counts_as_sent_and_is_logged(self, tmp_path: Path) -> None:
        mac = FakeMac(tmp_path, lambda _: QUIET, end=60.0, session_running=True)
        mac.marker_path.write_text("garbled")
        logs = mac.run()
        assert mac.verbs() == ["stop"]
        assert [log["event"] for log in logs].count("call_watch.marker_unreadable") == 1


def test_a_marker_that_cannot_be_written_holds_back_no_start(tmp_path: Path) -> None:
    def call(_elapsed: float) -> frozenset[str]:
        marker_path(tmp_path, "Call").mkdir(exist_ok=True)
        return ZOOM

    mac = FakeMac(tmp_path, call, end=40.0)
    logs = mac.run()
    assert mac.sent() == [(3.0, "start")]
    assert [log["event"] for log in logs].count("call_watch.marker_failed") == 1
    assert trouble(logs) == []
    assert len(mac.notifications) == 1
    assert "Recording started" in mac.notifications[0]


def test_a_marker_write_that_fails_on_the_start_poll_still_starts(tmp_path: Path) -> None:
    blocker = marker_path(tmp_path, "Call")

    def call(elapsed: float) -> frozenset[str]:
        if elapsed == 3.0:
            blocker.mkdir()
        elif elapsed == 4.0:
            blocker.rmdir()
        return ZOOM

    mac = FakeMac(tmp_path, call, end=10.0)
    logs = mac.run()
    assert mac.sent() == [(3.0, "start")]
    assert [log["event"] for log in logs].count("call_watch.marker_failed") == 1
    assert mac.marker_path.is_file()
    assert mac.marker_path.read_text() == "confirmed"


def deny_marker_checks(
    monkeypatch: pytest.MonkeyPatch, directory: Path, denied: Callable[[], bool]
) -> None:
    """Make the marker's existence check raise while `denied()` holds.

    Injected rather than a mode-000 directory, which root could still read.
    """
    real_exists = Path.exists

    def exists(path: Path, *, follow_symlinks: bool = True) -> bool:
        if denied() and path == marker_path(directory, "Call"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_exists(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "exists", exists)


def test_a_marker_check_that_raises_during_a_recording_ends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied = {"now": False}

    def call(elapsed: float) -> frozenset[str]:
        denied["now"] = 8.0 <= elapsed < 15.0
        return ZOOM if elapsed < 20.0 else QUIET

    deny_marker_checks(monkeypatch, tmp_path, lambda: denied["now"])
    mac = FakeMac(tmp_path, call, end=60.0)
    ended: OSError | None = None
    logs: list[dict[str, Any]] = []
    try:
        logs = mac.run()
    except OSError as exc:
        ended = exc
    assert ended is None
    assert mac.verbs() == ["start", "stop"]
    assert [log["event"] for log in logs].count("call_watch.marker_failed") == 1
    assert not mac.marker_path.exists()


def test_each_marker_check_outage_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied = {"now": False}

    def call(elapsed: float) -> frozenset[str]:
        denied["now"] = 8.0 <= elapsed < 10.0 or 15.0 <= elapsed < 17.0
        return ZOOM

    deny_marker_checks(monkeypatch, tmp_path, lambda: denied["now"])
    mac = FakeMac(tmp_path, call, end=30.0)
    logs = mac.run()
    assert [log["event"] for log in logs].count("call_watch.marker_failed") == 2


def test_a_directory_removed_between_calls_is_recreated_for_the_next(tmp_path: Path) -> None:
    directory = tmp_path / "call-watch"

    def call(elapsed: float) -> frozenset[str]:
        if elapsed == 5.0:
            shutil.rmtree(directory)
        return ZOOM if elapsed >= 10.0 else QUIET

    mac = FakeMac(directory, call, end=30.0)
    mac.run()
    assert mac.verbs() == ["start"]
    assert mac.marker_path.read_text() == "confirmed"


def test_a_directory_removed_during_a_recording_does_not_lose_the_stop(tmp_path: Path) -> None:
    directory = tmp_path / "call-watch"

    def call(elapsed: float) -> frozenset[str]:
        if elapsed == 8.0:
            shutil.rmtree(directory)
        return ZOOM if elapsed < 10.0 else QUIET

    mac = FakeMac(directory, call, end=60.0)
    mac.run()
    assert mac.verbs() == ["start", "stop"]
    assert not mac.marker_path.exists()


@given(st.text(), st.text())
def test_a_marker_filename_is_safe_for_any_session_name(first: str, second: str) -> None:
    path = marker_path(Path("state"), first)
    assert path.parent == Path("state")
    assert path.name.isascii()
    assert len(path.name) < 100
    assert "/" not in path.name
    assert (path == marker_path(Path("state"), second)) == (first == second)


def test_a_name_that_is_no_filename_gets_a_marker(tmp_path: Path) -> None:
    name = "../" * 200 + "a\nb\u2028\udc80"
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=10.0)
    logs = mac.run(session=name)
    assert marker_path(tmp_path, name).read_text() == "confirmed"
    assert "call_watch.marker_failed" not in [log["event"] for log in logs]


def test_a_marker_from_another_session_is_never_acted_on(tmp_path: Path) -> None:
    first = FakeMac(tmp_path, call_between(0.0, 3600.0), end=10.0)
    first.run(session="A")
    assert first.verbs() == ["start"]
    other = FakeMac(tmp_path, lambda _: QUIET, end=200.0, session_running=True)
    other.run(session="B")
    assert other.opens == []
    back = FakeMac(tmp_path, lambda _: QUIET, end=60.0, session_running=True)
    back.run(session="A")
    assert back.verbs() == ["stop"]


def test_each_session_keeps_its_own_command_file(tmp_path: Path) -> None:
    first = FakeMac(tmp_path, call_between(0.0, 3600.0), end=40.0, answers=False)
    first.run(session="A")
    second = FakeMac(tmp_path, call_between(0.0, 3600.0), end=40.0, answers=False)
    second.run(session="B")
    assert first.sent() == second.sent() == [(3.0, "start"), (33.0, "start")]
    first_names = {name for _, name, _ in first.opens}
    second_names = {name for _, name, _ in second.opens}
    assert len(first_names) == len(second_names) == 1
    assert first_names.isdisjoint(second_names)


def _killed_after_a_stop_went_out(directory: Path) -> FakeMac:
    # The call ends at 10 s, the grace at 40 s, and the stop goes out on the poll at 41 s.
    mac = FakeMac(directory, call_between(0.0, 10.0), end=42.0)
    mac.run()
    assert mac.verbs() == ["start", "stop"]
    assert mac.marker_seen_at_open == ["sent", "stopping"]
    return mac


def test_a_restart_after_a_sent_stop_starts_the_next_call(tmp_path: Path) -> None:
    _killed_after_a_stop_went_out(tmp_path)
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=20.0)
    logs = mac.run()
    assert mac.sent() == [(3.0, "start")]
    assert trouble(logs) == []
    assert len(mac.notifications) == 1
    assert "Recording started" in mac.notifications[0]
    assert mac.marker_path.read_text() == "confirmed"


def test_a_restart_after_a_sent_stop_with_no_call_clears_the_marker(tmp_path: Path) -> None:
    _killed_after_a_stop_went_out(tmp_path)
    mac = FakeMac(tmp_path, lambda _: QUIET, end=5.0)
    mac.run()
    assert mac.opens == []
    assert mac.running_checks == 0
    assert not mac.marker_path.exists()


def test_a_restart_after_a_sent_stop_resumes_its_read_back(tmp_path: Path) -> None:
    _killed_after_a_stop_went_out(tmp_path)
    mac = FakeMac(tmp_path, lambda _: QUIET, end=40.0, session_running=True, stop_ignored=True)
    mac.run()
    assert mac.sent() == [(30.0, "stop")]
    assert len(mac.notifications) == 1
    assert mac.running_checks == 0
    assert mac.marker_path.exists()


def test_a_persons_stop_still_stands_across_a_restart(tmp_path: Path) -> None:
    first = FakeMac(tmp_path, call_between(0.0, 3600.0), end=10.0)
    first.run()
    assert first.marker_path.read_text() == "confirmed"
    mac = FakeMac(tmp_path, call_between(0.0, 3600.0), end=100.0)
    mac.run()
    assert mac.opens == []
    assert len(mac.notifications) == 1


def test_a_directory_removed_during_a_recording_gets_its_marker_back(tmp_path: Path) -> None:
    directory = tmp_path / "call-watch"

    def call(elapsed: float) -> frozenset[str]:
        if elapsed == 8.0:
            shutil.rmtree(directory)
        return ZOOM

    mac = FakeMac(directory, call, end=9.0)
    mac.run()
    assert mac.marker_path.is_file()
    assert mac.marker_path.read_text() == "confirmed"
    after = FakeMac(directory, lambda _: QUIET, end=40.0, session_running=True)
    after.run()
    assert after.verbs() == ["stop"]
    assert after.opens[0][0] >= 30.0


def test_a_read_outage_in_the_loop_notifies_once(tmp_path: Path) -> None:
    def call(elapsed: float) -> frozenset[str] | None:
        return None if 20.0 <= elapsed < 100.0 else ZOOM

    mac = FakeMac(tmp_path, call, end=110.0)
    logs = mac.run()
    assert mac.verbs() == ["start"]
    assert trouble(logs) == ["call_watch.reads_failing"]
    assert len(mac.notifications) == 2
    assert "Recording started" in mac.notifications[0]
    failing = next(log for log in logs if log["event"] == "call_watch.reads_failing")
    assert failing["log_level"] == "error"
    assert "CoreAudio status 1" in cast("str", failing["error"])
    assert [log["event"] for log in logs].count("call_watch.reads_recovered") == 1


def test_a_steady_call_logs_per_state_change_and_not_per_poll(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path, call_between(0.0, 4000.0), end=4100.0)
    logs = mac.run()
    assert len(logs) <= 6


def test_log_lines_hold_only_ids_names_times_and_error_text(tmp_path: Path) -> None:
    def call(elapsed: float) -> frozenset[str] | None:
        if 20.0 <= elapsed < 60.0:
            return None
        return ZOOM if elapsed < 100.0 else QUIET

    mac = FakeMac(tmp_path, call, end=200.0, app_running=True)
    logs = mac.run()
    assert logs
    allowed = {"event", "log_level", "session", "apps", "error", "action"}
    for log in logs:
        assert set(log) <= allowed


def test_the_directory_is_created_and_an_unusable_one_is_an_app_error(tmp_path: Path) -> None:
    mac = FakeMac(tmp_path / "a" / "b", call_between(0.0, 3600.0), end=10.0)
    mac.run()
    assert mac.directory.is_dir()
    blocked = tmp_path / "file"
    blocked.write_text("")
    with pytest.raises(AppError):
        watch(blocked / "sub", reader=mac.reader, runner=mac.runner, sleep=mac.sleep)


def test_require_macos_refuses_other_platforms() -> None:
    require_macos("darwin")
    with pytest.raises(UnsupportedPlatformError):
        require_macos("linux")


def test_importing_the_module_loads_no_framework() -> None:
    code = (
        "import ctypes\n"
        "def refuse(*a, **k): raise AssertionError('loaded at import')\n"
        "ctypes.CDLL = refuse\n"
        "import scribe.call_watch\n"
    )
    done = subprocess.run(  # noqa: S603  # this interpreter and a constant program
        [sys.executable, "-c", code], capture_output=True, check=False, timeout=60
    )
    assert done.returncode == 0, done.stderr


cli = CliRunner()


def _record_watch(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(directory: Path, **kwargs: object) -> None:
        calls.append({"directory": directory, **kwargs})

    monkeypatch.setattr("scribe.call_watch.watch", fake)
    monkeypatch.setattr("scribe.cli.sys.platform", "darwin")
    return calls


def test_the_command_exits_two_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scribe.cli.sys.platform", "linux")
    result = cli.invoke(app, ["call-watch"])
    assert result.exit_code == 2
    assert "macOS" in result.output


def test_the_command_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _record_watch(monkeypatch)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    result = cli.invoke(app, ["call-watch"])
    assert result.exit_code == 0, result.output
    assert calls == [
        {"directory": tmp_path / "scribe" / "call-watch", "session": "Call", "apps": DEFAULT_APPS}
    ]


def test_app_replaces_the_default_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _record_watch(monkeypatch)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    result = cli.invoke(app, ["call-watch", "--session", "Other", "--app", "a.b", "--app", "c.d"])
    assert result.exit_code == 0, result.output
    assert calls[0]["session"] == "Other"
    assert calls[0]["apps"] == ("a.b", "c.d")


@pytest.mark.parametrize("arguments", [["--app", ""], ["--session", ""], ["--app", "  "]])
def test_an_empty_name_is_refused(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    calls = _record_watch(monkeypatch)
    result = cli.invoke(app, ["call-watch", *arguments])
    assert result.exit_code == 2
    assert calls == []


def test_an_app_error_from_the_watcher_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_directory: Path, **_kwargs: object) -> None:
        raise ExternalServiceError("state directory")

    monkeypatch.setattr("scribe.call_watch.watch", fail)
    monkeypatch.setattr("scribe.cli.sys.platform", "darwin")
    assert cli.invoke(app, ["call-watch"]).exit_code == 2


def test_the_help_lists_only_the_two_options() -> None:
    result = cli.invoke(app, ["call-watch", "--help"])
    assert result.exit_code == 0
    assert "--session" in result.output
    assert "--app" in result.output
    assert "--help" in result.output
