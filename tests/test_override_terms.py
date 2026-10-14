from __future__ import annotations

import json
import os
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import anyio
import httpx
import pytest
from structlog.testing import capture_logs
from typer.testing import CliRunner

from scribe.cli import app
from scribe.serve import KeepRecord, dictation_client
from tests.voiceink_fixtures import closed_store
from tests.xai_fixtures import xai_payload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from starlette.types import ASGIApp
    from typer.testing import Result

    from scribe.xai_stt import XaiStt

runner = CliRunner()
KEY = "xai-test-key-never-logged"
# The recorded reply hears "Luke", which each file's alias rewrites differently.
PERSON = "Luke => Look\nZeta\nHerdr\n"
STATIC = "Luke => Lucas\nherdr\nAlpha\n"
# A rule only the person file gives, to tell its last good read from no read.
GOOD = "Zeta\nLuke => Look\n"
ENDPOINT = "/v1/audio/transcriptions"
AUDIO = {"file": ("a.wav", b"RIFF", "audio/wav")}


class Served:
    """Each app a `scribe serve` start would serve, and each request its fake xAI saw."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.apps: list[ASGIApp] = []
        self.seen: list[httpx.Request] = []
        self.static = tmp_path / "terms.txt"
        self.static.write_text(STATIC, encoding="utf-8")
        self.person = tmp_path / "person.txt"

    def start(self, *, override: bool = True, dictionary: Path | None = None) -> Result:
        args = ["serve", "--terms", str(self.static), "--no-session-terms"]
        if override:
            args += ["--override-terms", str(self.person)]
        if dictionary is None:
            args.append("--no-voiceink-dictionary")
        else:
            args += ["--voiceink-dictionary", str(dictionary)]
        return runner.invoke(app, args)

    def xai(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.seen.append(request)
        return httpx.Response(200, json=xai_payload())

    def keyterms(self, index: int = -1) -> list[str]:
        request = self.seen[index]
        boundary = request.headers["content-type"].partition("boundary=")[2].encode()
        sent: list[str] = []
        for section in request.content.split(b"--" + boundary):
            head, _, body = section.partition(b"\r\n\r\n")
            if b'name="keyterm"' in head:
                sent.append(body.removesuffix(b"\r\n").decode())
        return sent

    def dictate(self) -> str:
        """Send one dictation to the one app served; return the delivered text."""
        response = self._call(lambda client: client.post(ENDPOINT, files=AUDIO))
        assert response.status_code == 200, response.text
        return cast("str", response.json()["text"])

    def health(self) -> dict[str, object]:
        response = self._call(lambda client: client.get("/health"))
        return cast("dict[str, object]", response.json())

    def records(self) -> list[dict[str, object]]:
        root = self.tmp_path / "state" / "scribe" / "serve"
        return [
            cast("dict[str, object]", json.loads((kept / "result.json").read_text()))
            for kept in sorted(root.iterdir())
        ]

    def _call(
        self, call: Callable[[httpx.AsyncClient], Awaitable[httpx.Response]]
    ) -> httpx.Response:
        (served,) = self.apps

        async def go() -> httpx.Response:
            transport = httpx.ASGITransport(app=served)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1"
            ) as client:
                return await call(client)

        # The CLI run bound its logging to a stream the runner has since closed.
        with capture_logs():
            return anyio.run(go)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Served:
    served = Served(tmp_path)

    def record(app_: ASGIApp, **_settings: object) -> None:
        served.apps.append(app_)

    def fake_xai(_key: str) -> XaiStt:
        return dictation_client(KEY, transport=httpx.MockTransport(served.xai))

    monkeypatch.setattr("uvicorn.run", record)
    monkeypatch.setattr("scribe.serve.dictation_client", fake_xai)
    monkeypatch.setenv("XAI_API_KEY", KEY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return served


def _replace(path: Path, text: str) -> None:
    """Write `text` beside `path`, then rename it over `path`, as an editor saving atomically."""
    staged = path.with_name(path.name + ".new")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def _override(served: Served) -> dict[str, object]:
    return cast("dict[str, object]", served.health()["override_terms"])


def test_the_person_files_aliases_win_and_its_terms_lead(served: Served) -> None:
    served.person.write_text(PERSON, encoding="utf-8")
    store = closed_store(served.tmp_path / "dictionary.store", "Zorblatt", "zeta")

    result = served.start(dictionary=store)

    assert result.exit_code == 0, result.output
    text = served.dictate()

    assert "Look" in text
    assert "Lucas" not in text
    # A term in both files, or in the Dictionary, is sent once, as the person spells it.
    assert served.keyterms() == ["Zeta", "Herdr", "Alpha", "Zorblatt"]
    (record,) = served.records()
    assert record["terms"] == ["Zeta", "Herdr", "Alpha"]
    assert record["override_terms"] == 2
    assert record["dictionary_terms"] == ["Zorblatt"]
    assert record["aliases"] == [
        {"heard": "Luke", "written": "Look"},
        {"heard": "Luke", "written": "Lucas"},
    ]
    assert cast("dict[str, object]", record["flags"])["override_terms"] == str(served.person)
    assert served.health()["override_terms"] == {
        "path": str(served.person),
        "state": "ok",
        "terms": 2,
        "aliases": 1,
    }
    assert served.health()["static_terms"] == 2


def test_serve_never_writes_the_person_file(served: Served) -> None:
    served.person.write_text(PERSON, encoding="utf-8")
    os.utime(served.person, ns=(1, 1))

    assert served.start().exit_code == 0
    served.dictate()
    served.health()

    assert served.person.read_text(encoding="utf-8") == PERSON
    assert served.person.stat().st_mtime_ns == 1


def test_the_two_files_together_are_cut_to_the_keyterm_limit_person_first(
    served: Served,
) -> None:
    person = [f"person{n}" for n in range(60)]
    static = [f"static{n}" for n in range(60)]
    served.person.write_text("".join(f"{t}\n" for t in person), encoding="utf-8")
    served.static.write_text("".join(f"{t}\n" for t in static), encoding="utf-8")

    assert served.start().exit_code == 0
    served.dictate()

    assert served.keyterms() == [*person, *static[:40]]
    (record,) = served.records()
    assert record["terms"] == [*person, *static[:40]]
    assert record["override_terms"] == 60


def _fifo(path: Path) -> None:
    os.mkfifo(path)


def _unblock(path: Path) -> None:
    """Give a reader blocked on the FIFO `path` an end of file."""
    # No reader waiting means nothing is blocked, and the open fails.
    with suppress(OSError):
        os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))


PERSON_STATES: dict[str, Callable[[Path], object]] = {
    "malformed": lambda path: path.write_text("ok\nbad =>\n", encoding="utf-8"),
    "over the limit": lambda path: path.write_text(
        "".join(f"t{n}\n" for n in range(101)), encoding="utf-8"
    ),
    "undecodable": lambda path: path.write_bytes(b"ok\n\xff\xfe\n"),
    "a directory": lambda path: path.mkdir(),
    "a FIFO": _fifo,
}


@pytest.mark.parametrize("state", list(PERSON_STATES))
def test_a_bad_person_file_at_start_is_logged_and_serve_starts_without_it(
    served: Served, state: str
) -> None:
    PERSON_STATES[state](served.person)
    results: list[Result] = []
    start = threading.Thread(target=lambda: results.append(served.start()), daemon=True)

    start.start()
    start.join(timeout=10)
    if start.is_alive():
        _unblock(served.person)
        start.join(timeout=10)
        pytest.fail("reading the person file blocked serve's start")

    (result,) = results
    assert result.exit_code == 0, result.output
    if state == "malformed":
        assert result.output.count(f"{served.person}:2") == 1
    served.dictate()
    assert served.keyterms() == ["herdr", "Alpha"]
    assert _override(served) == {
        "path": str(served.person),
        "state": "failed",
        "terms": 0,
        "aliases": 0,
    }


def test_a_missing_person_file_at_start_means_no_person_terms(served: Served) -> None:
    result = served.start()

    assert result.exit_code == 0, result.output
    assert "Lucas" in served.dictate()
    assert served.keyterms() == ["herdr", "Alpha"]
    assert _override(served) == {
        "path": str(served.person),
        "state": "missing",
        "terms": 0,
        "aliases": 0,
    }
    (record,) = served.records()
    assert record["override_terms"] == 0


@pytest.mark.parametrize("state", list(PERSON_STATES))
def test_a_person_file_gone_bad_keeps_its_last_good_read(served: Served, state: str) -> None:
    served.person.write_text(GOOD, encoding="utf-8")
    assert served.start().exit_code == 0
    assert "Look" in served.dictate()

    served.person.unlink()
    PERSON_STATES[state](served.person)
    done: list[str] = []
    dictation = threading.Thread(target=lambda: done.append(served.dictate()), daemon=True)
    dictation.start()
    dictation.join(timeout=10)
    if dictation.is_alive():
        _unblock(served.person)
        dictation.join(timeout=10)
        pytest.fail("reading the person file blocked a dictation")

    assert "Look" in done[0]
    assert served.keyterms() == ["Zeta", "herdr", "Alpha"]
    assert _override(served) == {
        "path": str(served.person),
        "state": "failed",
        "terms": 1,
        "aliases": 1,
    }


def test_a_person_file_removed_after_a_good_read_gives_no_person_terms(
    served: Served,
) -> None:
    served.person.write_text(GOOD, encoding="utf-8")
    assert served.start().exit_code == 0
    served.dictate()

    served.person.unlink()

    assert "Lucas" in served.dictate()
    assert served.keyterms() == ["herdr", "Alpha"]
    assert _override(served)["state"] == "missing"


def test_a_bad_reread_is_logged_once(served: Served) -> None:
    served.person.write_text(GOOD, encoding="utf-8")
    assert served.start().exit_code == 0
    served.person.write_text("ok\nbad =>\n", encoding="utf-8")

    with capture_logs() as logs:
        (served_app,) = served.apps

        async def go() -> None:
            transport = httpx.ASGITransport(app=served_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1"
            ) as client:
                await client.post(ENDPOINT, files=AUDIO)
                await client.post(ENDPOINT, files=AUDIO)
                await client.get("/health")

        anyio.run(go)

    errors = [entry for entry in logs if entry["log_level"] == "error"]
    assert len(errors) == 1
    assert f"{served.person}:2" in str(errors[0]["error"])  # pyright: ignore[reportAny]  # captured log values are Any


EDITED = GOOD + "Omega\nOmicron => Omega\n"
PERSON_EDITS: dict[str, Callable[[Path], object]] = {
    "rewrite": lambda path: path.write_text(EDITED, encoding="utf-8"),
    "atomic replace": lambda path: _replace(path, EDITED),
}


@pytest.mark.parametrize("edit", list(PERSON_EDITS))
def test_an_edit_to_the_person_file_is_in_effect_on_the_next_dictation(
    served: Served, edit: str
) -> None:
    served.person.write_text(GOOD, encoding="utf-8")
    assert served.start().exit_code == 0
    served.dictate()

    PERSON_EDITS[edit](served.person)
    served.dictate()

    assert served.keyterms() == ["Zeta", "Omega", "herdr", "Alpha"]
    assert _override(served) == {
        "path": str(served.person),
        "state": "ok",
        "terms": 2,
        "aliases": 2,
    }


def test_an_appended_alias_is_in_effect_on_the_next_dictation(served: Served) -> None:
    served.person.write_text("Zeta\n", encoding="utf-8")
    assert served.start().exit_code == 0
    assert "Lucas" in served.dictate()

    with served.person.open("a", encoding="utf-8") as file:
        file.write("Luke => Look\n")

    assert "Look" in served.dictate()


def test_a_person_file_created_after_start_is_in_effect_on_the_next_dictation(
    served: Served,
) -> None:
    assert served.start().exit_code == 0
    served.dictate()

    served.person.write_text(GOOD, encoding="utf-8")

    assert "Look" in served.dictate()
    assert served.keyterms() == ["Zeta", "herdr", "Alpha"]
    assert _override(served)["state"] == "ok"


def test_without_override_terms_everything_is_as_before(served: Served) -> None:
    result = served.start(override=False)

    assert result.exit_code == 0, result.output
    assert "Lucas" in served.dictate()
    assert served.keyterms() == ["herdr", "Alpha"]
    health = served.health()
    assert health.get("override_terms", "absent") is None
    assert health["static_terms"] == 2
    (record,) = served.records()
    assert record["terms"] == ["herdr", "Alpha"]
    assert record["override_terms"] == 0
    assert cast("dict[str, object]", record["flags"])["override_terms"] is None


def test_a_record_written_before_override_terms_still_parses(served: Served) -> None:
    assert served.start(override=False).exit_code == 0
    served.dictate()
    (record,) = served.records()
    record.pop("override_terms", None)

    assert KeepRecord.model_validate(record).model_dump().get("override_terms") == 0


def test_help_names_override_terms() -> None:
    result = runner.invoke(app, ["serve", "--help"], terminal_width=200)

    assert "--override-terms" in result.output
