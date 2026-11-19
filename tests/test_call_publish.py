from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner

from scribe import call_publish
from scribe.call_publish import pair_name, publish
from scribe.cli import app
from scribe.errors import (
    AppError,
    ExternalServiceError,
    InputValidationError,
    PublishBusyError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

cli = CliRunner()


@pytest.fixture(autouse=True)
def own_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every run's publish lock under the test's own folder."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


MIC = "Call 20261118 0958 mic.mp4"
APP = "Call 20261118 0958 app.mp4"
RECEIPT = "Call 20261118 0958.published.json"
SECONDS = 12.3456
# Fixed and well apart, so the sidecar's start and end can be told apart.
BIRTH = 1_795_000_000
# A sub-second part, as a real recording's birth time has.
BIRTH_NS = BIRTH * 10**9 + 987_654_321
MIC_MTIME_NS = 1_795_003_600_123_456_789
APP_MTIME_NS = 1_795_003_601_500_000_000


def ffprobe(seconds: float = SECONDS) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "ffprobe"
        return subprocess.CompletedProcess(argv, 0, stdout=f"{seconds}\n", stderr="")

    return run


def broken_ffprobe(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="moov atom not found")


def _stem(folder: Path, name: str, body: bytes, mtime_ns: int) -> Path:
    path = folder / name
    path.write_bytes(body)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def recorded(
    folder: Path, mic: str = MIC, far: str = APP, *, call: bytes = b"mic side "
) -> tuple[Path, Path]:
    # `call` tells two recordings of the same sizes apart.
    folder.mkdir(parents=True, exist_ok=True)
    return (
        _stem(folder, mic, call * 1000, MIC_MTIME_NS),
        _stem(folder, far, call.replace(b"mic", b"app") * 1500, APP_MTIME_NS),
    )


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def sidecar(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def listing(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def stamp(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch).astimezone()


@pytest.mark.parametrize(
    ("mic", "far", "name"),
    [
        (MIC, APP, "Call 20261118 0958"),
        ("Call 20261118 0958 mic 2.mp4", "Call 20261118 0958 app 2.mp4", "Call 20261118 0958 2"),
        ("Call 20261118 0958 mic 12.mp4", "Call 20261118 0958 app 12.mp4", "Call 20261118 0958 12"),
    ],
)
def test_the_pair_is_named_by_its_shared_part(mic: str, far: str, name: str) -> None:
    assert pair_name(Path(mic), Path(far)) == name


@pytest.mark.parametrize(
    ("mic", "far"),
    [
        (APP, MIC),
        (MIC, MIC),
        (MIC, "Call 20261118 0959 app.mp4"),
        (MIC, "Call 20261118 0958 app 2.mp4"),
        ("Call 20261118 0958 mic 2.mp4", APP),
        ("Call 20261118 0958 mic.m4a", "Call 20261118 0958 app.m4a"),
        ("call-mic.mp4", "call-app.mp4"),
        ("Call 2026111 0958 mic.mp4", "Call 2026111 0958 app.mp4"),
    ],
)
def test_a_mismatched_pair_is_refused_before_anything_moves(
    tmp_path: Path, mic: str, far: str
) -> None:
    folder = tmp_path / "rec"
    folder.mkdir()
    (folder / mic).write_bytes(b"m")
    if far != mic:
        (folder / far).write_bytes(b"a")
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(InputValidationError, match="pair"):
        publish(folder / mic, folder / far, out, run=ffprobe())

    assert listing(out) == []
    assert (folder / mic).exists()


def test_without_a_folder_the_stems_stay_and_the_sidecar_lands_beside_them(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")

    paths = publish(mic, far, None, run=ffprobe())

    assert paths == (mic.resolve(), far.resolve())
    assert listing(tmp_path / "rec") == sorted([MIC, APP, "Call 20261118 0958.json"])
    car = sidecar(tmp_path / "rec" / "Call 20261118 0958.json")
    assert car["schema"] == "scribe.call/1"
    assert car["host"] == socket.gethostname().split(".")[0]
    assert car["stems"] == {
        "mic": {"name": MIC, "size": 9000, "md5": md5(mic), "duration_ms": 12346},
        "app": {"name": APP, "size": 13500, "md5": md5(far), "duration_ms": 12346},
    }
    assert datetime.fromisoformat(str(car["end"])) == stamp(APP_MTIME_NS // 10**9)
    assert datetime.fromisoformat(str(car["end"])).utcoffset() is not None


def test_a_folder_naming_the_stems_own_moves_nothing(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")

    paths = publish(mic, far, tmp_path / "rec" / ".." / "rec", run=ffprobe())

    assert paths == (mic.resolve(), far.resolve())
    assert listing(tmp_path / "rec") == sorted([MIC, APP, "Call 20261118 0958.json"])


def test_the_stems_move_whole_with_their_sidecar(tmp_path: Path) -> None:
    mic_name, far_name = "Call 20261118 0958 mic 2.mp4", "Call 20261118 0958 app 2.mp4"
    mic, far = recorded(tmp_path / "rec", mic_name, far_name)
    want = {"mic": (mic.read_bytes(), md5(mic)), "app": (far.read_bytes(), md5(far))}
    out = tmp_path / "out"
    out.mkdir()

    paths = publish(mic, far, out, run=ffprobe())

    assert paths == (out.resolve() / mic_name, out.resolve() / far_name)
    assert listing(tmp_path / "rec") == ["Call 20261118 0958 2.published.json"]
    assert listing(out) == sorted([mic_name, far_name, "Call 20261118 0958 2.json"])
    assert paths[0].read_bytes() == want["mic"][0]
    assert paths[1].read_bytes() == want["app"][0]
    assert paths[0].stat().st_mtime_ns == MIC_MTIME_NS
    assert paths[1].stat().st_mtime_ns == APP_MTIME_NS
    stems = sidecar(out / "Call 20261118 0958 2.json")["stems"]
    assert stems == {
        "mic": {"name": mic_name, "size": 9000, "md5": want["mic"][1], "duration_ms": 12346},
        "app": {"name": far_name, "size": 13500, "md5": want["app"][1], "duration_ms": 12346},
    }


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS records a settable birth time")
def test_the_moved_stems_keep_their_birth_time(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    # A birth time follows an mtime set earlier than it, so this puts it in the past.
    for path in (mic, far):
        os.utime(path, ns=(BIRTH_NS, BIRTH_NS))
    os.utime(mic, ns=(MIC_MTIME_NS, MIC_MTIME_NS))
    os.utime(far, ns=(APP_MTIME_NS, APP_MTIME_NS))
    births = (mic.stat().st_birthtime, far.stat().st_birthtime)
    assert all(abs(birth - BIRTH_NS / 1e9) < 1e-6 for birth in births)
    out = tmp_path / "out"
    out.mkdir()

    paths = publish(mic, far, out, run=ffprobe())

    for path, birth, mtime_ns in zip(paths, births, (MIC_MTIME_NS, APP_MTIME_NS), strict=True):
        # Python reads a birth time only as a float, which holds it to about 0.2 µs here.
        assert abs(path.stat().st_birthtime - birth) < 1e-6
        assert path.stat().st_mtime_ns == mtime_ns
    car = sidecar(out / "Call 20261118 0958.json")
    assert datetime.fromisoformat(str(car["start"])) == stamp(BIRTH)


def test_without_a_birth_time_the_start_is_the_mic_mtime_less_its_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unrecorded(_info: os.stat_result) -> None:
        return None

    monkeypatch.setattr("scribe.call_publish.birth_time", unrecorded)
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()

    publish(mic, far, out, run=ffprobe(600.0))

    car = sidecar(out / "Call 20261118 0958.json")
    assert datetime.fromisoformat(str(car["start"])) == stamp(MIC_MTIME_NS // 10**9 - 600)


def _refused_whole(tmp_path: Path, sources: tuple[Path, Path], out: Path) -> None:
    assert listing(tmp_path / "rec") == sorted(path.name for path in sources)
    assert listing(out) == []


def test_a_failed_second_copy_leaves_nothing_in_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    real = call_publish.copy_stem

    def copy(source: Path, part: Path) -> None:
        if source.name == APP:
            part.write_bytes(b"half")
            raise OSError(28, "No space left on device")
        real(source, part)

    monkeypatch.setattr("scribe.call_publish.copy_stem", copy)

    with pytest.raises(AppError, match="No space left"):
        publish(mic, far, out, run=ffprobe())

    _refused_whole(tmp_path, (mic, far), out)


def test_a_failed_second_install_takes_the_first_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    real = call_publish.install_final

    def install(part: Path, final: Path) -> None:
        if final.name == APP:
            raise OSError(5, "Input/output error")
        real(part, final)

    monkeypatch.setattr("scribe.call_publish.install_final", install)

    with pytest.raises(AppError, match="Input/output error"):
        publish(mic, far, out, run=ffprobe())

    _refused_whole(tmp_path, (mic, far), out)


def test_a_failed_probe_moves_nothing(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ExternalServiceError, match="ffprobe exited 1"):
        publish(mic, far, out, run=broken_ffprobe)

    _refused_whole(tmp_path, (mic, far), out)


def _other_call(out: Path, *names: str, size: int | None = None) -> dict[str, bytes]:
    # Another recording under a pair's names: the given size, else its own.
    body = b"another call" if size is None else b"x" * size
    for name in names:
        (out / name).write_bytes(body)
    return {name: body for name in names}


def _published_as(out: Path, suffix: str) -> tuple[Path, Path]:
    return (
        out.resolve() / f"Call 20261118 0958 mic{suffix}.mp4",
        out.resolve() / f"Call 20261118 0958 app{suffix}.mp4",
    )


def test_a_different_file_at_a_final_name_sends_the_pair_to_the_next_number(
    tmp_path: Path,
) -> None:
    mic, far = recorded(tmp_path / "rec")
    want = (mic.read_bytes(), far.read_bytes())
    out = tmp_path / "out"
    out.mkdir()
    others = _other_call(out, APP)

    paths = publish(mic, far, out, run=ffprobe())

    assert paths == _published_as(out, " 2")
    assert (paths[0].read_bytes(), paths[1].read_bytes()) == want
    assert (out / APP).read_bytes() == others[APP]
    assert not (out / MIC).exists()
    assert listing(out) == sorted([APP, paths[0].name, paths[1].name, "Call 20261118 0958 2.json"])
    stems = sidecar(out / "Call 20261118 0958 2.json")["stems"]
    assert isinstance(stems, dict)
    assert stems["mic"] == {
        "name": paths[0].name,
        "size": 9000,
        "md5": md5(paths[0]),
        "duration_ms": 12346,
    }
    assert listing(tmp_path / "rec") == [RECEIPT]


def test_a_same_size_file_of_other_content_is_not_taken_for_the_published_stem(
    tmp_path: Path,
) -> None:
    mic, far = recorded(tmp_path / "rec")
    want = (mic.read_bytes(), far.read_bytes())
    out = tmp_path / "out"
    out.mkdir()
    # Each side's name holds other bytes of exactly that side's size.
    others = {
        **_other_call(out, MIC, size=len(want[0])),
        **_other_call(out, APP, size=len(want[1])),
    }
    assert (others[MIC], others[APP]) != want

    paths = publish(mic, far, out, run=ffprobe())

    assert paths == _published_as(out, " 2")
    assert (paths[0].read_bytes(), paths[1].read_bytes()) == want
    assert (out / MIC).read_bytes() == others[MIC]
    assert (out / APP).read_bytes() == others[APP]


def test_taken_numbers_are_skipped(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    _other_call(out, MIC, APP, "Call 20261118 0958 mic 2.mp4", "Call 20261118 0958 app 3.mp4")

    paths = publish(mic, far, out, run=ffprobe())

    assert paths == _published_as(out, " 4")
    assert (out / "Call 20261118 0958 4.json").exists()


def test_a_numbered_source_counts_from_its_own_number(tmp_path: Path) -> None:
    mic_name, far_name = "Call 20261118 0958 mic 2.mp4", "Call 20261118 0958 app 2.mp4"
    mic, far = recorded(tmp_path / "rec", mic_name, far_name)
    out = tmp_path / "out"
    out.mkdir()
    _other_call(out, far_name)

    paths = publish(mic, far, out, run=ffprobe())

    assert paths == _published_as(out, " 3")
    assert not (out / MIC).exists()
    assert (out / "Call 20261118 0958 3.json").exists()


def test_a_rerun_after_a_numbered_publish_recovers_onto_its_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    _other_call(out, MIC, APP)
    real = Path.unlink

    def stuck(self: Path, *, missing_ok: bool = False) -> None:
        if self.parent.name == "rec" and self.suffix == ".mp4":
            raise PermissionError(1, "Operation not permitted")
        real(self, missing_ok=missing_ok)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", stuck)
        first = publish(mic, far, out, run=ffprobe())
    assert listing(tmp_path / "rec") == sorted([MIC, APP, RECEIPT])
    (out / "Call 20261118 0958 2.json").unlink()

    again = publish(mic, far, out, run=ffprobe())

    assert again == first == _published_as(out, " 2")
    assert listing(tmp_path / "rec") == [RECEIPT]
    assert (out / "Call 20261118 0958 2.json").exists()
    assert not (out / "Call 20261118 0958 mic 3.mp4").exists()


def test_a_published_pair_is_left_as_it_is_and_gets_a_missing_sidecar(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    first = publish(mic, far, out, run=ffprobe())
    car = out / "Call 20261118 0958.json"
    written = car.read_bytes()
    car.unlink()

    again = publish(mic, far, out, run=ffprobe())

    assert again == first
    assert car.read_bytes() == written
    assert listing(out) == sorted([MIC, APP, car.name])
    assert listing(tmp_path / "rec") == [RECEIPT]


def test_a_published_pair_keeps_the_sidecar_it_has(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    publish(mic, far, out, run=ffprobe())
    car = out / "Call 20261118 0958.json"
    car.write_text("{}", encoding="utf-8")

    publish(mic, far, out, run=broken_ffprobe)

    assert car.read_text(encoding="utf-8") == "{}"


def test_a_source_already_copied_whole_is_deleted_and_the_sidecar_written(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    for path in (mic, far):
        (out / path.name).write_bytes(path.read_bytes())

    paths = publish(mic, far, out, run=ffprobe())

    assert paths == (out.resolve() / MIC, out.resolve() / APP)
    assert listing(tmp_path / "rec") == [RECEIPT]
    assert listing(out) == sorted([MIC, APP, "Call 20261118 0958.json"])
    stems = sidecar(out / "Call 20261118 0958.json")["stems"]
    assert isinstance(stems, dict)
    assert stems["app"] == {"name": APP, "size": 13500, "md5": md5(out / APP), "duration_ms": 12346}


def test_a_missing_stem_is_refused(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    far.unlink()
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(AppError, match="published copy not found"):
        publish(mic, far, out, run=ffprobe())

    assert listing(out) == []
    assert mic.exists()


def test_cli_expands_the_folder_and_prints_the_two_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("scribe.call_publish.default_run", ffprobe())
    mic, far = recorded(tmp_path / "rec")
    (tmp_path / "Calls").mkdir()

    result = cli.invoke(app, ["call-publish", str(mic), str(far), "--to", "~/Calls"])

    assert result.exit_code == 0, result.output
    out = (tmp_path / "Calls").resolve()
    assert result.stdout == f"{out / MIC}\n{out / APP}\n"
    assert listing(tmp_path / "rec") == [RECEIPT]


def test_cli_exits_2_on_a_failure_and_prints_no_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scribe.call_publish.default_run", broken_ffprobe)
    mic, far = recorded(tmp_path / "rec")

    result = cli.invoke(app, ["call-publish", str(mic), str(far)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "ffprobe exited 1" in result.stderr


def test_a_copy_that_does_not_match_its_source_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()

    def truncated(source: Path, part: Path) -> None:
        part.write_bytes(source.read_bytes()[:-1])

    monkeypatch.setattr("scribe.call_publish.copy_stem", truncated)

    with pytest.raises(AppError, match="does not match"):
        publish(mic, far, out, run=ffprobe())

    _refused_whole(tmp_path, (mic, far), out)


def test_stems_in_two_folders_are_refused(tmp_path: Path) -> None:
    mic, _ = recorded(tmp_path / "rec")
    _, far = recorded(tmp_path / "elsewhere")

    with pytest.raises(InputValidationError, match="different folders"):
        publish(mic, far, None, run=ffprobe())

    assert listing(tmp_path / "rec") == sorted([MIC, APP])


def test_a_missing_publish_folder_is_refused(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")

    with pytest.raises(AppError, match="no folder"):
        publish(mic, far, tmp_path / "absent", run=ffprobe())

    assert listing(tmp_path / "rec") == sorted([MIC, APP])
    assert not (tmp_path / "absent").exists()


def test_a_source_that_cannot_be_deleted_still_publishes_and_a_rerun_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scribe.call_publish.default_run", ffprobe())
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    real = Path.unlink

    def stuck(self: Path, *, missing_ok: bool = False) -> None:
        if self == far:
            raise PermissionError(1, "Operation not permitted")
        real(self, missing_ok=missing_ok)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", stuck)
        result = cli.invoke(app, ["call-publish", str(mic), str(far), "--to", str(out)])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{out.resolve() / MIC}\n{out.resolve() / APP}\n"
    assert "cannot delete a published source" in result.stderr
    assert listing(tmp_path / "rec") == sorted([APP, RECEIPT])
    assert listing(out) == sorted([MIC, APP, "Call 20261118 0958.json"])

    again = cli.invoke(app, ["call-publish", str(mic), str(far), "--to", str(out)])

    assert again.exit_code == 0, again.output
    assert again.stdout == result.stdout
    assert listing(tmp_path / "rec") == [RECEIPT]


def test_a_copy_of_the_right_size_but_other_bytes_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()

    def flipped(source: Path, part: Path) -> None:
        body = bytearray(source.read_bytes())
        body[len(body) // 2] ^= 0xFF
        part.write_bytes(bytes(body))

    monkeypatch.setattr("scribe.call_publish.copy_stem", flipped)

    with pytest.raises(AppError, match="does not match"):
        publish(mic, far, out, run=ffprobe())

    _refused_whole(tmp_path, (mic, far), out)


def _failing_sidecar_write(monkeypatch: pytest.MonkeyPatch) -> None:
    real = Path.replace

    def replace(self: Path, target: Path) -> Path:
        if Path(target).suffix == ".json":
            raise OSError(28, "No space left on device")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)


def test_a_failed_sidecar_write_keeps_both_sources_and_a_rerun_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()

    with monkeypatch.context() as patched:
        _failing_sidecar_write(patched)
        with pytest.raises(AppError, match="No space left"):
            publish(mic, far, out, run=ffprobe())

    assert listing(tmp_path / "rec") == sorted([MIC, APP])
    assert listing(out) == sorted([MIC, APP])

    publish(mic, far, out, run=ffprobe())

    assert listing(tmp_path / "rec") == [RECEIPT]
    assert listing(out) == sorted([MIC, APP, "Call 20261118 0958.json"])


def test_a_final_that_changes_before_the_delete_keeps_both_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    real = Path.replace

    def replace(self: Path, target: Path) -> Path:
        done = real(self, target)
        if Path(target).suffix == ".json":
            with (out / APP).open("r+b") as handle:
                handle.truncate(10)
        return done

    monkeypatch.setattr(Path, "replace", replace)

    with pytest.raises(AppError, match="not this call's"):
        publish(mic, far, out, run=ffprobe())

    assert listing(tmp_path / "rec") == sorted([MIC, APP])


def test_a_pair_stopped_between_its_renames_is_finished(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    (out / MIC).write_bytes(mic.read_bytes())

    paths = publish(mic, far, out, run=ffprobe())

    assert paths == _published_as(out, "")
    assert paths[1].read_bytes() == b"app side " * 1500
    assert listing(tmp_path / "rec") == [RECEIPT]
    assert listing(out) == sorted([MIC, APP, "Call 20261118 0958.json"])


def _hold_lock(tmp_path: Path) -> int:
    lock = tmp_path / "state" / "scribe" / "call-publish.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return descriptor


def test_a_publish_that_waits_out_the_lock_does_nothing(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    held = _hold_lock(tmp_path)
    try:
        with pytest.raises(PublishBusyError, match="another publish held"):
            publish(mic, far, out, run=ffprobe(), lock_timeout=0.2)
    finally:
        os.close(held)

    assert listing(tmp_path / "rec") == sorted([MIC, APP])
    assert listing(out) == []


def test_cli_exits_3_when_the_lock_stays_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scribe.call_publish.default_run", ffprobe())
    monkeypatch.setattr("scribe.call_publish.LOCK_SECONDS", 0.2)
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    held = _hold_lock(tmp_path)
    try:
        result = cli.invoke(app, ["call-publish", str(mic), str(far), "--to", str(out)])
    finally:
        os.close(held)

    assert result.exit_code == 3
    assert result.stdout == ""
    assert listing(tmp_path / "rec") == sorted([MIC, APP])
    assert listing(out) == []


def test_a_duration_under_a_millisecond_is_refused_before_anything_moves(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(AppError, match="under a millisecond"):
        publish(mic, far, out, run=ffprobe(0.0004))

    _refused_whole(tmp_path, (mic, far), out)


def test_two_publishers_of_one_name_install_one_pair_and_the_loser_keeps_its_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = recorded(tmp_path / "a")
    second = recorded(tmp_path / "b", call=b"MIC SIDE ")
    assert [path.stat().st_size for path in first] == [path.stat().st_size for path in second]
    bodies = {
        "a": tuple(path.read_bytes() for path in first),
        "b": tuple(path.read_bytes() for path in second),
    }
    out = tmp_path / "out"
    out.mkdir()
    a_verified, b_verified, a_done = threading.Event(), threading.Event(), threading.Event()
    real = os.utime

    # The first time stamp a run makes falls after it verified its copies and
    # before its renames: A waits there while B stages and verifies, and B
    # waits there until A has finished.
    def utime(path: Path, *, ns: tuple[int, int]) -> None:
        name = threading.current_thread().name
        if name == "a" and not a_verified.is_set():
            a_verified.set()
            assert b_verified.wait(5)
        elif name == "b" and not b_verified.is_set():
            b_verified.set()
            assert a_done.wait(5)
        real(path, ns=ns)

    monkeypatch.setattr(os, "utime", utime)
    outcome: dict[str, tuple[Path, Path] | BaseException] = {}

    def run(name: str, stems: tuple[Path, Path]) -> None:
        try:
            # Two users' runs: each holds only its own lock.
            outcome[name] = publish(*stems, out, run=ffprobe(), state=tmp_path / f"state-{name}")
        except BaseException as exc:  # noqa: BLE001  # handed to the test thread to assert on
            outcome[name] = exc
        finally:
            if name == "a":
                a_done.set()

    a = threading.Thread(target=run, args=("a", first), name="a")
    b = threading.Thread(target=run, args=("b", second), name="b")
    a.start()
    assert a_verified.wait(5)
    b.start()
    a.join(10)
    b.join(10)

    assert outcome["a"] == _published_as(out, "")
    assert ((out / MIC).read_bytes(), (out / APP).read_bytes()) == bodies["a"]
    assert isinstance(outcome["b"], AppError)
    assert "appeared while publishing" in str(outcome["b"])
    assert listing(out) == sorted([MIC, APP, "Call 20261118 0958.json"])
    assert listing(tmp_path / "a") == [RECEIPT]
    assert listing(tmp_path / "b") == sorted([MIC, APP])

    monkeypatch.setattr(os, "utime", real)
    again = publish(*second, out, run=ffprobe())

    assert again == _published_as(out, " 2")
    assert (again[0].read_bytes(), again[1].read_bytes()) == bodies["b"]
    assert listing(tmp_path / "b") == [RECEIPT]


def test_a_repeat_after_a_renumbered_publish_returns_the_renumbered_pair(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    older = publish(*recorded(tmp_path / "rec"), out, run=ffprobe())
    newer = recorded(tmp_path / "rec", call=b"MIC SIDE ")
    bodies = tuple(path.read_bytes() for path in newer)

    published = publish(*newer, out, run=ffprobe())
    again = publish(*newer, out, run=ffprobe())

    assert older == _published_as(out, "")
    assert published == again == _published_as(out, " 2")
    assert (again[0].read_bytes(), again[1].read_bytes()) == bodies
    receipt = sidecar(tmp_path / "rec" / RECEIPT)
    assert receipt == {
        "mic": {"path": str(published[0]), "md5": md5(published[0])},
        "app": {"path": str(published[1]), "md5": md5(published[1])},
    }


def test_published_copies_without_a_receipt_are_not_taken_for_the_sources(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    for path in (mic, far):
        (out / path.name).write_bytes(path.read_bytes())
        path.unlink()

    with pytest.raises(AppError, match="published copy not found"):
        publish(mic, far, out, run=ffprobe())

    assert listing(out) == sorted([MIC, APP])


def test_a_published_copy_that_changed_since_its_receipt_is_not_taken(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    publish(mic, far, out, run=ffprobe())
    (out / APP).write_bytes(b"app side " * 1499 + b"APP SIDE ")

    with pytest.raises(AppError, match="published copy not found"):
        publish(mic, far, out, run=ffprobe())


def test_an_unreadable_receipt_is_not_taken(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    publish(mic, far, out, run=ffprobe())
    (tmp_path / "rec" / RECEIPT).write_text('{"mic": {}}', encoding="utf-8")

    with pytest.raises(AppError, match="published copy not found: unreadable"):
        publish(mic, far, out, run=ffprobe())


def test_a_run_paused_before_its_install_never_replaces_the_winners_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    winner = recorded(tmp_path / "a")
    loser = recorded(tmp_path / "b", call=b"MIC SIDE ")
    bodies = tuple(path.read_bytes() for path in winner)
    out = tmp_path / "out"
    out.mkdir()
    paused, finished = threading.Event(), threading.Event()
    real = call_publish.install_final

    # The loser has checked that its finals are free and stops before its
    # first install until the winner has published and deleted its sources.
    def install(part: Path, final: Path) -> None:
        if threading.current_thread().name == "b" and not paused.is_set():
            paused.set()
            assert finished.wait(5)
        real(part, final)

    monkeypatch.setattr("scribe.call_publish.install_final", install)
    outcome: dict[str, tuple[Path, Path] | BaseException] = {}

    def run() -> None:
        try:
            outcome["b"] = publish(*loser, out, run=ffprobe(), state=tmp_path / "state-b")
        except BaseException as exc:  # noqa: BLE001  # handed to the test thread to assert on
            outcome["b"] = exc

    b = threading.Thread(target=run, name="b")
    b.start()
    assert paused.wait(5)
    won = publish(*winner, out, run=ffprobe(), state=tmp_path / "state-a")
    finished.set()
    b.join(10)

    assert won == _published_as(out, "")
    assert listing(tmp_path / "a") == [RECEIPT]
    assert (won[0].read_bytes(), won[1].read_bytes()) == bodies
    assert isinstance(outcome["b"], AppError)
    assert listing(tmp_path / "b") == sorted([MIC, APP])
    assert listing(out) == sorted([MIC, APP, "Call 20261118 0958.json"])


@pytest.mark.parametrize(
    "platform",
    [
        pytest.param(
            "darwin",
            marks=pytest.mark.skipif(sys.platform != "darwin", reason="renamex_np is macOS's"),
        ),
        "linux",
    ],
)
def test_an_install_never_replaces_a_final(tmp_path: Path, platform: str) -> None:
    part, final, taken = tmp_path / "a.part", tmp_path / "a.mp4", tmp_path / "b.mp4"
    part.write_bytes(b"ours")
    taken.write_bytes(b"theirs")

    with pytest.raises(FileExistsError):
        call_publish.install_final(part, taken, platform)
    assert (part.read_bytes(), taken.read_bytes()) == (b"ours", b"theirs")

    call_publish.install_final(part, final, platform)
    assert final.read_bytes() == b"ours"
    assert not part.exists()


def test_a_final_swapped_for_same_size_bytes_before_the_delete_keeps_both_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    real = Path.replace

    def replace(self: Path, target: Path) -> Path:
        done = real(self, target)
        if Path(target).name == "Call 20261118 0958.json":
            swapped = bytearray((out / APP).read_bytes())
            swapped[0] ^= 0xFF
            (out / APP).write_bytes(bytes(swapped))
        return done

    monkeypatch.setattr(Path, "replace", replace)

    with pytest.raises(AppError, match="not this call's"):
        publish(mic, far, out, run=ffprobe())

    assert listing(tmp_path / "rec") == sorted([MIC, APP])
