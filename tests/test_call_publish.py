from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner

from scribe import call_publish
from scribe.call_publish import pair_name, publish
from scribe.cli import app
from scribe.errors import AppError, ExternalServiceError, InputValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

cli = CliRunner()

MIC = "Call 20261118 0958 mic.mp4"
APP = "Call 20261118 0958 app.mp4"
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


def recorded(folder: Path, mic: str = MIC, far: str = APP) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    return (
        _stem(folder, mic, b"mic side " * 1000, MIC_MTIME_NS),
        _stem(folder, far, b"app side " * 1500, APP_MTIME_NS),
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
    assert listing(tmp_path / "rec") == []
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


def test_a_failed_second_rename_takes_the_first_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    real = Path.rename

    def rename(self: Path, target: Path) -> Path:
        if Path(target).name == APP:
            raise OSError(5, "Input/output error")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", rename)

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


def test_a_different_file_at_a_final_name_is_never_overwritten(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    out = tmp_path / "out"
    out.mkdir()
    (out / APP).write_bytes(b"another call")

    with pytest.raises(InputValidationError, match="differs"):
        publish(mic, far, out, run=ffprobe())

    assert listing(tmp_path / "rec") == sorted([MIC, APP])
    assert listing(out) == [APP]
    assert (out / APP).read_bytes() == b"another call"


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
    assert listing(tmp_path / "rec") == []


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
    assert listing(tmp_path / "rec") == []
    assert listing(out) == sorted([MIC, APP, "Call 20261118 0958.json"])
    stems = sidecar(out / "Call 20261118 0958.json")["stems"]
    assert isinstance(stems, dict)
    assert stems["app"] == {"name": APP, "size": 13500, "md5": md5(out / APP), "duration_ms": 12346}


def test_a_missing_stem_is_refused(tmp_path: Path) -> None:
    mic, far = recorded(tmp_path / "rec")
    far.unlink()
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(AppError, match="no recording"):
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
    assert listing(tmp_path / "rec") == []


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
