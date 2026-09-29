"""Command line entry point: one subcommand per stage."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import typer

from scribe import attribution, ensemble, gemini_stt
from scribe.claude_cli import DEFAULT_MAX_BUDGET_USD, ClaudeCliBackend
from scribe.cleanup import (
    CLEANUP_PROMPT_VERSION,
    DEFAULT_CHUNK_WORDS,
    DEFAULT_CLEANUP_MODEL,
    CleanupRequest,
    clean,
    final_speakers,
    parse_pairs,
    provenance_header,
    read_pairs_file,
    strip_speaker_labels,
    verify_numbers,
)
from scribe.coverage import describe, fill_holes, moved, possible_drop
from scribe.curated import fill_ranges, read_front, render_curated
from scribe.diarizer import MODEL, PACKAGE, REVISION, VERSION, PyannoteDiarizer
from scribe.errors import AppError, ExternalServiceError, InputValidationError
from scribe.fidelity import check_fidelity
from scribe.gaps import check_gaps, clock, format_gap
from scribe.log import configure
from scribe.outputs import plan_outputs, prove_writable, sibling
from scribe.parakeet import ParakeetMlx
from scribe.pick import (
    DEFAULT_PICK_MODEL,
    GUARDED_DROP,
    PICK_PROMPT_VERSION,
    RESTORED_ADD,
    find_spots,
    pick_readings,
)
from scribe.schema import Engine, Source, Transcript, from_xai_response
from scribe.speakers import (
    DEFAULT_SPEAKER_MODEL,
    SPEAKER_PROMPT_VERSION,
    needs_relabeling,
    relabel,
)
from scribe.turns import (
    DEFAULT_MIN_TURN_SECONDS,
    DEFAULT_MIN_TURN_WORDS,
    DEFAULT_SNAP_WORDS,
    turns_from_speakers,
    word_speakers,
)
from scribe.vote import first_decrease, vote_transcripts
from scribe.writers import to_markdown, to_srt, to_vtt
from scribe.xai_stt import (
    DEFAULT_MODEL,
    DEFAULT_VAD_THRESHOLD,
    MAX_UPLOAD_BYTES,
    XaiStt,
    check_input,
    check_keyterms,
    check_vad_threshold,
    resolve_api_key,
)

if TYPE_CHECKING:
    from scribe.attribution import Naming
    from scribe.cleanup import CleanResult, MalformedCause, NumberDiff
    from scribe.coverage import Fill
    from scribe.diarizer import Diarization
    from scribe.fidelity import Fidelity
    from scribe.outputs import OutputPlan
    from scribe.pick import Picking
    from scribe.speakers import Relabeling

# Enough of a stderr line to act on without pasting a whole transcript into it.
_MISSING_SHOWN = 10
_MOVES_SHOWN = 3
# How a chunk whose reply was set aside is named on stderr, by the rule it broke.
_MALFORMED: dict[MalformedCause, str] = {
    "max_tokens": "stopped at the output limit",
    "outside_text": "came back with text outside its turn tags",
    "foreign_id": "came back with an id it was not sent",
    "repeated_id": "came back with one of its turns more than once",
    "missing_id": "came back without one of its turns",
    "out_of_order": "came back with its turns out of order",
    "wordless": "came back with no words",
    "short": "came back with far fewer words than it was sent",
}

app = typer.Typer(
    help="Turn recorded speech into a diarized, cleaned transcript.",
    no_args_is_help=True,
    add_completion=False,
)
main = app


class Format(StrEnum):
    """An artifact `scribe turns` can write."""

    json = "json"
    md = "md"
    srt = "srt"
    vtt = "vtt"


_SUFFIXES = {Format.json: ".turns.json", Format.md: ".md", Format.srt: ".srt", Format.vtt: ".vtt"}
_SPEAKERS_SIDECAR = "speakers sidecar"
# Distinct from 2 (nothing was run) and 3 (cleanup drift): the artifacts exist,
# but no model call behind them succeeded.
_EXIT_NO_CHUNK = 4


def _fail(exc: AppError) -> NoReturn:
    # A pydantic report is multi-line; collapsed, it is the one line a CLI owes
    # its caller instead of a traceback.
    typer.echo(f"scribe: {' '.join(str(exc).split())}", err=True)
    raise typer.Exit(2)


def _sha256(path: Path) -> str:
    """Hash a file's bytes so a transcript records which audio produced it."""
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise InputValidationError(f"cannot read audio file {path}: {exc}") from exc


def _keyterms(items: list[str], path: Path | None) -> list[str]:
    """Merge a one-term-per-line file with repeated flags, dropping repeats."""
    terms: list[str] = []
    if path is not None:
        try:
            # utf-8-sig: a byte order mark some editors write is not part of a term.
            raw = path.read_text(encoding="utf-8-sig")
        # ValueError too: a non-UTF-8 file raises UnicodeDecodeError, not OSError.
        except (OSError, ValueError) as exc:
            raise InputValidationError(f"cannot read --keyterm-file {path}: {exc}") from exc
        lines = [line.strip() for line in raw.splitlines()]
        terms = [line for line in lines if line and not line.startswith("#")]
    # A repeat would spend one of the request's term slots on nothing.
    merged = list(dict.fromkeys([*terms, *(item.strip() for item in items)]))
    check_keyterms(merged)
    return merged


def _xai_transcript(
    client: XaiStt,
    audio: Path,
    source: Source,
    *,
    model: str,
    language: str,
    format_text: bool,
    diarize: bool,
    keyterms: list[str],
    vad_threshold: float,
) -> Transcript:
    payload = client.transcribe(
        audio,
        model=model,
        language=language or None,
        format_text=format_text,
        diarize=diarize,
        keyterms=keyterms,
        vad_threshold=vad_threshold,
    )
    params: dict[str, float | int | bool | str] = {
        "language": language,
        # The effective value, not the flag: the request omits
        # `format` when no language accompanies it.
        "format": format_text and bool(language),
        "diarize": diarize,
        "vad_threshold": vad_threshold,
    }
    if keyterms:
        params["keyterms"] = json.dumps(keyterms)
    return from_xai_response(
        payload, source=source, engine=Engine(name="xai-stt", model=model, params=params)
    )


def _progress(line: str) -> None:
    typer.echo(line, err=True)


@app.command()
def transcribe(
    audio_path: Path = typer.Argument(..., metavar="AUDIO", help="Audio file to transcribe."),
    out: Path | None = typer.Option(
        None, "--out", help="Where to write the transcript. Default: AUDIO's stem beside the input."
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Language code that enables number and currency formatting; empty sends none.",
    ),
    format_text: bool = typer.Option(
        True, "--format/--no-format", help="Write numbers and currency as digits, not words."
    ),
    diarize: bool = typer.Option(
        True, "--diarize/--no-diarize", help="Ask the engine for a speaker id per word."
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Transcription model to send."),
    keyterm: list[str] = typer.Option(
        [],
        "--keyterm",
        help="A name or term to bias recognition toward. Repeat for more.",
    ),
    keyterm_file: Path | None = typer.Option(
        None, "--keyterm-file", help="File of terms, one per line; # comments and blanks skipped."
    ),
    vad_threshold: float = typer.Option(
        DEFAULT_VAD_THRESHOLD,
        "--vad-threshold",
        help="Speech probability, 0 to 1, below which xAI skips audio as silence; 0 is off.",
    ),
    vote_engines: bool = typer.Option(
        False, "--vote", help="Also run Parakeet and Gemini, and vote the three into --out."
    ),
    max_usd: float | None = typer.Option(
        None,
        "--max-usd",
        help="With --vote, the most Gemini may spend, in USD. "
        f"Default: {gemini_stt.DEFAULT_MAX_USD}",
    ),
    cross_check: bool = typer.Option(
        True,
        "--cross-check/--no-cross-check",
        help="Fill holes in the words from a local Parakeet transcript of the same audio.",
    ),
    pick: bool = typer.Option(
        True,
        "--pick/--no-pick",
        help="Where the filled words and Parakeet's disagree, have a model pick which was said.",
    ),
    pick_model: str = typer.Option(
        DEFAULT_PICK_MODEL, "--pick-model", help="Model the pick is asked for."
    ),
    pick_context: str | None = typer.Option(
        None,
        "--pick-context",
        help="Background on the recording for the pick. Never added to the transcript.",
    ),
) -> None:
    """Transcribe an audio file with xAI speech-to-text.

    Needs XAI_API_KEY in the environment. Render the result with `scribe turns`.
    xAI costs about $0.10 per audio hour.

    --keyterm and --keyterm-file together may give at most 100 distinct terms
    of at most 50 characters each, the limits the API documents. The terms sent
    are recorded in the transcript's engine params.

    As keyterms, list distinctive names and terms: a short, common-sounding
    name can be forced in where a similar word was said.

    With --vote, Parakeet TDT v3 transcribes the audio locally, then xAI, then
    Gemini, its words timed by Parakeet's, and a word of xAI's changes, or one is
    added, only where the other two agree. --out holds the voted transcript;
    each engine's own is written beside it as soon as that engine finishes, as
    BASE.parakeet.json, BASE.xai.json and BASE.gemini.json, where BASE is --out
    less .transcript.json or .json. A failed engine exits 2 and keeps those
    already written. --vote needs Apple silicon, uvx, ffmpeg and ffprobe on PATH,
    and GEMINI_API_KEY besides XAI_API_KEY; the audio goes to xAI and to Google.
    It costs xAI about $0.10 and Gemini about $0.55 per audio hour, Gemini's part
    capped by --max-usd. A first Parakeet run downloads a ~1.2 GB model.

    Unless --no-cross-check is given, Parakeet TDT v3 then transcribes the
    audio locally, about 45 s per audio hour on Apple silicon (a first run
    downloads a ~1.2 GB model), and its transcript goes beside --out as
    BASE.parakeet.json. Where 3 or more of xAI's words in a row are each more
    than 2 s from where Parakeet heard them, they first move to Parakeet's
    times, and each run moved is printed. Where xAI's words then have a hole
    of 2 s or more in which Parakeet heard at least 3 words xAI has nowhere
    near, those words are inserted, with no speaker (`scribe turns` shows them
    as "Speaker ?"), and each filled span is printed. With --vote the voted words are filled from
    the Parakeet run the vote made. Where Parakeet cannot run, the holes are
    checked by loudness as `scribe gaps` does. Nothing in the cross-check
    changes the exit code or the path printed: a failure is one stderr line,
    and --out keeps xAI's words.

    Unless --no-pick is given, a model then picks, at each spot where the
    filled words and Parakeet's disagree in what was said, which of the two
    readings was said, as `scribe pick` does; --pick-model and --pick-context
    are its --model and --context. It is asked through the `claude` CLI on
    PATH: the words, not the audio, go to Anthropic. No pick runs with --vote
    or --no-cross-check, or where Parakeet could not run, and --pick-context
    with --no-pick, --vote or --no-cross-check exits 2. Nor does the pick
    change the exit code or the path printed: a failure ends in one stderr
    line (a retried claude call is logged as a warning before it), and --out
    keeps the filled words.
    """
    started = time.monotonic()
    try:
        if max_usd is not None and not vote_engines:
            raise InputValidationError("--max-usd caps Gemini's spending, which only --vote runs")
        if pick_context is not None and (not pick or vote_engines or not cross_check):
            raise InputValidationError(
                "--pick-context is background for the pick, which --no-pick, --vote "
                "and --no-cross-check each turn off"
            )
        terms = _keyterms(keyterm, keyterm_file)
        check_vad_threshold(vad_threshold)
        client = XaiStt(resolve_api_key())
        # Before anything opens the input: reading a FIFO or a character device
        # blocks, so the regular-file check has to come first, not just inside
        # the client. The client still runs it against its own max_bytes.
        check_input(audio_path, MAX_UPLOAD_BYTES)
        destination = (
            audio_path.with_name(f"{audio_path.stem}.transcript.json") if out is None else out
        )
        # Planned before the upload, which is paid for.
        inputs = {"the audio file": audio_path}
        if keyterm_file is not None:
            inputs["the --keyterm-file"] = keyterm_file
        xai = partial(
            _xai_transcript,
            client,
            audio_path,
            model=model,
            language=language,
            format_text=format_text,
            diarize=diarize,
            keyterms=terms,
            vad_threshold=vad_threshold,
        )
        if vote_engines:
            # Gemini logs through structlog, which prints to stdout unconfigured.
            # Warnings only: its pre-flight figure is in the Gemini progress line.
            configure("warning")
            voted = ensemble.transcribe_voted(
                audio_path,
                Source(kind="audio", ref=str(audio_path), sha256=_sha256(audio_path)),
                destination,
                inputs,
                xai=xai,
                parakeet=ParakeetMlx(),
                max_usd=gemini_stt.DEFAULT_MAX_USD if max_usd is None else max_usd,
                report=_progress,
                fill=cross_check,
            )
            typer.echo(str(voted))
            return
        plan = plan_outputs({"--out": destination}, inputs)
        # The hash and the upload read the file separately, so a file that
        # changes between them (or between upload retries) is sent as bytes the
        # recorded digest does not describe: it names the bytes read here.
        # Accepted, rather than holding a 500 MB upload in memory to hash it.
        digest = _sha256(audio_path)
        transcript = xai(Source(kind="audio", ref=str(audio_path), sha256=digest))
        try:
            transcript.dump(plan["--out"])
        # ValueError too: `model_dump_json` raises PydanticSerializationError, a
        # ValueError, on upstream text that UTF-8 cannot encode.
        except (OSError, ValueError) as exc:
            raise InputValidationError(
                f"cannot write transcript to {plan['--out']}: {exc}"
            ) from exc
    except AppError as exc:
        _fail(exc)

    spoken = "audio" if transcript.duration is None else f"{transcript.duration:.1f} s of audio"
    typer.echo(f"transcribed {spoken} in {time.monotonic() - started:.1f} s", err=True)
    if cross_check:
        checked = _cross_check(audio_path, transcript, plan["--out"], inputs)
        if pick and checked is not None:
            _pick(*checked, plan["--out"], model=pick_model, context=pick_context)
    typer.echo(str(plan["--out"]))


def _warn(message: str) -> None:
    typer.echo(f"scribe: {' '.join(message.split())}", err=True)


def _replace(transcript: Transcript, path: Path) -> None:
    """Write `transcript` over `path` whole: a write that fails leaves `path` as it was."""
    target = path.resolve()
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        # mkstemp's owner-only mode would otherwise replace the file's own.
        temporary.chmod(stat.S_IMODE(target.stat().st_mode))
        transcript.dump(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _cross_check(
    audio: Path, transcript: Transcript, out: Path, inputs: dict[str, Path]
) -> tuple[Transcript, Transcript] | None:
    """Fill `out`'s holes from Parakeet, or check them by loudness; never fail the run.

    Returns:
        The filled transcript and Parakeet's, when the filled one was written
        to `out`; otherwise None.

    """
    label = "the Parakeet transcript"
    try:
        kept: Path | None = plan_outputs(
            {label: sibling(out, ".parakeet.json")}, {"--out": out, **inputs}
        )[label]
    except AppError as exc:
        _warn(f"not keeping Parakeet's transcript: {exc}")
        kept = None
    try:
        backend = ParakeetMlx()
        backend.resolve()
        _warn(
            "cross-checking against Parakeet, run locally "
            "(about 45 s per audio hour; a first run downloads a ~1.2 GB model)"
        )
        reference = backend.transcribe(audio, source=transcript.source)
        # Filled from no words, every hole would be reported clear without being checked.
        if transcript.words and not reference.words:
            raise ExternalServiceError(f"it heard no words in {audio}")
    # OSError too: the run's temporary directory can fail to be made or removed.
    except (AppError, OSError) as exc:
        _warn(f"Parakeet cannot cross-check: {exc}; checking the holes by loudness instead")
        try:
            found = check_gaps(transcript, audio)
        except AppError as exc:
            _warn(f"no cross-check ran: {exc}")
            return None
        for gap in found.gaps:
            _warn(possible_drop(gap.start, gap.end))
        flags = f"{len(found.gaps)} {'hole' if len(found.gaps) == 1 else 'holes'} flagged"
        _warn(f"cross-check by loudness: {flags}")
        return None
    if kept is not None:
        try:
            reference.dump(kept)
        except (OSError, ValueError) as exc:
            _warn(f"not keeping Parakeet's transcript: cannot write {kept}: {exc}")
    filled, fill = fill_holes(transcript, reference)
    _report_fill(fill)
    try:
        _replace(filled, out)
    except (OSError, ValueError) as exc:
        _warn(f"cannot write the filled transcript to {out}: {exc}; it keeps xAI's words")
        return None
    return filled, reference


def _pick(
    filled: Transcript, reference: Transcript, out: Path, *, model: str, context: str | None
) -> None:
    """Pick between `filled`'s words and `reference`'s where they disagree; never fail the run."""
    # The pick logs through structlog, which prints to stdout unconfigured.
    configure("warning")
    kept = "--out keeps the filled words"
    for role, words in (("the filled transcript's", filled.words), ("Parakeet's", reference.words)):
        # xAI's words can step back, and the pick refuses words out of start order.
        if (index := first_decrease(words)) is not None:
            _warn(f"not picking: {role} word {index} starts before word {index - 1}; {kept}")
            return
    if not find_spots(filled.words, reference.words):
        return
    try:
        backend = ClaudeCliBackend(model=model, disable_tools=True)
        backend.resolve()
    except AppError as exc:
        _warn(f"not picking: {exc}; {kept}")
        return
    picking = pick_readings(filled, reference, backend, context=context)
    if len(picking.failed) == len(picking.chunks):
        _warn(f"the pick failed on {len(picking.failed)} of {len(picking.chunks)} chunks; {kept}")
        return
    try:
        _replace(picking.transcript, out)
    except (OSError, ValueError) as exc:
        _warn(f"cannot write the picked transcript to {out}: {exc}; {kept}")
        return
    _report_pick(picking, backend.model)


def _speakers_metadata(result: Relabeling, backend: ClaudeCliBackend, *, words: int) -> str:
    metadata = {
        "backend": backend.name,
        "model": backend.model,
        "prompt_version": SPEAKER_PROMPT_VERSION,
        "words": words,
        "words_relabeled": result.relabeled,
        "chunks_failed": list(result.failed),
        "chunks": [dataclasses.asdict(chunk) for chunk in result.chunks],
    }
    return json.dumps(metadata, indent=2) + "\n"


def _report_speakers(result: Relabeling, backend: ClaudeCliBackend, *, words: int) -> str:
    return (
        f"scribe: speakers relabeled by {backend.model} (prompt {SPEAKER_PROMPT_VERSION}): "
        f"{result.relabeled} of {words} words changed, "
        f"{len(result.chunks) - len(result.failed)} of {len(result.chunks)} chunks answered"
    )


@app.command()
def turns(
    input_path: Path = typer.Argument(
        ..., metavar="INPUT.json", help="Transcript JSON, as `scribe schema` describes it."
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", help="Where to write artifacts. Default: beside the input."
    ),
    formats: list[Format] = typer.Option(
        [Format.json, Format.md], "--format", help="Artifact to write; repeat for more than one."
    ),
    min_turn_seconds: float = typer.Option(
        DEFAULT_MIN_TURN_SECONDS,
        "--min-turn-seconds",
        help="Turns shorter than this may be merged into a neighbor.",
    ),
    min_turn_words: int = typer.Option(
        DEFAULT_MIN_TURN_WORDS,
        "--min-turn-words",
        help="Turns with fewer words than this may be merged.",
    ),
    snap_words: int = typer.Option(
        DEFAULT_SNAP_WORDS,
        "--snap-words",
        min=0,
        help="Words a mid-sentence speaker switch may move to reach a sentence end; 0 is off.",
    ),
    llm_speakers: bool = typer.Option(
        True,
        "--llm-speakers/--no-llm-speakers",
        help="Have a model correct who said what, one call per ~700 words; the words never change.",
    ),
    speaker_model: str = typer.Option(
        DEFAULT_SPEAKER_MODEL, "--speaker-model", help="Model the speaker pass is asked for."
    ),
    audio_speakers: bool = typer.Option(
        True,
        "--audio-speakers/--no-audio-speakers",
        help="Name 'Speaker ?' words from a local diarization of the audio where a voice "
        "match agrees.",
    ),
    audio: Path | None = typer.Option(
        None, "--audio", help="The recording, for --audio-speakers. Default: INPUT's source audio."
    ),
    to_stdout: bool = typer.Option(
        False, "--stdout", help="Print the markdown to stdout and write nothing."
    ),
) -> None:
    """Group a transcript's words into speaker turns and render them.

    A transcript that arrived with turns but no word timings keeps its turns:
    there is nothing to rebuild them from.

    Unless --no-llm-speakers is given, a model then corrects each word's
    speaker through the `claude` CLI on PATH, on its own subscription auth. Its
    provenance goes to INPUT's stem plus .speakers.json beside the artifacts
    (to stderr with --stdout); a run that writes none removes one left there.
    The pass can reorder who counts as "Speaker 1", "Speaker 2", ..., so a
    `cleanup --speaker` key made from other output may need its numbers
    checked. A chunk whose call fails keeps the diarizer's
    speakers, with one stderr line saying so.

    Unless --no-audio-speakers is given, the words no engine attributed
    ("Speaker ?") are then named from the audio, when there are any and at
    least two speakers: pyannote's Community-1 diarizes the recording locally,
    and a word takes a speaker only where the diarizer's cluster for it and a
    match of its voice against each speaker's other speech agree; the rest stay
    "Speaker ?". The audio is --audio, else INPUT's source audio, read relative
    to the current directory; a recorded sha256 must match it. It needs Apple
    silicon, uvx and ffmpeg on PATH, and takes about 85 s per audio hour; a
    first run downloads ~960 MB. The model is gated: accept its terms at
    https://hf.co/pyannote/speaker-diarization-community-1 and set HF_TOKEN.
    A named word before its speaker's first word changes the "Speaker N" ranks
    too, so a `cleanup --speaker` key made from other output may need its
    numbers checked. What was named goes to the turns JSON's engine params. A
    failure is one stderr line, and the words stay unattributed.

    Exit 2 is a bad input, an unwritable output, or, with the pass on, no usable
    claude CLI, found before anything is written; so are --audio with
    --no-audio-speakers and an --audio that does not exist or is not a regular
    file. Exit 4 means every model call failed; the artifacts are written
    anyway, with the diarizer's speakers.
    """
    # With --stdout the markdown is the output; unconfigured, structlog prints there.
    configure()
    try:
        _check_audio(audio, audio_speakers=audio_speakers)
        transcript = Transcript.load(input_path)
        if not transcript.words and not transcript.turns:
            raise InputValidationError(f"{input_path} has neither words nor turns")
    except AppError as exc:
        _fail(exc)

    words = transcript.words
    speakers = word_speakers(
        words,
        min_turn_seconds=min_turn_seconds,
        min_turn_words=min_turn_words,
        snap_words=snap_words,
    )
    relabeling = llm_speakers and needs_relabeling(speakers)

    directory = input_path.parent if out_dir is None else out_dir
    outputs: dict[str, Path] = {
        fmt: directory / f"{input_path.stem}{_SUFFIXES[fmt]}" for fmt in formats
    }
    sidecar = directory / f"{input_path.stem}.speakers.json"
    if relabeling:
        outputs[_SPEAKERS_SIDECAR] = sidecar
    try:
        # Before the mkdir, so a refused run leaves no new directory behind,
        # and before any model call is paid for.
        plan = (
            None
            if to_stdout
            else plan_outputs(outputs, {"the input": input_path}, missing_parent_ok=True)
        )
    except AppError as exc:
        _fail(exc)

    backend = _speaker_backend(speaker_model) if relabeling else None
    if plan is not None:
        # Before the calls, which are paid for; after the CLI lookup, so a
        # machine without `claude` is left without a new directory.
        _make_directory(directory)
        try:
            prove_writable(plan)
        except AppError as exc:
            _fail(exc)
        if not relabeling:
            _remove_stale_sidecar(sidecar)
    relabeled = (
        None
        if backend is None
        else (relabel([word.text for word in words], speakers, backend), backend)
    )
    if relabeled is not None:
        speakers = list(relabeled[0].speakers)
    # The words keep their own speakers, so an earlier run's names are not in them: its
    # counts would describe turns this run builds only if it names the words again.
    engine = _without_diarizer(transcript.engine)
    # After the pass: it leaves unattributed words alone, and could undo a name from audio.
    if audio_speakers:
        speakers, engine = _name_from_audio(transcript, speakers, audio, engine)
    built = (
        transcript
        if not words
        else transcript.model_copy(
            update={"engine": engine, "turns": turns_from_speakers(words, speakers)}
        )
    )

    if plan is None:
        typer.echo(to_markdown(built), nl=False)
    else:
        _write_turn_artifacts(plan, directory, formats, built, relabeled)
    if relabeled is not None:
        _finish_speakers(*relabeled, words=len(words), printed=plan is None)


def _check_audio(audio: Path | None, *, audio_speakers: bool) -> None:
    if audio is not None and not audio_speakers:
        raise InputValidationError("--audio is read only by --audio-speakers")
    # A usage error like a missing INPUT, where a missing source audio is not.
    if audio is not None and not audio.exists():
        raise InputValidationError(f"--audio {audio} does not exist")
    if audio is not None:
        check_input(audio, sys.maxsize)


def _source_audio(source: Source, audio: Path | None) -> Path:
    if audio is not None:
        path = audio
    elif source.kind == "audio":
        # The path as typed at transcription, so a relative one is read from here.
        path = Path(source.ref)
        if not path.exists():
            raise InputValidationError(f"the source audio {path} does not exist; pass --audio")
    else:
        raise InputValidationError("the transcript names no source audio; pass --audio")
    # Before the hash or ffmpeg opens it: reading a FIFO or a character device blocks.
    check_input(path, sys.maxsize)
    if source.sha256 is not None and _sha256(path) != source.sha256:
        raise InputValidationError(f"{path} is not the audio the transcript was made from")
    return path


def _name_from_audio(
    transcript: Transcript, speakers: list[int | None], audio: Path | None, engine: Engine
) -> tuple[list[int | None], Engine]:
    """Name the unattributed words from the audio and record it in `engine`; never fail the run."""
    if not attribution.can_name(speakers):
        return speakers, engine
    words, unattributed = transcript.words, speakers.count(None)
    try:
        path = _source_audio(transcript.source, audio)
        backend = PyannoteDiarizer()
        backend.resolve()
        _warn(
            f"naming {unattributed} unattributed words with pyannote Community-1, run locally "
            "(about 85 s per audio hour; a first run downloads ~960 MB)"
        )
        planned = attribution.plan(words, speakers)
        diarization = backend.diarize(path, planned.intervals)
        naming = attribution.name_unattributed(
            planned, words, speakers, diarization.speech, diarization.embeddings
        )
    # OSError too: the run's temporary directory can fail to be made.
    except (AppError, OSError) as exc:
        _warn(f"the diarizer cannot name {unattributed} unattributed words: {exc}")
        return speakers, engine
    _warn(f"the diarizer named {naming.named} of {unattributed} unattributed words")
    return list(naming.speakers), _diarizer_engine(engine, naming, diarization)


def _without_diarizer(engine: Engine) -> Engine:
    params = {
        key: value
        for key, value in engine.params.items()
        if key != "diarizer" and not key.startswith("diarizer_")
    }
    return engine.model_copy(update={"params": params})


def _diarizer_engine(engine: Engine, naming: Naming, diarization: Diarization) -> Engine:
    runs = [
        [run.start, run.end, run.words, {str(speaker): count for speaker, count in run.named}]
        for run in naming.runs
    ]
    params = engine.params | {
        "diarizer": f"{PACKAGE} {VERSION} {MODEL}@{REVISION}",
        "diarizer_device": diarization.device,
        "diarizer_unattributed": naming.unattributed,
        "diarizer_named": naming.named,
        "diarizer_runs": json.dumps(runs),
    }
    return engine.model_copy(update={"params": params})


def _speaker_backend(model: str) -> ClaudeCliBackend:
    backend = ClaudeCliBackend(model=model, disable_tools=True)
    try:
        # Once, before the fan-out, so a machine without `claude` fails before
        # anything is written.
        backend.resolve()
    except AppError as exc:
        typer.echo(
            f"scribe: {' '.join(str(exc).split())}; "
            "pass --no-llm-speakers to build turns without it",
            err=True,
        )
        raise typer.Exit(2) from exc
    return backend


def _make_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # An unusable --out-dir is caller input, so it earns the same one stderr
        # line and exit 2 that an unreadable input file gets.
        _fail(InputValidationError(f"cannot write artifacts to {directory}: {exc}"))


def _remove_stale_sidecar(path: Path) -> None:
    # This run writes no sidecar, so one an earlier run left would describe the
    # artifacts about to replace its own. A link goes, never what it points
    # to, and a directory there is not one scribe wrote.
    if not (path.is_symlink() or path.is_file()):
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _fail(InputValidationError(f"cannot remove the stale {path}: {exc}"))


def _write_turn_artifacts(
    plan: OutputPlan,
    directory: Path,
    formats: list[Format],
    built: Transcript,
    relabeled: tuple[Relabeling, ClaudeCliBackend] | None,
) -> None:
    try:
        if Format.json in formats:
            built.dump(plan[Format.json])
        if Format.md in formats:
            plan[Format.md].write_text(to_markdown(built), encoding="utf-8")
        if Format.srt in formats:
            plan[Format.srt].write_text(to_srt(built), encoding="utf-8")
        if Format.vtt in formats:
            plan[Format.vtt].write_text(to_vtt(built), encoding="utf-8")
        if relabeled is not None:
            plan[_SPEAKERS_SIDECAR].write_text(
                _speakers_metadata(*relabeled, words=len(built.words)), encoding="utf-8"
            )
    except OSError as exc:
        _fail(InputValidationError(f"cannot write artifacts to {directory}: {exc}"))


def _finish_speakers(
    result: Relabeling, backend: ClaudeCliBackend, *, words: int, printed: bool
) -> None:
    if printed:
        # No sidecar beside printed markdown, so its provenance goes here.
        typer.echo(_report_speakers(result, backend, words=words), err=True)
    if result.failed:
        listed = ", ".join(str(index) for index in result.failed)
        typer.echo(
            f"scribe: the speaker pass failed on {len(result.failed)} of {len(result.chunks)} "
            f"chunks ({listed}); their words keep the diarizer's speakers",
            err=True,
        )
    if len(result.failed) == len(result.chunks):
        raise typer.Exit(_EXIT_NO_CHUNK)


def _turns_input(path: Path) -> Transcript:
    """Load a transcript and insist its turns are filled."""
    transcript = Transcript.load(path)
    if not transcript.turns:
        raise InputValidationError(f"{path} has no turns; run `scribe turns` on it first")
    return transcript


def _pairs(items: list[str], path: Path | None, *, option: str, file_option: str) -> dict[str, str]:
    """Merge a KEY=VALUE file with repeated flags, the flags winning."""
    pairs = {} if path is None else read_pairs_file(path, file_option)
    pairs.update(parse_pairs(items, option))
    return pairs


def _clean_destination(input_path: Path, out: Path | None) -> Path:
    if out is not None:
        return out
    return input_path.with_name(f"{input_path.stem.removesuffix('.turns')}.clean.md")


def _sidecar(destination: Path) -> Path:
    # Derived from the OUTPUT, so an --out pointing elsewhere leaves nothing at
    # all beside the input.
    return destination.with_name(f"{destination.stem.removesuffix('.clean')}.cleanup.json")


def _write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    # ValueError too: text the filesystem encoding cannot represent fails here,
    # not at the render.
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"cannot write {path}: {exc}") from exc


def _write_sidecar(
    path: Path,
    result: CleanResult,
    diff: NumberDiff,
    fidelity: Fidelity,
    backend: ClaudeCliBackend,
) -> None:
    metadata = {
        "backend": backend.name,
        "model": backend.model,
        "prompt_version": CLEANUP_PROMPT_VERSION,
        "chunks": result.chunks,
        "truncated_chunks": result.truncated_chunks,
        "malformed_chunks": [item.model_dump() for item in result.malformed_chunks],
        "emptied_turns": result.emptied_turns,
        "stripped_labels": result.stripped_labels,
        # Metadata only: the replies' text would make this a second copy of the
        # recording, and the prompts would make it a copy of the input.
        "completions": [item.model_dump(exclude={"text"}) for item in result.completions],
        "numbers_checked": diff.checked,
        "numbers_missing": diff.missing,
        "numbers_reduced": diff.reduced,
        "numbers_added": diff.added,
        "fidelity": fidelity.model_dump(),
    }
    _write_text(path, json.dumps(metadata, indent=2) + "\n")


@app.command()
def cleanup(
    input_path: Path = typer.Argument(
        ..., metavar="INPUT.turns.json", help="Transcript JSON whose turns are already built."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Markdown to write. Default: INPUT's stem plus .clean.md, beside it."
    ),
    curated: Path | None = typer.Option(
        None,
        "--curated",
        help="Also write a reading copy here: each turn under its speaker and start time.",
    ),
    front: Path | None = typer.Option(
        None, "--front", help="File whose text opens the --curated copy, copied as written."
    ),
    model: str = typer.Option(
        DEFAULT_CLEANUP_MODEL, "--model", help="Model the cleanup backend is asked for."
    ),
    speaker: list[str] = typer.Option(
        [], "--speaker", help='Rename a speaker, as "Speaker 1=Ann Lee". Repeat for more.'
    ),
    speakers_file: Path | None = typer.Option(
        None, "--speakers-file", help="File of Speaker=Name lines; # comments and blanks skipped."
    ),
    glossary: list[str] = typer.Option(
        [], "--glossary", help='Correct an ASR garble, as "ackme=Acme". Repeat for more.'
    ),
    glossary_file: Path | None = typer.Option(
        None, "--glossary-file", help="File of wrong=right lines; # comments and blanks skipped."
    ),
    context: str | None = typer.Option(
        None, "--context", help="Background on the recording. Never added to the transcript."
    ),
    title: str | None = typer.Option(
        None, "--title", help="Title for the provenance header. Default: the input's stem."
    ),
    max_budget_usd: float = typer.Option(
        DEFAULT_MAX_BUDGET_USD,
        "--max-budget-usd",
        help="Ceiling for ONE backend call, not for the run. The floor is about $0.10 a call.",
    ),
    chunk_words: int = typer.Option(
        DEFAULT_CHUNK_WORDS,
        "--chunk-words",
        min=1,
        help="Words per chunk; each chunk is one call.",
    ),
    allow_number_drift: bool = typer.Option(
        False, "--allow-number-drift", help="Still exit 0 when a number went missing or changed."
    ),
    allow_speaker_moves: bool = typer.Option(
        False,
        "--allow-speaker-moves",
        help="Still exit 0 when cleanup put words under another speaker.",
    ),
) -> None:
    """Clean up a transcript's turns with an LLM, checking numbers and speakers survive.

    Needs the `claude` CLI on PATH, on its own subscription auth. Writes the
    cleaned markdown and a metadata sidecar beside it. Exit 3 means a number
    went missing or changed, a word ended up under another speaker than the
    one who said it, or a chunk's reply was malformed and the chunk kept its
    input text uncleaned; both files are written anyway. Content words
    changed beyond the edits cleanup is asked to make are warned about and
    recorded in the sidecar, without changing the exit.
    A bad input or a backend failure exits 2.
    """
    # stdout carries only the output path; unconfigured, structlog prints there.
    configure()
    try:
        if front is not None and curated is None:
            raise InputValidationError("--front needs --curated")
        transcript = _turns_input(input_path)
        request = CleanupRequest(
            turns=transcript.turns,
            speaker_key=_pairs(
                speaker, speakers_file, option="--speaker", file_option="--speakers-file"
            ),
            glossary=_pairs(
                glossary, glossary_file, option="--glossary", file_option="--glossary-file"
            ),
            context=context,
        )
        front_text = None if front is None else read_front(front)
        # Only the reading copy reads the fill record, so a malformed one fails
        # only a run that writes it.
        ranges = [] if curated is None else fill_ranges(transcript.engine, input_path)
        destination = _clean_destination(input_path, out)
        # Planned before a backend call is paid for. The input transcript is a
        # paid transcription's output and nothing else holds a copy.
        inputs = {"the input transcript": input_path}
        if speakers_file is not None:
            inputs["the --speakers-file"] = speakers_file
        if glossary_file is not None:
            inputs["the --glossary-file"] = glossary_file
        if front is not None:
            inputs["the --front file"] = front
        outputs = {"--out": destination, "sidecar": _sidecar(destination)}
        if curated is not None:
            outputs["--curated"] = curated
        plan = plan_outputs(outputs, inputs)
        prove_writable(plan)
        backend = ClaudeCliBackend(model=model, max_budget_usd=max_budget_usd)
        result = clean(request, backend, max_words=chunk_words)
        diff = verify_numbers(
            "\n\n".join(turn.text for turn in request.turns),
            strip_speaker_labels(result.text, final_speakers(request)),
        )
        fidelity = check_fidelity(request, result.text, max_words=chunk_words)
        header = provenance_header(
            title=title or input_path.stem.removesuffix(".turns"),
            source=transcript.source,
            engine=transcript.engine,
            backend_name=backend.name,
            backend_model=backend.model,
            speaker_key=request.speaker_key,
            number_diff=diff,
            generated_at=datetime.now(UTC),
        )
        _write_text(plan["--out"], header + result.text)
        _write_sidecar(plan["sidecar"], result, diff, fidelity, backend)
        # Rendered from the kept turns, not from the markdown: a turn's text
        # can hold blank lines, so the markdown does not split back into turns.
        if curated is not None:
            _write_text(plan["--curated"], render_curated(request, result.kept, ranges, front_text))
    except AppError as exc:
        _fail(exc)

    _report_replies(result)
    if diff.missing:
        listed = ", ".join(diff.missing[:_MISSING_SHOWN])
        typer.echo(
            f"scribe: {len(diff.missing)} number(s) did not survive cleanup: {listed}", err=True
        )
    if diff.reduced:
        listed = ", ".join(diff.reduced[:_MISSING_SHOWN])
        new = ", ".join(diff.added[:_MISSING_SHOWN])
        typer.echo(
            f"scribe: {len(diff.reduced)} number(s) appear fewer times after cleanup: {listed}"
            + (f"; new after cleanup: {new}" if diff.added else ""),
            err=True,
        )
    _report_fidelity(fidelity)
    typer.echo(str(plan["--out"]))
    if (
        result.truncated_chunks
        or (diff.drifted and not allow_number_drift)
        or (fidelity.moved_words and not allow_speaker_moves)
    ):
        raise typer.Exit(3)


def _report_replies(result: CleanResult) -> None:
    for item in result.malformed_chunks[:_MISSING_SHOWN]:
        # Only a reply stopped at the output limit is helped by smaller chunks.
        advice = "; lower --chunk-words" if item.cause == "max_tokens" else ""
        typer.echo(
            f"scribe: chunk {item.chunk} {_MALFORMED[item.cause]} ({item.cause}); "
            f"its input text is kept uncleaned{advice}",
            err=True,
        )
    if len(result.malformed_chunks) > _MISSING_SHOWN:
        typer.echo(
            f"scribe: {len(result.malformed_chunks) - _MISSING_SHOWN} more chunk(s) kept their "
            "input text uncleaned; the sidecar names each",
            err=True,
        )


def _report_fidelity(fidelity: Fidelity) -> None:
    if fidelity.moved_words:
        listed = "; ".join(
            f'turn {span.turn} {span.from_speaker} to {span.to_speaker} "{span.words}"'
            for span in fidelity.moved_spans[:_MOVES_SHOWN]
        )
        typer.echo(
            f"scribe: {fidelity.moved_words} word(s) moved to another speaker in cleanup: {listed}",
            err=True,
        )
    if fidelity.content_edit_words:
        typer.echo(
            f"scribe: {fidelity.content_edit_words} content word(s) changed in cleanup "
            f"({fidelity.content_edits_per_1000} per 1,000); spans are in the sidecar",
            err=True,
        )


@app.command()
def schema() -> None:
    """Print the `Transcript` JSON schema."""
    typer.echo(json.dumps(Transcript.model_json_schema(), indent=2))


def _voter(path: Path, role: str, *, ordered: bool) -> Transcript:
    """Load one of `vote`'s inputs, insisting on words, in start order if `ordered`."""
    transcript = Transcript.load(path)
    words = transcript.words
    if not words:
        raise InputValidationError(f"{role} {path} has no words")
    # The aligner walks a hypothesis in start order and would misalign it silently.
    if ordered and (index := first_decrease(words)) is not None:
        raise InputValidationError(
            f"{role} {path} word {index} starts at {words[index].start} s, before word "
            f"{index - 1} at {words[index - 1].start} s; word starts must not decrease"
        )
    return transcript


@app.command()
def vote(
    backbone_path: Path = typer.Argument(
        ..., metavar="BACKBONE", help="Transcript whose words are kept, in order, as the slots."
    ),
    primary_path: Path = typer.Argument(
        ...,
        metavar="PRIMARY",
        help="Hypothesis a changed word takes its spelling from, and an inserted word its times.",
    ),
    secondary_path: Path = typer.Argument(
        ..., metavar="SECONDARY", help="The second hypothesis voting with PRIMARY."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Where to write the result. Default: BACKBONE's name ending .voted.json, beside it.",
    ),
) -> None:
    """Vote three transcripts of one recording into one, word by word.

    A BACKBONE word changes only to a token PRIMARY and SECONDARY agree on,
    and keeps its times and speaker. Words both place in a gap between
    BACKBONE's words are inserted, with PRIMARY's times and the speaker of the
    nearest BACKBONE word; 3 or more in a row inside a pause of 2 s or more in
    BACKBONE's words, where it may have dropped another speaker's turn, get no
    speaker ("Speaker ?" in `scribe turns`). Nothing is deleted, fillers are never
    introduced, and a run of only backchannels ("yeah", "okay") is not inserted.

    PRIMARY's and SECONDARY's word starts must not decrease; their speakers
    are ignored. The result has no turns: run `scribe turns` on it next.
    """
    try:
        backbone = _voter(backbone_path, "BACKBONE", ordered=False)
        primary = _voter(primary_path, "PRIMARY", ordered=True)
        secondary = _voter(secondary_path, "SECONDARY", ordered=True)
        destination = sibling(backbone_path, ".voted.json") if out is None else out
        plan = plan_outputs(
            {"--out": destination},
            {"BACKBONE": backbone_path, "PRIMARY": primary_path, "SECONDARY": secondary_path},
        )
        voted = vote_transcripts(backbone, primary, secondary)
        try:
            voted.dump(plan["--out"])
        # ValueError too: `model_dump_json` raises PydanticSerializationError, a
        # ValueError, on text that UTF-8 cannot encode.
        except (OSError, ValueError) as exc:
            raise InputValidationError(
                f"cannot write transcript to {plan['--out']}: {exc}"
            ) from exc
    except AppError as exc:
        _fail(exc)

    typer.echo(str(plan["--out"]))


def _report_fill(fill: Fill) -> None:
    for run in fill.retimed:
        _warn(moved(run))
    for span in fill.filled:
        _warn(
            f"filled {clock(span.start)}-{clock(span.end)} "
            f"({span.words} words from Parakeet, no speaker)"
        )
    for span in fill.unresolved:
        _warn(possible_drop(span.start, span.end))
    _warn(f"cross-check: {describe(fill)}")


@app.command(name="fill")
def fill_command(
    transcript_path: Path = typer.Argument(
        ..., metavar="TRANSCRIPT", help="Transcript whose holes are filled; its words all stay."
    ),
    reference_path: Path = typer.Argument(
        ..., metavar="REFERENCE", help="Transcript of the same audio, as `scribe parakeet` writes."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Where to write the result. Default: TRANSCRIPT's name ending .filled.json.",
    ),
) -> None:
    """Fill holes in a transcript's words with the words a reference heard there.

    First, where 3 or more TRANSCRIPT words in a row are each more than 2 s
    from the REFERENCE word their text lines up with, they move to REFERENCE's
    times, held between the words around them, and each run moved is printed.
    Words said again near where REFERENCE heard them stay, and so does every
    word when TRANSCRIPT's are not in start order. A TRANSCRIPT filled before
    is not re-timed again, even against a different REFERENCE.

    A hole is a stretch of at least 2 s with no word, the start and end of the
    audio included; a word counts for at most its first 2 s. A hole is filled
    when REFERENCE has 3 words or more there, fillers aside, that TRANSCRIPT
    lacks within 8 s of it: every REFERENCE word from the first such word to
    the last goes in, with REFERENCE's times and no speaker. Every TRANSCRIPT
    word is kept, in order, with its text and speaker. A 30 s window where
    REFERENCE has 20 such words and no hole took them is printed as possible
    dropped speech.

    The result has no turns: run `scribe turns` on it next, which labels the
    filled words "Speaker ?" when the others have speakers.
    """
    try:
        transcript = Transcript.load(transcript_path)
        # Its turns would go stale with no words left to rebuild them from.
        if transcript.turns and not transcript.words:
            raise InputValidationError(f"{transcript_path} has turns but no words to fill between")
        reference = Transcript.load(reference_path)
        destination = sibling(transcript_path, ".filled.json") if out is None else out
        plan = plan_outputs(
            {"--out": destination}, {"TRANSCRIPT": transcript_path, "REFERENCE": reference_path}
        )
        filled, fill = fill_holes(transcript, reference)
        try:
            filled.dump(plan["--out"])
        except (OSError, ValueError) as exc:
            raise InputValidationError(
                f"cannot write transcript to {plan['--out']}: {exc}"
            ) from exc
    except AppError as exc:
        _fail(exc)

    _report_fill(fill)
    typer.echo(str(plan["--out"]))


def _report_pick(picking: Picking, model: str) -> None:
    picked = picking.picked
    line = (
        f"scribe: picked the reference's reading at {picked.count('reference')} of "
        f"{len(picked)} disputed spots ({picked.count('unsure')} unsure) with {model}, "
        f"prompt {PICK_PROMPT_VERSION}"
    )
    if guarded := picked.count("guarded"):
        line += (
            f"; {guarded} guarded, keeping the transcript's words where the reading "
            f"picked is {GUARDED_DROP} or more words shorter"
        )
    if restored := picked.count("restored"):
        line += (
            f"; {restored} restored, putting in the reference's words where they are "
            f"{RESTORED_ADD} or more words longer than the transcript's"
        )
    typer.echo(line, err=True)
    if picking.failed:
        listed = ", ".join(str(chunk.index) for chunk in picking.failed)
        spots = sum(len(chunk.spots) for chunk in picking.failed)
        typer.echo(
            f"scribe: the pick failed on {len(picking.failed)} of {len(picking.chunks)} chunks "
            f"({listed}); their {spots} spots keep the transcript's words",
            err=True,
        )


@app.command(name="pick")
def pick_command(
    transcript_path: Path = typer.Argument(
        ..., metavar="TRANSCRIPT", help="Transcript whose words stay except where REFERENCE's win."
    ),
    reference_path: Path = typer.Argument(
        ..., metavar="REFERENCE", help="Transcript of the same audio, as `scribe parakeet` writes."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Where to write the result. Default: TRANSCRIPT's name ending .picked.json.",
    ),
    model: str = typer.Option(DEFAULT_PICK_MODEL, "--model", help="Model the pick is asked for."),
    context: str | None = typer.Option(
        None, "--context", help="Background on the recording. Never added to the transcript."
    ),
) -> None:
    """Pick, where two transcripts of one recording disagree, which reading was said.

    TRANSCRIPT and REFERENCE are aligned word by word. A disputed spot is where
    they differ in what was said, not only in how it was written: numbers,
    contractions, fillers, repeats, spacing and accents aside, and neither
    side fillers alone. Words only one of them heard are left to `scribe fill`.

    A model reads the conversation around each spot and picks one of the two
    readings, or neither when unsure; it never writes words of its own. It is
    asked through the `claude` CLI on PATH, on its own subscription auth, one
    call per ~1,500 words, six at a time: the words, not the audio, go to
    Anthropic. --context gives it background, such as who was at the
    recording; a reading that matches a name or term in it, or sounds like
    one, is more likely picked. The background is never added to the words.

    Where REFERENCE's reading is picked, its words replace TRANSCRIPT's there,
    each with REFERENCE's times held between the words around it and the
    speaker of the nearest word replaced, unless they are 5 or more fewer
    than TRANSCRIPT's: that pick is guarded and TRANSCRIPT's words stay, as a
    wrong removal loses speech. Where REFERENCE's reading has 10 or more words
    more than TRANSCRIPT's, its words go in the same way whatever the model
    picks: the spot is restored, as keeping the shorter loses speech. Every
    other word stays as it was.
    The result records each spot and its pick in its engine params, and has
    no turns: run `scribe turns` on it next.

    Both inputs' word starts must not decrease. Exit 2 is a bad input, inputs
    that record different audio sha256, a TRANSCRIPT picked before, an
    unwritable output, or no usable claude CLI, found before any call. Exit 4
    means every call failed; the result is written anyway, with TRANSCRIPT's
    words.
    """
    # stdout carries only the output path; unconfigured, structlog prints there.
    configure()
    try:
        transcript = _voter(transcript_path, "TRANSCRIPT", ordered=True)
        # Its spots would compare REFERENCE with words REFERENCE already gave it.
        if "pick_record" in transcript.engine.params:
            raise InputValidationError(f"TRANSCRIPT {transcript_path} was picked already")
        reference = _voter(reference_path, "REFERENCE", ordered=True)
        ours, theirs = transcript.source.sha256, reference.source.sha256
        if ours and theirs and ours != theirs:
            raise InputValidationError(
                f"REFERENCE {reference_path} was made from other audio than TRANSCRIPT "
                f"{transcript_path}: their sha256 differ"
            )
        destination = sibling(transcript_path, ".picked.json") if out is None else out
        plan = plan_outputs(
            {"--out": destination}, {"TRANSCRIPT": transcript_path, "REFERENCE": reference_path}
        )
        # Before the calls, which are paid for.
        prove_writable(plan)
        backend = ClaudeCliBackend(model=model, disable_tools=True)
        backend.resolve()
        picking = pick_readings(transcript, reference, backend, context=context)
        try:
            picking.transcript.dump(plan["--out"])
        except (OSError, ValueError) as exc:
            raise InputValidationError(
                f"cannot write transcript to {plan['--out']}: {exc}"
            ) from exc
    except AppError as exc:
        _fail(exc)

    _report_pick(picking, backend.model)
    typer.echo(str(plan["--out"]))
    if picking.chunks and len(picking.failed) == len(picking.chunks):
        raise typer.Exit(_EXIT_NO_CHUNK)


@app.command()
def parakeet(
    audio_path: Path = typer.Argument(..., metavar="AUDIO", help="Audio file to transcribe."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Where to write the transcript. Default: AUDIO's stem plus .parakeet.json, beside it.",
    ),
) -> None:
    """Transcribe an audio file locally with Parakeet TDT v3, on Apple silicon.

    Runs parakeet-mlx 0.5.2 through `uvx`, which must be on PATH; nothing is
    uploaded and no key is needed. The first run downloads the package and a
    ~1.2 GB model and can take minutes. A run is stopped after 60 minutes.
    The words carry timings but no speakers.
    """
    started = time.monotonic()
    try:
        backend = ParakeetMlx()
        backend.resolve()
        # Before the hash opens it: reading a FIFO or a character device blocks.
        check_input(audio_path, sys.maxsize)
        destination = (
            audio_path.with_name(f"{audio_path.stem}.parakeet.json") if out is None else out
        )
        plan = plan_outputs({"--out": destination}, {"the audio file": audio_path})
        # A first run can take minutes, and an unwritable output would discard it.
        prove_writable(plan)
        source = Source(kind="audio", ref=str(audio_path), sha256=_sha256(audio_path))
        typer.echo(
            "running parakeet-mlx locally; a first run downloads a ~1.2 GB model "
            "and can take minutes",
            err=True,
        )
        transcript = backend.transcribe(audio_path, source=source)
        try:
            transcript.dump(plan["--out"])
        except OSError as exc:
            raise InputValidationError(
                f"cannot write transcript to {plan['--out']}: {exc}"
            ) from exc
    except AppError as exc:
        _fail(exc)

    elapsed = time.monotonic() - started
    typer.echo(f"transcribed {len(transcript.words)} words in {elapsed:.1f} s", err=True)
    typer.echo(str(plan["--out"]))


@app.command()
def gemini(
    audio_path: Path = typer.Argument(..., metavar="AUDIO", help="Audio file to transcribe."),
    anchor_path: Path = typer.Option(
        ...,
        "--anchor",
        metavar="ANCHOR.json",
        help="Word-timed transcript of the same audio, whose word times the Gemini words take.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Where to write the transcript. Default: AUDIO's stem plus .gemini.json, beside it.",
    ),
    max_usd: float = typer.Option(
        gemini_stt.DEFAULT_MAX_USD,
        "--max-usd",
        help="Most the run may spend, in USD.",
    ),
) -> None:
    """Transcribe an audio file with Gemini, timed by another transcript's words.

    Needs GEMINI_API_KEY in the environment, and ffmpeg and ffprobe on PATH.
    The audio goes to Gemini in 600 s chunks starting every 595 s. Each word
    takes its time from the ANCHOR word it aligns with, as a point (end equals
    start), and is a lowercase token without punctuation or speaker: input for
    word voting, not a transcript to read.

    Refused before any call when the expected cost passes --max-usd, and
    stopped with nothing written when an attempt's worst case could pass it.
    The cost goes to stderr and into the transcript's engine params.
    """
    # stdout carries only the output path; unconfigured, structlog prints there.
    configure()
    try:
        anchor = Transcript.load(anchor_path)
        api_key = gemini_stt.resolve_api_key()
        destination = audio_path.with_name(f"{audio_path.stem}.gemini.json") if out is None else out
        plan = plan_outputs(
            {"--out": destination},
            {"the audio file": audio_path, "the --anchor transcript": anchor_path},
        )
        prove_writable(plan)
        transcript = gemini_stt.transcribe(
            audio_path,
            anchor,
            source=Source(kind="audio", ref=str(audio_path), sha256=_sha256(audio_path)),
            api_key=api_key,
            max_usd=max_usd,
        )
        try:
            transcript.dump(plan["--out"])
        except (OSError, ValueError) as exc:
            raise InputValidationError(
                f"cannot write transcript to {plan['--out']}: {exc}"
            ) from exc
    except AppError as exc:
        _fail(exc)

    cost = transcript.engine.params["cost_usd"]
    typer.echo(f"scribe: {len(transcript.words)} words from Gemini for ${cost}", err=True)
    typer.echo(str(plan["--out"]))


@app.command()
def gaps(
    transcript_path: Path = typer.Argument(
        ..., metavar="TRANSCRIPT", help="Transcript whose words are checked."
    ),
    audio_path: Path = typer.Argument(
        ..., metavar="AUDIO", help="The audio the transcript was made from."
    ),
) -> None:
    """Flag holes in a transcript where the audio still sounds like speech.

    A hole is a stretch of at least 6 s with no word, the start and end of the
    audio included; a word counts for at most its first 2 s. A hole is flagged
    when its loudest tenth of 50 ms frames is within 25 dB of the level of the
    transcript's words: speech the engine may have dropped, worth a listen.
    Prints one line per flag: its span, its length, and that level relative to
    speech. A transcript with no words is one hole, flagged on length alone.

    Needs ffmpeg on PATH. Writes no file and sends nothing over the network.
    """
    try:
        transcript = Transcript.load(transcript_path)
        # Before ffmpeg opens it: reading a FIFO or a character device blocks.
        check_input(audio_path, sys.maxsize)
        found = check_gaps(transcript, audio_path)
    except AppError as exc:
        _fail(exc)

    for gap in found.gaps:
        typer.echo(format_gap(gap))
    flags = f"{len(found.gaps)} {'hole' if len(found.gaps) == 1 else 'holes'} flagged"
    if found.speech_db is None:
        typer.echo(f"scribe: {flags}; no words, so no speech level", err=True)
    else:
        typer.echo(f"scribe: {flags}; speech level {found.speech_db:.1f} dB", err=True)
