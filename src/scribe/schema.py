"""The one `Transcript` schema every stage reads and writes.

Stages never mutate a `Transcript`: a stage that fills a field returns
`transcript.model_copy(update={...})` so an earlier stage's output stays
valid for re-running a later one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, ValidationError, model_validator

from scribe.errors import ExternalServiceError, InputValidationError

if TYPE_CHECKING:
    from pathlib import Path

# json.loads and pydantic's JSON parser both accept the non-RFC literals Infinity
# and NaN, which pydantic then serializes as null: a transcript this schema
# cannot load back.
FiniteFloat = Annotated[float, AllowInfNan(False)]


def _is_none(value: object) -> bool:
    return value is None


class Word(BaseModel):
    """One recognized word with its span, and its raw diarization id when present."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    start: FiniteFloat
    end: FiniteFloat
    speaker: int | None = None
    # Left out of the JSON when None, so a one-track transcript dumps as it always has.
    track: int | None = Field(default=None, exclude_if=_is_none)


class Turn(BaseModel):
    """A contiguous stretch of speech by one speaker.

    `speaker` is a display label ("Speaker 1"), not the API's diarization id —
    a later relabel stage replaces it with a real name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    speaker: str
    start: FiniteFloat
    end: FiniteFloat
    text: str


class Source(BaseModel):
    """Where the audio came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["audio", "krisp", "other"]
    ref: str
    sha256: str | None = None


class Track(BaseModel):
    """One side of a call recorded on its own, as merged into a transcript with the other."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["mic", "app"]
    # The mic side is one person, shown by this name; the app side's speakers are ranked.
    label: str | None = None
    source: Source
    transcript_sha256: str

    @model_validator(mode="after")
    def _named_mic(self) -> Self:
        if self.role == "mic" and self.label is None:
            raise ValueError("a mic track needs a label")
        return self


class Engine(BaseModel):
    """Which transcription engine produced the text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    model: str | None = None
    # Float first: load reports a union's first failing member, and a NaN is a
    # bad number, not a bad string. Smart-mode matching keeps each JSON type.
    params: dict[str, FiniteFloat | int | bool | str] = Field(default_factory=dict)


class Transcript(BaseModel):
    """The artifact every stage reads and writes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source: Source
    engine: Engine
    language: str | None = None
    duration: FiniteFloat | None = None
    text: str
    words: list[Word] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    # A merged call's sides, which each word's `track` indexes.
    tracks: list[Track] | None = Field(default=None, exclude_if=_is_none)

    @model_validator(mode="after")
    def _words_fit_tracks(self) -> Self:
        if self.tracks is None:
            if any(word.track is not None for word in self.words):
                raise ValueError("a word names a track, but the transcript lists none")
            return self
        last = len(self.tracks) - 1
        if any(word.track is None or not 0 <= word.track <= last for word in self.words):
            raise ValueError(f"every word of a merged transcript needs a track, 0 to {last}")
        # Turns label a mic track's words by their speaker id, so that id must be
        # the track's alone.
        for index, track in enumerate(self.tracks):
            if track.role != "mic":
                continue
            held = {word.speaker for word in self.words if word.track == index}
            elsewhere = {word.speaker for word in self.words if word.track != index}
            if len(held) > 1 or held & elsewhere:
                raise ValueError(
                    f"mic track {index}'s words must share one speaker id, held by no other track"
                )
        return self

    @classmethod
    def load(cls, path: Path) -> Transcript:
        """Read a transcript JSON file.

        Args:
            path: File written by `dump` (or by hand against this schema).

        Returns:
            The parsed transcript.

        Raises:
            InputValidationError: the path is unreadable, or its contents do
                not validate against this schema.

        """
        # Bytes, not text: pydantic does the decoding, so a non-UTF-8 file arrives
        # as the ValidationError below instead of a stray UnicodeDecodeError.
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise InputValidationError(f"cannot read transcript {path}: {exc}") from exc
        return cls.parse(raw, path)

    @classmethod
    def parse(cls, raw: bytes, path: Path) -> Transcript:
        """Parse a transcript file's bytes, naming `path` in the error.

        Raises:
            InputValidationError: the bytes do not validate against this schema.

        """
        try:
            return cls.model_validate_json(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            raise InputValidationError(
                f"{path} is not a transcript: {first['msg']} (at {where or 'top level'})"
            ) from exc

    def dump(self, path: Path) -> None:
        """Write this transcript as indented JSON with a trailing newline."""
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class XaiWord(BaseModel):
    """One word of an xAI speech-to-text response. `speaker` appears only when diarized."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    start: FiniteFloat
    end: FiniteFloat
    speaker: int | None = None
    # DIVERGE: documented as 0.0-1.0 but entropy-based and read by nothing, so an
    # out-of-range rounding overshoot must not fail the whole transcription. NaN
    # and Infinity stay refused: they are not valid JSON and mean a broken
    # upstream, not rounding.
    confidence: FiniteFloat | None = None


class XaiResponse(BaseModel):
    """An xAI batch speech-to-text response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    language: str | None = None
    duration: FiniteFloat | None = None
    words: list[XaiWord] = Field(default_factory=list)


def from_xai_response(payload: dict[str, object], *, source: Source, engine: Engine) -> Transcript:
    """Build a `Transcript` from a decoded xAI speech-to-text response.

    Args:
        payload: Decoded JSON body of the batch transcription response.
        source: Provenance of the audio the request carried.
        engine: Engine identity to record on the transcript.

    Returns:
        A transcript with `text`, `words` and metadata filled; `turns` is empty
        until the turns stage runs.

    Raises:
        ExternalServiceError: the response does not match the documented shape.

    """
    try:
        parsed = XaiResponse.model_validate(payload)
    except ValidationError as exc:
        raise ExternalServiceError(f"malformed xAI transcription response: {exc}") from exc
    return Transcript(
        source=source,
        engine=engine,
        language=parsed.language,
        duration=parsed.duration,
        text=parsed.text,
        words=[
            Word(text=w.text, start=w.start, end=w.end, speaker=w.speaker) for w in parsed.words
        ],
    )
