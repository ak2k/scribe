"""Name the speaker of words no engine attributed, where two readings of the audio agree.

The first reading is the diarizer's: the cluster of its exclusive timeline a
word overlaps most, taken to be the speaker most of that cluster's attributed
words have. The second is a voice match: the stretch around the word, embedded,
against each speaker's centroid, the mean embedding of that speaker's own
speech away from the stretch. A word is named only where both give the same
speaker; every other word stays unattributed.

Every label read here is the caller's settled one per word, never
`Word.speaker`. Everything is pure Python, over embeddings the caller had made
for the intervals `plan` lists.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from scribe.schema import Word

# The rule and all of these were measured together on spans masked out of 4
# transcripts: 96.5% of the masked words named, 0.12% of them wrongly.
# A pause this long ends a speaker's centroid segment.
SEGMENT_PAUSE_S = 0.5
# A shorter segment holds too little of a voice to embed.
SEGMENT_MIN_S = 2.0
# A segment is embedded on this much of its start.
SEGMENT_MAX_S = 10.0
# Segments this near a run are left out of its centroids, so a run is never
# matched against speech it overlaps or borders.
CENTROID_GUARD_S = 10.0
CENTROID_SEGMENTS = 30
WINDOW_S = 3.0
HOP_S = 1.5
# A run's end left more than this past its last whole window gets a window of its own.
END_WINDOW_S = 0.25
# Which of a speaker's segments make a centroid, when it has more than enough.
SHUFFLE_SEED = "45"

# Keeps a window whose end falls on the run's end but for floating-point error.
_SLACK_S = 1e-6

Vector = tuple[float, ...]


@dataclass(frozen=True)
class Speech:
    """A stretch of the diarizer's exclusive timeline and its cluster's label."""

    start: float
    end: float
    label: str


@dataclass(frozen=True)
class Segment:
    """A stretch of one speaker's words, embedded toward that speaker's centroids."""

    speaker: int
    start: float
    # At most SEGMENT_MAX_S after start: the end of what is embedded.
    end: float


@dataclass(frozen=True)
class Run:
    """A maximal list-order run of unattributed words, and the windows it is heard in."""

    first: int
    stop: int
    start: float
    end: float
    windows: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Plan:
    """What to embed to name the unattributed words.

    `intervals` holds every run's windows, run by run, then the segments in
    `requested`: those some run's guard admits, in list order. `order` lists
    each speaker's segments, speakers ascending, in the order its centroids
    take them.
    """

    runs: tuple[Run, ...]
    segments: tuple[Segment, ...]
    order: Mapping[int, tuple[int, ...]]
    requested: tuple[int, ...]
    intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RunNaming:
    """A run's span and length, and how many of its words each speaker was given."""

    start: float
    end: float
    words: int
    # (speaker, words named), speakers ascending; a speaker given none is absent.
    named: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Naming:
    """The speakers with the named words filled in, and what each run gained."""

    speakers: tuple[int | None, ...]
    runs: tuple[RunNaming, ...]

    @property
    def unattributed(self) -> int:
        """Words there were to name."""
        return sum(run.words for run in self.runs)

    @property
    def named(self) -> int:
        """Words that were named."""
        return sum(count for run in self.runs for _, count in run.named)


def can_name(speakers: Sequence[int | None]) -> bool:
    """Whether there is a word to name, and two speakers a voice can be told between."""
    return None in speakers and len(set(speakers) - {None}) > 1


def unattributed_runs(speakers: Sequence[int | None]) -> list[tuple[int, int]]:
    """Return the (first, stop) index range of each maximal run of None speakers."""
    runs: list[tuple[int, int]] = []
    for index, speaker in enumerate(speakers):
        if speaker is not None:
            continue
        if runs and runs[-1][1] == index:
            runs[-1] = (runs[-1][0], index + 1)
        else:
            runs.append((index, index + 1))
    return runs


def windows(start: float, end: float) -> tuple[tuple[float, float], ...]:
    """Cut a run's span into the windows its words are matched in.

    A span of at most WINDOW_S is one window. A longer one takes WINDOW_S
    windows every HOP_S from its start that end by its end, and one more
    ending at its end when the last of those ends more than END_WINDOW_S
    before it.
    """
    if end - start <= WINDOW_S:
        return ((start, end),)
    cut: list[tuple[float, float]] = []
    at = start
    while at + WINDOW_S <= end + _SLACK_S:
        cut.append((at, at + WINDOW_S))
        at += HOP_S
    if cut[-1][1] < end - END_WINDOW_S:
        cut.append((end - WINDOW_S, end))
    return tuple(cut)


def centroid_segments(words: Sequence[Word], speakers: Sequence[int | None]) -> list[Segment]:
    """Split each speaker's list-order runs of words into segments to embed.

    A run breaks where the speaker changes, at a word without one, and where
    a word starts SEGMENT_PAUSE_S or more after the previous word ends. A
    piece spanning SEGMENT_MIN_S or more is kept, cut to its first
    SEGMENT_MAX_S.
    """
    segments: list[Segment] = []
    index = 0
    while index < len(words):
        speaker = speakers[index]
        if speaker is None:
            index += 1
            continue
        last = index
        while (
            last + 1 < len(words)
            and speakers[last + 1] == speaker
            and words[last + 1].start - words[last].end < SEGMENT_PAUSE_S
        ):
            last += 1
        start = words[index].start
        end = max(word.end for word in words[index : last + 1])
        if end - start >= SEGMENT_MIN_S:
            segments.append(Segment(speaker, start, min(end, start + SEGMENT_MAX_S)))
        index = last + 1
    return segments


def shuffled(segments: Sequence[Segment]) -> dict[int, tuple[int, ...]]:
    """Order each speaker's segment indices by one SHUFFLE_SEED draw, speakers ascending."""
    # A fixed seed, so a rerun on the same input names the same words.
    draws = random.Random(SHUFFLE_SEED)  # noqa: S311  # a sample, not a secret
    order: dict[int, tuple[int, ...]] = {}
    for speaker in sorted({segment.speaker for segment in segments}):
        indices = [index for index, segment in enumerate(segments) if segment.speaker == speaker]
        draws.shuffle(indices)
        order[speaker] = tuple(indices)
    return order


def admits(segment: Segment, start: float, end: float) -> bool:
    """Whether a centroid for the span [start, end] may use `segment`."""
    return segment.end <= start - CENTROID_GUARD_S or segment.start >= end + CENTROID_GUARD_S


def plan(words: Sequence[Word], speakers: Sequence[int | None]) -> Plan:
    """List the runs to name and the intervals to embed for them."""
    runs: list[Run] = []
    for first, stop in unattributed_runs(speakers):
        start = words[first].start
        end = max(word.end for word in words[first:stop])
        runs.append(Run(first, stop, start, end, windows(start, end)))
    segments = centroid_segments(words, speakers)
    requested = tuple(
        index
        for index, segment in enumerate(segments)
        if any(admits(segment, run.start, run.end) for run in runs)
    )
    intervals = [window for run in runs for window in run.windows]
    intervals += [(segments[index].start, segments[index].end) for index in requested]
    return Plan(tuple(runs), tuple(segments), shuffled(segments), requested, tuple(intervals))


class Timeline:
    """The diarizer's exclusive timeline, searchable by time."""

    def __init__(self, speech: Sequence[Speech]) -> None:
        """Sort `speech` by start; the diarizer does not promise an order."""
        self._speech = sorted(speech, key=lambda part: part.start)
        self._starts = [part.start for part in self._speech]
        self._reach = max([0.0, *(part.end - part.start for part in self._speech)])

    def label(self, start: float, end: float) -> str | None:
        """Return the label of the speech overlapping [start, end] most.

        A tie goes to the speech that starts first. An instant, start == end,
        takes the latest-starting speech holding it, ends included. None when
        nothing overlaps.
        """
        low = bisect.bisect_left(self._starts, start - self._reach - _SLACK_S)
        near = self._speech[low : bisect.bisect_right(self._starts, end)]
        if start == end:
            holding = [part for part in near if part.start <= start <= part.end]
            return holding[-1].label if holding else None
        best, found = 0.0, None
        for part in near:
            overlap = min(part.end, end) - max(part.start, start)
            if overlap > best:
                best, found = overlap, part.label
        return found


def cluster_speakers(
    labels: Sequence[str | None], speakers: Sequence[int | None]
) -> dict[str, int]:
    """Map each cluster to the speaker most of its attributed words have.

    A tie goes to the speaker whose word the cluster met first.
    """
    tallies: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for label, speaker in zip(labels, speakers, strict=True):
        if label is not None and speaker is not None:
            tallies[label][speaker] += 1
    return {label: tally.most_common(1)[0][0] for label, tally in tallies.items()}


def diarized_speakers(
    words: Sequence[Word], speakers: Sequence[int | None], speech: Sequence[Speech]
) -> list[int | None]:
    """Give every word its cluster's speaker, or None where it has no mapped cluster."""
    timeline = Timeline(speech)
    labels = [timeline.label(word.start, word.end) for word in words]
    mapped = cluster_speakers(labels, speakers)
    return [None if label is None else mapped.get(label) for label in labels]


def unit(vector: Sequence[float] | None) -> Vector | None:
    """Scale `vector` to length 1; None when there is no vector or it has no direction."""
    if vector is None:
        return None
    length = math.sqrt(math.sumprod(vector, vector))
    if not length or not math.isfinite(length):
        return None
    return tuple(value / length for value in vector)


def centroids(
    segments: Sequence[Segment],
    order: Mapping[int, Sequence[int]],
    vectors: Sequence[Vector | None],
    start: float,
    end: float,
) -> dict[int, Vector]:
    """Build each speaker's centroid for the span [start, end].

    A centroid is the unit mean of the first CENTROID_SEGMENTS of the
    speaker's segments, in `order`, that have a vector and that `admits`
    lets it use. A speaker with none has no centroid.

    Args:
        segments: Every centroid segment, in list order.
        order: Each speaker's segment indices in the order they are taken.
        vectors: A unit vector per segment, or None where it has none.
        start: The span's start.
        end: The span's end.

    """
    means: dict[int, Vector] = {}
    for speaker, indices in order.items():
        picked = [
            vector
            for index in indices
            if (vector := vectors[index]) is not None and admits(segments[index], start, end)
        ][:CENTROID_SEGMENTS]
        if not picked:
            continue
        mean = unit([sum(column) / len(picked) for column in zip(*picked, strict=True)])
        if mean is not None:
            means[speaker] = mean
    return means


def nearest(vector: Vector | None, means: Mapping[int, Vector]) -> int | None:
    """Return the speaker whose centroid is most like `vector`, the first on a tie.

    None without a vector, or with fewer than two centroids: a match against
    one speaker alone cannot disagree with the diarizer.
    """
    if vector is None or len(means) <= 1:
        return None
    best, found = -math.inf, None
    for speaker, mean in means.items():
        score = math.sumprod(vector, mean)
        if score > best:
            best, found = score, speaker
    return found


def _window_of(run: Run, word: Word) -> int:
    middle = (word.start + word.end) / 2
    centers = [(low + high) / 2 for low, high in run.windows]
    return min(range(len(centers)), key=lambda at: abs(centers[at] - middle))


def name_unattributed(
    planned: Plan,
    words: Sequence[Word],
    speakers: Sequence[int | None],
    speech: Sequence[Speech],
    embeddings: Sequence[Sequence[float] | None],
) -> Naming:
    """Name each unattributed word whose cluster's speaker and heard speaker agree.

    A word is heard in the window of its run whose center is nearest its
    midpoint, the first on a tie, as the speaker `nearest` finds for that
    window among the run's `centroids`.

    Args:
        planned: The `plan` for these words and speakers.
        words: The transcript's words.
        speakers: One settled speaker per word, None where unattributed.
        speech: The diarizer's exclusive timeline.
        embeddings: One embedding per interval of `planned`, None where none was made.

    Raises:
        ValueError: `embeddings` does not hold one entry per planned interval.

    """
    if len(embeddings) != len(planned.intervals):
        raise ValueError(
            f"{len(embeddings)} embeddings for {len(planned.intervals)} planned intervals"
        )
    diarized = diarized_speakers(words, speakers, speech)
    heard_at = sum(len(run.windows) for run in planned.runs)
    vectors: list[Vector | None] = [None] * len(planned.segments)
    for slot, index in enumerate(planned.requested):
        vectors[index] = unit(embeddings[heard_at + slot])
    named = list(speakers)
    runs: list[RunNaming] = []
    offset = 0
    for run in planned.runs:
        means = centroids(planned.segments, planned.order, vectors, run.start, run.end)
        heard = [
            nearest(unit(embedding), means)
            for embedding in embeddings[offset : offset + len(run.windows)]
        ]
        offset += len(run.windows)
        given: Counter[int] = Counter()
        for index in range(run.first, run.stop):
            speaker = diarized[index]
            if speaker is not None and speaker == heard[_window_of(run, words[index])]:
                named[index] = speaker
                given[speaker] += 1
        counts = tuple(sorted(given.items()))
        runs.append(RunNaming(run.start, run.end, run.stop - run.first, counts))
    return Naming(tuple(named), tuple(runs))
