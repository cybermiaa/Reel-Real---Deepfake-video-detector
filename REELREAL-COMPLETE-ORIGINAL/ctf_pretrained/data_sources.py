"""Dataset scanners: FaceForensics++, DeepFakeDetection (DFD), and Celeb-DF v2.

One job: turn a downloaded dataset directory into a uniform list of
VideoRecords carrying a *correct group key*, so no split can ever put a fake
and the real video it was made from on opposite sides.

Group keys, per dataset:

  FF++    A fake named `000_003.mp4` is built from originals `000.mp4`
          (target) and `003.mp4` (source). All four of 000, 003, 000_003 and
          003_000 must move together, so pairs are merged with union-find.
          FF++ pairs 1000 videos into 500 disjoint couples, which yields ~500
          groups -- fine granularity.

  DFD     Same idea, actors instead of YouTube clips (`01_02__scene__HASH`).
          28 actors are swapped fairly densely, so union-find can collapse to
          one blob; _group_pairs falls back to target-only grouping and says so.

  Celeb-DF  `id0_id1_0000.mp4` swaps identity id1 onto id0. Celeb-DF swaps
          *every pair* of its 59 subjects, so union-find would merge all of
          them into a single group. Grouping is therefore by TARGET identity.
          This matters little in practice because Celeb-DF is used here as a
          cross-dataset test set, not for training.

Labels are always derived from the directory the file sits in, never from a
numeric column in a list file. Directory position is unambiguous; a label
convention read backwards would silently invert every metric in the project.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
COMPRESSIONS = ["raw", "c23", "c40"]

DATASETS = ["ffpp", "dfd", "celebdf"]


@dataclass
class VideoRecord:
    path: str
    label: int          # 0 = real, 1 = fake
    video_id: str
    group_id: str
    dataset: str
    method: str         # "real", "Deepfakes", "Celeb-synthesis", ...
    compression: str
    in_official_test: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Union-find
# --------------------------------------------------------------------------
class _UF:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _group_pairs(pairs: Sequence[Tuple[str, str]], singles: Iterable[str]
                 ) -> Tuple[Dict[str, str], str, bool]:
    """Merge identity pairs into groups.

    Returns (identity -> group_id, strategy, is_leaky).

    Union-find is the safe grouping and is used whenever it leaves anything to
    split with. The per-identity fallback is LEAKY -- it puts a fake and the
    original it was made from in different groups -- so it fires only when
    union-find collapses everything into a single group, which makes splitting
    impossible otherwise. That happens on datasets that swap every pair of
    subjects (DFD, Celeb-DF). It is never correct for FF++, whose 500 disjoint
    couples always yield 500 groups.

    An earlier version fell back whenever union-find produced fewer than five
    groups. That silently chose the leaky grouping on any small subset -- the
    precise failure this module exists to prevent.
    """
    uf = _UF()
    for s in singles:
        uf.find(s)
    for a, b in pairs:
        uf.union(a, b)

    mapping = {x: uf.find(x) for x in uf.parent}
    if len(set(mapping.values())) >= 2:
        return mapping, "union-find over source/target pairs", False
    return ({x: x for x in mapping},
            "per-identity FALLBACK -- union-find collapsed to one group", True)


def _videos_in(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)


# --------------------------------------------------------------------------
# FaceForensics++
# --------------------------------------------------------------------------
_FFPP_FAKE_RE = re.compile(r"^(?P<target>\d+)_(?P<source>\d+)$")


def scan_ffpp(root, compression: str = "c23", methods: Optional[Sequence[str]] = None,
              include_original: bool = True, verbose: bool = True) -> List[VideoRecord]:
    root = Path(root)
    methods = list(methods) if methods else list(FFPP_METHODS)

    orig_dir = root / "original_sequences" / "youtube" / compression / "videos"
    originals = _videos_in(orig_dir)

    fakes: List[Tuple[Path, str, str, str]] = []  # path, method, target, source
    for m in methods:
        for p in _videos_in(root / "manipulated_sequences" / m / compression / "videos"):
            mt = _FFPP_FAKE_RE.match(p.stem)
            if mt:
                fakes.append((p, m, mt.group("target"), mt.group("source")))

    if not originals and not fakes:
        raise FileNotFoundError(
            f"No FF++ videos found under {root} at compression '{compression}'.\n"
            f"Expected e.g. {orig_dir}\n"
            "Check --root and --compression, and that the download finished.")

    mapping, strategy, leaky = _group_pairs(
        [(t, s) for _, _, t, s in fakes],
        [p.stem for p in originals])

    records: List[VideoRecord] = []
    if include_original:
        for p in originals:
            records.append(VideoRecord(
                path=str(p), label=0, video_id=p.stem,
                group_id=mapping.get(p.stem, p.stem),
                dataset="ffpp", method="real", compression=compression))
    for p, m, tgt, src in fakes:
        records.append(VideoRecord(
            path=str(p), label=1, video_id=f"{m}__{p.stem}",
            group_id=mapping.get(tgt, tgt),
            dataset="ffpp", method=m, compression=compression))

    if verbose:
        n_groups = len({r.group_id for r in records})
        print(f"[ffpp] {len(originals)} real, {len(fakes)} fake "
              f"({len(methods)} methods, {compression}), "
              f"{n_groups} groups, grouping: {strategy}")
    if leaky:
        print("[ffpp] WARNING: source/target pairs collapsed to a single group, "
              "so grouping fell back to per-identity. On the full FF++ release "
              "this never happens (500 disjoint couples). Any split built from "
              "this scan can put a fake and its own original on opposite sides.")
    return records


# --------------------------------------------------------------------------
# DeepFakeDetection (Google/Jigsaw actors), shipped by the same downloader
# --------------------------------------------------------------------------
_DFD_FAKE_RE = re.compile(r"^(?P<target>\d+)_(?P<source>\d+)__")
_DFD_REAL_RE = re.compile(r"^(?P<actor>\d+)__")


def scan_dfd(root, compression: str = "c23", verbose: bool = True) -> List[VideoRecord]:
    root = Path(root)
    real_dir = root / "original_sequences" / "actors" / compression / "videos"
    fake_dir = root / "manipulated_sequences" / "DeepFakeDetection" / compression / "videos"

    originals = _videos_in(real_dir)
    fakes = _videos_in(fake_dir)
    if not originals and not fakes:
        raise FileNotFoundError(
            f"No DFD videos under {root} at '{compression}'. Download with "
            "-d DeepFakeDetection and -d DeepFakeDetection_original.")

    pairs, singles = [], []
    for p in originals:
        m = _DFD_REAL_RE.match(p.stem)
        singles.append(m.group("actor") if m else p.stem)
    fake_actors = []
    for p in fakes:
        m = _DFD_FAKE_RE.match(p.stem)
        if m:
            pairs.append((m.group("target"), m.group("source")))
            fake_actors.append(m.group("target"))
        else:
            fake_actors.append(p.stem)

    mapping, strategy, leaky = _group_pairs(pairs, singles)

    records = []
    for p in originals:
        m = _DFD_REAL_RE.match(p.stem)
        actor = m.group("actor") if m else p.stem
        records.append(VideoRecord(
            path=str(p), label=0, video_id=p.stem,
            group_id=mapping.get(actor, actor),
            dataset="dfd", method="real", compression=compression))
    for p, actor in zip(fakes, fake_actors):
        records.append(VideoRecord(
            path=str(p), label=1, video_id=p.stem,
            group_id=mapping.get(actor, actor),
            dataset="dfd", method="DeepFakeDetection", compression=compression))

    if verbose:
        print(f"[dfd] {len(originals)} real, {len(fakes)} fake ({compression}), "
              f"{len({r.group_id for r in records})} groups, grouping: {strategy}")
    if leaky:
        print("[dfd] NOTE: DFD swaps its 28 actors densely, so pair-merging "
              "collapses to one group and grouping is per-actor instead. Use DFD "
              "as a test-only set, where this does not matter.")
    return records


# --------------------------------------------------------------------------
# Celeb-DF v2
# --------------------------------------------------------------------------
_CELEB_REAL_RE = re.compile(r"^(?P<id>id\d+)_\d+$")
_CELEB_FAKE_RE = re.compile(r"^(?P<target>id\d+)_(?P<source>id\d+)_\d+$")

# Directory -> label. This is the ground truth; the numeric column in
# List_of_testing_videos.txt is only cross-checked against it.
_CELEB_DIRS = {"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1}


def _find_celeb_root(root: Path) -> Path:
    """Tolerate an extra nesting level from unzipping."""
    if any((root / d).is_dir() for d in _CELEB_DIRS):
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if any((child / d).is_dir() for d in _CELEB_DIRS):
            return child
    return root


def load_celebdf_test_list(root) -> Tuple[set, Optional[str]]:
    """Read List_of_testing_videos.txt -> (set of relative paths, convention note).

    The numeric column is checked against directory-derived truth so a
    misread convention surfaces as a warning instead of inverted metrics.
    """
    root = _find_celeb_root(Path(root))
    f = root / "List_of_testing_videos.txt"
    if not f.exists():
        return set(), None

    entries, agree_one_is_real, total = set(), 0, 0
    for line in f.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        rel = parts[-1].replace("\\", "/")
        entries.add(rel)
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            num = int(parts[0])
            truth_fake = rel.split("/")[0] == "Celeb-synthesis"
            total += 1
            if (num == 1) == (not truth_fake):
                agree_one_is_real += 1

    note = None
    if total:
        frac = agree_one_is_real / total
        if frac > 0.95:
            note = "file's numeric column uses 1=real (labels taken from directory anyway)"
        elif frac < 0.05:
            note = "file's numeric column uses 1=fake (labels taken from directory anyway)"
        else:
            note = (f"numeric column inconsistent with directories "
                    f"({frac:.0%} match 1=real) -- ignoring it, using directories")
    return entries, note


def scan_celebdf(root, testing_list_only: bool = False, verbose: bool = True
                 ) -> List[VideoRecord]:
    root = _find_celeb_root(Path(root))
    test_set, note = load_celebdf_test_list(root)

    records = []
    for dname, label in _CELEB_DIRS.items():
        for p in _videos_in(root / dname):
            rel = f"{dname}/{p.name}"
            if testing_list_only and test_set and rel not in test_set:
                continue
            if label == 1:
                m = _CELEB_FAKE_RE.match(p.stem)
                group = m.group("target") if m else p.stem
            else:
                m = _CELEB_REAL_RE.match(p.stem)
                group = m.group("id") if m else f"yt_{p.stem}"
            records.append(VideoRecord(
                path=str(p), label=label, video_id=f"{dname}__{p.stem}",
                group_id=group, dataset="celebdf", method=dname,
                compression="mpeg4", in_official_test=rel in test_set))

    if not records:
        raise FileNotFoundError(
            f"No Celeb-DF videos under {root}. Expected subfolders: "
            f"{list(_CELEB_DIRS)}")

    if verbose:
        n_fake = sum(r.label for r in records)
        print(f"[celebdf] {len(records)-n_fake} real, {n_fake} fake, "
              f"{len(test_set)} in official test list"
              + (f"; {note}" if note else ""))
        if testing_list_only and not test_set:
            print("[celebdf] WARNING: --official-test-only requested but "
                  "List_of_testing_videos.txt was not found; using everything.")
    return records


# --------------------------------------------------------------------------
# Dispatch and reporting
# --------------------------------------------------------------------------
def scan(dataset: str, root, compression: str = "c23",
         methods: Optional[Sequence[str]] = None,
         testing_list_only: bool = False, verbose: bool = True) -> List[VideoRecord]:
    dataset = dataset.lower()
    if dataset == "ffpp":
        return scan_ffpp(root, compression, methods, verbose=verbose)
    if dataset == "dfd":
        return scan_dfd(root, compression, verbose=verbose)
    if dataset == "celebdf":
        return scan_celebdf(root, testing_list_only, verbose=verbose)
    raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")


def summarize(records: Sequence[VideoRecord]) -> str:
    from collections import Counter
    n = len(records)
    n_fake = sum(r.label for r in records)
    groups = len({r.group_id for r in records})
    by_method = Counter(r.method for r in records)
    lines = [f"  videos {n}   real {n-n_fake}   fake {n_fake} "
             f"({100.0*n_fake/n:.1f}%)   groups {groups}"]
    for m, c in sorted(by_method.items()):
        lines.append(f"    {m:<22} {c}")
    return "\n".join(lines)


def to_dataframe(records: Sequence[VideoRecord]):
    import pandas as pd
    return pd.DataFrame([r.as_dict() for r in records])


def check_group_disjoint(*record_groups: Sequence[VideoRecord]) -> None:
    """Raise if any two collections share a group_id."""
    sets = [{r.group_id for r in g} for g in record_groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(
                    f"LEAK: collections {i} and {j} share {len(overlap)} groups "
                    f"(e.g. {sorted(overlap)[:5]})")
