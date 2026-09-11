"""Fetch FaceForensics++, DFD and Celeb-DF v2 into the Colab session disk.

Handles the two things that trip people up on Colab:

  * The official FF++ downloader blocks on `input()` waiting for you to accept
    the terms of use. In a notebook that hangs forever with no output. We feed
    it a newline, which is equivalent to pressing a key at that prompt -- you
    are still accepting the TOS, so read them first (the script prints the URL).

  * The official Celeb-DF v2 link is a SINGLE ZIP FILE on Google Drive, not a
    folder of loose videos -- clicking it just opens Drive's zip preview.
    "Add shortcut to Drive" works on a file exactly as it does on a folder, so
    the same drive-mount route handles it: the zip is copied to local session
    disk (unzipping straight off the Drive FUSE mount is slow) and then
    extracted there. A Kaggle mirror is also supported as a fallback.

Examples
--------
  # FF++ c23, all five manipulations plus originals (the training set)
  python download_data.py ffpp --script faceforensics_download_v4.py \\
      --root /content/data/ffpp --compression c23

  # Same videos at c40 -- a real codec robustness test set, for free
  python download_data.py ffpp --script faceforensics_download_v4.py \\
      --root /content/data/ffpp --compression c40 --num-videos 200

  # DFD (Google/Jigsaw actors), free second cross-dataset point
  python download_data.py dfd --script faceforensics_download_v4.py \\
      --root /content/data/ffpp --compression c23

  # Celeb-DF v2, from a Drive shortcut to the official zip file
  python download_data.py celebdf --root /content/data/celebdf \\
      --via drive --drive-dir "/content/drive/MyDrive/Celeb-DF-v2.zip"

  # Celeb-DF v2 from a Kaggle mirror
  python download_data.py celebdf --root /content/data/celebdf --via kaggle
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import config

FFPP_DATASETS = ["original", "Deepfakes", "Face2Face", "FaceSwap",
                 "NeuralTextures", "FaceShifter"]
DFD_DATASETS = ["DeepFakeDetection_original", "DeepFakeDetection"]

_CELEB_SUBDIRS = ["Celeb-real", "YouTube-real", "Celeb-synthesis"]

# Community mirror. Verify it matches the official release before relying on
# it for reported numbers; the official Drive link is the source of truth.
CELEBDF_KAGGLE_SLUG = "reubensuju/celeb-df-v2"


def run_ffpp(script: Path, root: Path, dataset: str, compression: str,
             num_videos, server: str) -> bool:
    """Invoke the official downloader, auto-answering its TOS prompt."""
    cmd = [sys.executable, str(script), str(root),
           "-d", dataset, "-c", compression, "-t", "videos", "--server", server]
    if num_videos:
        cmd += ["-n", str(num_videos)]
    print(f"\n$ {' '.join(cmd)}")
    # The script does `input('')` after printing the TOS URL; a newline is the
    # documented way to continue.
    proc = subprocess.run(cmd, input="\n", text=True)
    if proc.returncode != 0:
        print(f"  FAILED ({dataset}, exit {proc.returncode})")
        return False
    return True


def cmd_ffpp(args) -> int:
    script = Path(args.script)
    if not script.exists():
        raise SystemExit(
            f"FF++ downloader not found: {script}\n"
            "Get it by filling the form at "
            "https://github.com/ondyari/FaceForensics -- they email you the script.")
    datasets = args.datasets or FFPP_DATASETS
    args.root.mkdir(parents=True, exist_ok=True)
    print(f"FF++ -> {args.root}  compression={args.compression}")
    print("You are accepting the FaceForensics terms of use. Read them at the "
          "URL the downloader prints.")
    ok = [d for d in datasets if run_ffpp(script, args.root, d, args.compression,
                                          args.num_videos, args.server)]
    print(f"\ndownloaded {len(ok)}/{len(datasets)}: {ok}")
    return 0 if len(ok) == len(datasets) else 1


def cmd_dfd(args) -> int:
    args.datasets = DFD_DATASETS
    return cmd_ffpp(args)


def _unpack_all(root: Path) -> None:
    """Extract every zip directly under root, then remove it.

    Removing the zip after a successful extract matters here specifically:
    Celeb-DF's zip is tens of GB, and leaving a copy sitting next to its own
    extracted contents can quietly fill the session disk.
    """
    for z in sorted(root.glob("*.zip")):
        print(f"  unzip {z.name} (this can take a while for a large archive)")
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(root)
            z.unlink()
        except zipfile.BadZipFile:
            print(f"    skipped (not a valid zip): {z.name}")


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy the Celeb-DF folders from a mounted Drive path to local disk.

    Worth the wait: crop extraction reads every video once, and doing that
    through the Drive FUSE layer is far slower than copying first.
    """
    import shutil as sh
    wanted = [d for d in _CELEB_SUBDIRS if (src / d).is_dir()]
    for d in wanted:
        target = dst / d
        if target.exists() and any(target.iterdir()):
            print(f"  {d}: already present, skipping")
            continue
        print(f"  copying {d} ...")
        sh.copytree(src / d, target, dirs_exist_ok=True)
    for extra in ("List_of_testing_videos.txt",):
        if (src / extra).exists():
            sh.copy2(src / extra, dst / extra)
            print(f"  copied {extra}")
        else:
            print(f"  NOTE: {extra} not found; --official-test-only will be "
                  "unavailable and the full set will be used instead.")


def _fetch_from_drive(src: Path, dst: Path) -> None:
    """Handle every shape --drive-dir can point at.

    The official Celeb-DF v2 link is a single zip file, not a folder -- Drive's
    web preview shows its contents when you open the link, which looks like a
    folder but isn't one. "Add shortcut to Drive" works on a file exactly as it
    does on a folder, so --drive-dir may point at:

      * a .zip file directly
      * a folder containing one or more .zip files (e.g. split archives)
      * a folder that is already extracted (Celeb-real/, etc. present)
    """
    import shutil as sh

    if src.is_file():
        if src.suffix.lower() != ".zip":
            raise SystemExit(f"{src} is a file but not a .zip -- unexpected "
                             "for Celeb-DF. Check the shortcut points at the "
                             "right item.")
        zips = [src]
    else:
        zips = sorted(src.glob("*.zip"))

    if zips:
        for z in zips:
            target = dst / z.name
            if target.exists() and target.stat().st_size == z.stat().st_size:
                print(f"  {z.name}: already copied, skipping")
                continue
            print(f"  copying {z.name} ({z.stat().st_size / 1e9:.1f} GB) from "
                  "Drive to local disk -- unzipping directly off the Drive "
                  "mount would be far slower")
            sh.copy2(z, target)
        for extra in ("List_of_testing_videos.txt",):
            p = src if src.is_dir() else src.parent
            if (p / extra).exists():
                sh.copy2(p / extra, dst / extra)
                print(f"  copied {extra}")
        _unpack_all(dst)
        return

    if src.is_dir() and any((src / d).is_dir() for d in _CELEB_SUBDIRS):
        _copy_tree(src, dst)
        return

    raise SystemExit(
        f"Nothing usable found at {src}.\n"
        f"Expected a .zip file, a folder of .zip files, or already-extracted "
        f"folders ({_CELEB_SUBDIRS}). If the shortcut created a nested folder, "
        "point --drive-dir one level deeper -- check with: "
        f"!ls \"{src}\"")


def cmd_celebdf(args) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.via == "drive":
        if not args.drive_dir:
            raise SystemExit(
                "--via drive needs --drive-dir, the mounted path to the Celeb-DF "
                "zip or folder, e.g. /content/drive/MyDrive/Celeb-DF-v2.zip\n"
                "Open the Drive link the authors sent, then right-click the "
                "item > Organise > Add shortcut to Drive > My Drive.")
        src = Path(args.drive_dir)
        if not src.exists():
            raise SystemExit(
                f"--drive-dir not found: {src}\n"
                "Is Drive mounted? Run drive.mount('/content/drive') first, then "
                "check the exact name with: !ls /content/drive/MyDrive")
        print(f"Celeb-DF v2 from mounted Drive: {src}")
        _fetch_from_drive(src, root)

    elif args.via == "kaggle":
        slug = args.kaggle_slug
        print(f"Celeb-DF v2 from Kaggle mirror: {slug}")
        print("Verify this mirror against the official release before "
              "reporting numbers from it.")
        proc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(root), "--unzip"])
        if proc.returncode != 0:
            print("kaggle download failed. Is kaggle.json in place, and have you "
                  "accepted the dataset's terms on its Kaggle page?")
            return 1
    else:
        if not args.gdrive_id:
            raise SystemExit(
                "--via gdrive needs --gdrive-id, the folder id from the link the "
                "Celeb-DF authors sent you (the part after 'id=').")
        try:
            import gdown  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"],
                           check=True)
        print(f"Celeb-DF v2 from Google Drive folder {args.gdrive_id}")
        print("WARNING: gdown --folder downloads at most 50 files per folder, and "
              "Celeb-DF v2 holds thousands. This will almost certainly give you an "
              "incomplete dataset.\n"
              "Prefer --via drive (add a shortcut to your own Drive, then copy) "
              "or --via kaggle.")
        proc = subprocess.run(
            [sys.executable, "-m", "gdown", "--folder", "--remaining-ok",
             "-O", str(root), args.gdrive_id])
        if proc.returncode != 0:
            print("gdown failed (Drive quota is the usual cause). "
                  "Use --via drive or --via kaggle.")
            return 1

    _unpack_all(root)

    import data_sources as ds
    try:
        records = ds.scan_celebdf(root)
        print(ds.summarize(records))
    except FileNotFoundError as exc:
        print(f"\nDownloaded, but the layout is not what we expect:\n  {exc}")
        print("Expected Celeb-real/, YouTube-real/, Celeb-synthesis/ under --root.")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("ffpp", "dfd"):
        p = sub.add_parser(name)
        p.add_argument("--script", default="faceforensics_download_v4.py",
                       help="the downloader the FF++ authors emailed you")
        p.add_argument("--root", type=Path, default=config.FFPP_ROOT)
        p.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
        p.add_argument("--num-videos", type=int, default=None,
                       help="limit per manipulation; use ~200 for a fast first pass")
        # EU2 is the default because as of 2026 the FaceForensics authors state
        # it is the only server still running. EU and CA are kept as options in
        # case that changes back.
        p.add_argument("--server", default="EU2", choices=["EU", "EU2", "CA"])
        p.add_argument("--datasets", nargs="*", default=None)
        p.set_defaults(func=cmd_ffpp if name == "ffpp" else cmd_dfd)

    p = sub.add_parser("celebdf")
    p.add_argument("--root", type=Path, default=config.CELEBDF_ROOT)
    p.add_argument("--via", default="drive", choices=["drive", "kaggle", "gdrive"],
                   help="drive: copy from a mounted Drive folder (most reliable "
                        "for the official release); kaggle: community mirror; "
                        "gdrive: gdown, limited to 50 files per folder")
    p.add_argument("--drive-dir", default=None,
                   help="mounted path to the Celeb-DF folder, used with --via drive")
    p.add_argument("--kaggle-slug", default=CELEBDF_KAGGLE_SLUG)
    p.add_argument("--gdrive-id", default=None)
    p.set_defaults(func=cmd_celebdf)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
