"""Batch embedding extraction for all manifests in $VANALENS_CACHE/manifests/.

For every `<name>.jsonl` in `$VANALENS_CACHE/manifests/` without an up-to-date
`$VANALENS_CACHE/emb/<name>.npz`, runs:
    `sys.executable -m vanalens.encoders --manifest ... --out ...`
one after another, logging to `runs/logs/extract_all.log` and printing "ALL DONE" at the end.

`--detach` relaunches this script in the background:
- Windows: `creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`
- POSIX: `start_new_session = True`
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vanalens import config, contracts


def is_up_to_date(manifest_path: Path, emb_path: Path) -> bool:
    """Return True if emb_path exists, is at least as new as manifest_path, and loads cleanly."""
    if not emb_path.exists() or not manifest_path.exists():
        return False
    try:
        if emb_path.stat().st_mtime < manifest_path.stat().st_mtime:
            return False
        _ = contracts.EmbeddingSet.load(emb_path)
        return True
    except Exception:
        return False


def _log(msg: str, log_file: Path) -> None:
    """Write a line to both log_file and stdout."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
    try:
        print(msg, flush=True)
    except OSError:
        pass


def detach_process(argv: list[str] | None = None, log_file: Path | None = None) -> int:
    """Relaunch scripts/extract_all.py in the background without --detach and return child PID."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    child_args = [a for a in raw_args if a != "--detach"]

    target_log = log_file or (config.RUNS_DIR / "logs" / "extract_all.log")
    target_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(Path(__file__).resolve()), *child_args]
    popen_kwargs: dict = {
        "cwd": str(REPO_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        detached_flag = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        new_group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        popen_kwargs["creationflags"] = detached_flag | new_group_flag
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    _log(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detached extract_all.py (pid={proc.pid})",
        target_log,
    )
    return proc.pid


def run_extraction(
    manifests_dir: Path | None = None,
    emb_dir: Path | None = None,
    log_file: Path | None = None,
    batch_size: int = 64,
    shard_size: int = 2000,
    device: str | None = None,
    force: bool = False,
    datasets: list[str] | None = None,
) -> list[Path]:
    """Extract embeddings sequentially for all manifests lacking an up-to-date .npz."""
    m_dir = manifests_dir or (config.CACHE_DIR / "manifests")
    e_dir = emb_dir or (config.CACHE_DIR / "emb")
    l_file = log_file or (config.RUNS_DIR / "logs" / "extract_all.log")

    e_dir.mkdir(parents=True, exist_ok=True)
    l_file.parent.mkdir(parents=True, exist_ok=True)

    manifests = sorted(m_dir.glob("*.jsonl")) if m_dir.exists() else []
    if datasets:
        prefixes = tuple(f"{ds}_" for ds in datasets)
        exact = {f"{ds}.jsonl" for ds in datasets}
        manifests = [m for m in manifests if m.name in exact or m.name.startswith(prefixes)]

    _log(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(manifests)} manifest(s) in {m_dir}",
        l_file,
    )

    outputs: list[Path] = []
    for manifest_path in manifests:
        out_path = e_dir / f"{manifest_path.stem}.npz"
        outputs.append(out_path)

        if not force and is_up_to_date(manifest_path, out_path):
            _log(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Up-to-date, skipping: {manifest_path.name} -> {out_path.name}",
                l_file,
            )
            continue

        cmd = [
            sys.executable,
            "-m",
            "vanalens.encoders",
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
            "--batch-size",
            str(batch_size),
            "--shard-size",
            str(shard_size),
        ]
        if device:
            cmd.extend(["--device", device])

        _log(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running: {' '.join(cmd)}",
            l_file,
        )
        t0 = time.perf_counter()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _log(line.rstrip("\r\n"), l_file)

        ret = proc.wait()
        elapsed = time.perf_counter() - t0
        if ret != 0:
            _log(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FAILED ({ret}) on {manifest_path.name} after {elapsed:.1f}s",
                l_file,
            )
            raise subprocess.CalledProcessError(ret, cmd)

        _log(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished {manifest_path.name} -> {out_path.name} in {elapsed:.1f}s",
            l_file,
        )

    _log("ALL DONE", l_file)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings for all manifests sequentially.")
    parser.add_argument("--detach", action="store_true", help="Relaunch in the background and return immediately.")
    parser.add_argument("--manifests-dir", type=Path, default=None, help="Override manifests directory")
    parser.add_argument("--emb-dir", type=Path, default=None, help="Override output embeddings directory")
    parser.add_argument("--log-file", type=Path, default=None, help="Override log file path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size passed to vanalens.encoders")
    parser.add_argument("--shard-size", type=int, default=2000, help="Shard size passed to vanalens.encoders")
    parser.add_argument("--device", type=str, default=None, help="Optional device passed to vanalens.encoders")
    parser.add_argument("--force", action="store_true", help="Re-extract even if .npz is up-to-date")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to filter (e.g. vl600 campus)")
    args = parser.parse_args(argv)

    if args.detach:
        detach_process(argv=argv, log_file=args.log_file)
        return

    run_extraction(
        manifests_dir=args.manifests_dir,
        emb_dir=args.emb_dir,
        log_file=args.log_file,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        device=args.device,
        force=args.force,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()

