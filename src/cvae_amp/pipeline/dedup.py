"""CD-HIT wrapper for sequence deduplication."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _resolve_cdhit(cdhit_bin: str | None = None) -> str:
    """Resolve cd-hit binary path.

    Priority: explicit argument > CDHIT_BIN env var > default 'cd-hit' on PATH.
    """
    if cdhit_bin:
        return cdhit_bin
    env = os.environ.get("CDHIT_BIN", "")
    if env:
        return env
    return "cd-hit"


def run_cdhit(
    input_fasta: Path,
    output_path: Path,
    identity: float = 0.70,
    threads: int = 4,
    cdhit_bin: str | None = None,
) -> Path:
    cmd = [
        _resolve_cdhit(cdhit_bin),
        "-i", str(input_fasta),
        "-o", str(output_path),
        "-c", str(identity),
        "-n", "5",
        "-d", "0",
        "-T", str(threads),
        "-M", "4000",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    n = sum(1 for line in open(output_path) if line.startswith(">"))
    print(f"  After CD-HIT: {n} sequences")
    return output_path
