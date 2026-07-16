"""CD-HIT wrapper for sequence deduplication."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cvae_amp.config.paths import CDHIT_BIN


def run_cdhit(
    input_fasta: Path,
    output_path: Path,
    identity: float = 0.70,
    threads: int = 4,
) -> Path:
    cmd = [
        str(CDHIT_BIN),
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
