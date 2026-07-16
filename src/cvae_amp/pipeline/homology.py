"""BLASTP wrapper for homology-based filtering."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _resolve_blast_bin(blast_bin: str | None = None) -> Path:
    """Resolve BLAST+ bin directory.

    Priority: explicit argument > BLAST_BIN env var > empty (tools on PATH).
    """
    if blast_bin:
        return Path(blast_bin)
    env = os.environ.get("BLAST_BIN", "")
    if env:
        return Path(env)
    return Path()


def build_blast_db(
    fasta_path: Path,
    db_name: str,
    blast_bin: str | None = None,
) -> None:
    makeblastdb = _resolve_blast_bin(blast_bin) / "makeblastdb"
    subprocess.run([
        str(makeblastdb), "-in", str(fasta_path),
        "-dbtype", "prot", "-out", db_name,
    ], capture_output=True, text=True)


def run_blastp(
    query_fasta: Path,
    db_name: str,
    output_path: Path,
    evalue: float = 1e-5,
    threads: int = 4,
    blast_bin: str | None = None,
) -> Path:
    blastp = _resolve_blast_bin(blast_bin) / "blastp"
    subprocess.run([
        str(blastp),
        "-query", str(query_fasta),
        "-db", db_name,
        "-out", str(output_path),
        "-outfmt", "6 qseqid sseqid pident evalue qlen slen length",
        "-evalue", str(evalue),
        "-num_threads", str(threads),
        "-max_target_seqs", "5",
    ], capture_output=True, text=True)
    return output_path


def parse_blast_hits(blast_output: Path, identity_cutoff: float = 30.0) -> set[str]:
    hits: set[str] = set()
    with open(blast_output) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            pident = float(parts[2])
            if pident >= identity_cutoff:
                hits.add(parts[0])
    return hits
