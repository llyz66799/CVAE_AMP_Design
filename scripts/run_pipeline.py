#!/usr/bin/env python3
"""Full end-to-end pipeline: generate → CD-HIT → BLAST → filter → predict."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_amp.pipeline.workflow import AMPWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Full AMP design pipeline")
    parser.add_argument("--model", default="cvae_pred", choices=["vae", "cvae", "cvae_pred"])
    parser.add_argument("--target", type=str, default="1.0,1.0,1.0")
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--work-dir", type=str, default=None)
    args = parser.parse_args()

    target = tuple(float(x) for x in args.target.split(","))
    if len(target) != 3:
        sys.exit("Target must be 3 comma-separated values (hemolysis,ecoli,amp)")

    work_dir = Path(args.work_dir) if args.work_dir else None
    wf = AMPWorkflow(
        model_name=args.model,
        target=target,
        num_gen=args.num,
        temperature=args.temperature,
        work_dir=work_dir,
    )
    wf.run()


if __name__ == "__main__":
    main()
