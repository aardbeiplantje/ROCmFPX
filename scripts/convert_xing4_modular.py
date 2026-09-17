#!/usr/bin/env python3
"""Convert Xing4.0 (XingChen-AGI/Xing4.0-29B-A4B) checkpoints with the modular ROCmFPX converter.

Modelled on scripts/convert_deepseek_v4_modular.py, minus the deepseek4-only
(--deepseek4-include-mtp / --deepseek4-max-layers) knobs: xing4 always emits its single
nextn/MTP block (layer 40) as part of block_count.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversion import ModelType, get_model_architecture, get_model_class
from conversion.base import ModelBase, gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Xing4.0 checkpoints (arch xing4).")
    parser.add_argument("model", type=Path)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", choices=["f16", "bf16", "auto"], default="bf16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-temp-file", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ftype_map = {
        "f16":  gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    hparams = ModelBase.load_hparams(args.model, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    logging.getLogger("hf-to-gguf").info("Model architecture: %s", arch)
    model_class = get_model_class(arch, mmproj=False)

    model = model_class(
        args.model,
        ftype_map[args.outtype],
        args.outfile,
        use_temp_file=args.use_temp_file,
        dry_run=args.dry_run,
    )
    model.write()


if __name__ == "__main__":
    main()
