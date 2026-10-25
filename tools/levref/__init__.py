"""Levanter reference harness: builds the plain muonh_qwen3 baseline with levanter (read-only) and writes fixtures.

Every module reads ROOT from NANOMARIN_ROOT, imports levanter from ROOT/project/marin, and writes only under
ROOT/project/nanomarin/cache/levref/<size>/. It never writes into the marin tree, so bytecode writing is off from here on.
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
