"""pyMSTM -- Python bindings for the Multiple Sphere T-Matrix (MSTM) code."""

from pymstm._cli import (
    MstmExecutionError,
    MstmNotFoundError,
    MstmRunResult,
    find_mstm_binary,
    run_mstm,
)
from pymstm._convert import cli_normalized_s11_to_raw, s11_to_phase_function
from pymstm._mstm import MSTM

__all__ = [
    "MSTM",
    "run_mstm",
    "find_mstm_binary",
    "MstmRunResult",
    "MstmNotFoundError",
    "MstmExecutionError",
    "s11_to_phase_function",
    "cli_normalized_s11_to_raw",
]
__version__ = "1.1.0"
