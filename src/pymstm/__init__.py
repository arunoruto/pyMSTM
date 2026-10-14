"""pyMSTM -- Python bindings for the Multiple Sphere T-Matrix (MSTM) code."""

from pymstm._cli import (
    MstmExecutionError,
    MstmNotFoundError,
    MstmRunResult,
    find_mstm_binary,
    run_mstm,
)
from pymstm._convert import cli_normalized_s11_to_raw, s11_to_phase_function
from pymstm._dask import MstmTask, run_dask, solve_task
from pymstm._mstm import MSTM_LOCK, MstmBindings
from pymstm._solver import (
    MSTM,
    MstmMuellerPoint,
    MstmPerSphereResult,
    MstmResult,
)

__all__ = [
    "MSTM",
    "MstmBindings",
    "MSTM_LOCK",
    "MstmTask",
    "solve_task",
    "run_dask",
    "run_mstm",
    "find_mstm_binary",
    "MstmRunResult",
    "MstmNotFoundError",
    "MstmExecutionError",
    "s11_to_phase_function",
    "cli_normalized_s11_to_raw",
    "MstmResult",
    "MstmPerSphereResult",
    "MstmMuellerPoint",
]
__version__ = "1.1.0"
