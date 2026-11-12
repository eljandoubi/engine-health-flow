"""The benchmark csv: column layout, parsing and the in-memory dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Zenodo sandbox record of the data (arXiv 2604.08460). The paper says the link will move
# after publication, so the url and the checksum live here and nowhere else.
URL = "https://sandbox.zenodo.org/records/469530/files/2000_aligned_and_clean_indexfalse.csv?download=1"
FILENAME = "2000_aligned_and_clean_indexfalse.csv"
MD5 = "ab0b9632b65e08a419ca3e29e1175d61"

# Efficiency and mass flow modifier of each module, in the column order of the csv.
# The paper numbers them HI1..HI10 without saying in which order, so only averages are comparable.
MODULES = ["CmpBst", "CmpFan", "CmpH", "TrbH", "TrbL"]
HEALTH = [f"deg_{m}_s_map{k}_in" for m in MODULES for k in ("Eff", "Wc")]
HEALTH_SHORT = [f"{m}_{k}" for m in MODULES for k in ("eff", "flow")]

PHASES = ["Cruise", "Takeoff", "Climb1", "Climb2"]
SENSORS = ["HPC_Tout", "HP_Nmech", "HPC_Tin", "LPT_Tin", "Fuel_flow", "HPC_Pout_st", "LP_Nmech"]
SENSOR_COLUMNS = [f"{p}_DeckSMR__{s}" for p in PHASES for s in SENSORS]


@dataclass
class Dataset:
    """Sequences stored back to back: rows starts[i]:starts[i + 1] belong to sequence i."""

    ids: np.ndarray  # (n_seq,) sequence ids
    starts: np.ndarray  # (n_seq + 1,) row offsets
    health: np.ndarray  # (n_rows, 10) health indicators, float32
    sensors: np.ndarray  # (n_rows, 28) sensor readings, float32
    maintenance: np.ndarray  # (n_rows,) true on the rows where a maintenance happened

    @property
    def n_seq(self) -> int:
        return len(self.ids)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.starts)

    def rows(self, seq: int) -> slice:
        return slice(self.starts[seq], self.starts[seq + 1])

    def take(self, seqs) -> "Dataset":
        """A new dataset with only the given sequences, in the given order."""
        seqs = np.asarray(seqs)
        keep = np.concatenate([np.arange(self.starts[i], self.starts[i + 1]) for i in seqs])
        starts = np.concatenate([[0], np.cumsum(self.lengths[seqs])])
        return Dataset(
            self.ids[seqs],
            starts,
            self.health[keep],
            self.sensors[keep],
            self.maintenance[keep],
        )


def read_csv(path: Path) -> Dataset:
    cols = ["sequence_id", "timestep", "maintenance", *HEALTH, *SENSOR_COLUMNS]
    floats = dict.fromkeys(HEALTH + SENSOR_COLUMNS, np.float32)
    df = pd.read_csv(path, usecols=cols, dtype=floats)

    # The real file is already grouped and sorted, but nothing in the record promises it.
    codes, ids = pd.factorize(df["sequence_id"])
    order = np.lexsort((df["timestep"].to_numpy(), codes))
    df = df.iloc[order]
    starts = np.concatenate([[0], np.cumsum(np.bincount(codes))])

    return Dataset(
        ids=np.asarray(ids),
        starts=starts,
        health=df[HEALTH].to_numpy(),
        sensors=df[SENSOR_COLUMNS].to_numpy(),
        maintenance=df["maintenance"].to_numpy(dtype=bool),
    )
