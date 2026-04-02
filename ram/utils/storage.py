"""Storage managers for training history and reconstruction samples.

Provides memory-efficient storage classes using lmbase's BlockBasedStoreManager
for handling large training histories and reconstruction sample datasets.

Classes:
    TrainingHistory - Manager for step-by-step training records
    ReconstructionSampleStore - Block-based storage for reconstruction samples
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from lmbase.utils.tools import BlockBasedStoreManager
except ImportError:
    # Optional dependency - will raise error if used
    BlockBasedStoreManager = None

from ram.generic import (
    ReconstructionSample,
    TrainingConfig,
    TrainingStep,
)
from ram.utils.serialization import load_json, save_json


class TrainingHistory:
    """Manager for training history storage.

    Handles step-by-step training records with JSON persistence.
    Each step contains loss values, learning rates, and optional reconstruction samples.

    Storage Format (training_history.json):
        {
            "config": {...},           # TrainingConfig snapshot
            "steps": [                 # List of TrainingStep records
                {
                    "epoch": 1,
                    "step_in_epoch": 100,
                    "global_step": 100,
                    "total_loss": 0.5,
                    "recon_loss": 0.5,
                    "avg_loss": 0.52,
                    "lr_encoder": 1e-5,
                    "lr_decoder": 1e-5,
                    "timestamp": "2024-01-01T12:00:00",
                    "reconstruction_samples": [...]  # Optional
                },
                ...
            ],
            "total_steps": 1000
        }

    Attributes:
        config: Training configuration snapshot.
        steps: List of TrainingStep records.
        path: Path to history JSON file.
    """

    def __init__(self, config: TrainingConfig, path: Path):
        """Initialize training history.

        Args:
            config: Training configuration (stored as snapshot).
            path: Path to history JSON file.
        """
        self.config = config
        self.steps: List[TrainingStep] = []
        self.path = path

    def append(self, step: TrainingStep) -> None:
        """Append a training step record.

        Args:
            step: TrainingStep record to append.
        """
        self.steps.append(step)

    def flush(self) -> None:
        """Flush history to JSON file.

        Writes complete history including config and all steps.
        Should be called at training completion and checkpoint saves.
        """
        data = {
            "config": self.config.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": len(self.steps),
        }
        save_json(data, self.path)

    def load(self) -> None:
        """Load existing history from file.

        Reconstructs TrainingStep objects including nested ReconstructionSample lists.
        Clears current steps before loading.
        """
        if self.path.exists():
            data = load_json(self.path)
            self.steps = []
            for s in data.get("steps", []):
                samples = [
                    ReconstructionSample(**rs)
                    for rs in s.pop("reconstruction_samples", [])
                ]
                step = TrainingStep(**s)
                step.reconstruction_samples = samples
                self.steps.append(step)

    def __len__(self) -> int:
        """Return number of stored steps."""
        return len(self.steps)

    def get_latest_avg_loss(self) -> float:
        """Get latest average loss from history.

        Returns:
            Average loss of most recent step, or 0.0 if no steps recorded.
        """
        if self.steps:
            return self.steps[-1].avg_loss
        return 0.0


class ReconstructionSampleStore:
    """Block-based storage for reconstruction samples.

    Uses BlockBasedStoreManager from lmbase for memory-efficient storage
    of reconstruction samples. Each sample is keyed by global_step
    for alignment with TrainingHistory.

    Storage Structure:
        samples/
        ├── samples_block_0.json     # Contains up to block_size sample records
        ├── samples_block_1.json
        └── samples-store-information.json  # Index file

    Sample Record Format:
        Key: "step_{global_step}"
        Value: {
            "epoch": int,
            "step_in_epoch": int,
            "global_step": int,
            "timestamp": str,
            "samples": [
                {"index": int, "original": str, "reconstructed": str},
                ...
            ]
        }

    Alignment with TrainingHistory:
        - Keys match TrainingStep.global_step for easy correlation
        - Record includes epoch/step_in_epoch for verification
        - Use global_step to query both history and samples

    Attributes:
        folder: Directory for sample storage.
        block_size: Maximum records per block file.
        store: BlockBasedStoreManager instance.
    """

    def __init__(
        self,
        folder: str,
        block_size: int = 50,
    ):
        """Initialize ReconstructionSampleStore.

        Args:
            folder: Directory path for sample storage.
            block_size: Maximum sample records per block file (default: 50).

        Raises:
            RuntimeError: If lmbase package is not installed.
        """
        if BlockBasedStoreManager is None:
            raise RuntimeError(
                "lmbase package is required for ReconstructionSampleStore. "
                "Install it with: pip install lmbase"
            )
        self.folder = folder
        self.block_size = block_size
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.store = BlockBasedStoreManager(
            folder=folder,
            file_format="json",
            block_size=block_size,
        )

    def save_samples(
        self,
        step: TrainingStep,
        samples: List[ReconstructionSample],
    ) -> str:
        """Save reconstruction samples for a training step.

        Args:
            step: TrainingStep record with epoch/step/global_step info.
            samples: List of ReconstructionSample instances.

        Returns:
            str: The save key ("step_{global_step}").

        Example:
            >>> key = store.save_samples(step_record, recon_samples)
            >>> print(key)
            "step_100"
        """
        save_key = f"step_{step.global_step}"

        record = {
            "epoch": step.epoch,
            "step_in_epoch": step.step_in_epoch,
            "global_step": step.global_step,
            "timestamp": step.timestamp,
            "samples": [asdict(s) for s in samples],
        }

        self.store.save(save_key, record)
        return save_key

    def load_samples(self, global_step: int) -> Optional[Dict[str, Any]]:
        """Load reconstruction samples for a specific global step.

        Args:
            global_step: Global step to load samples for.

        Returns:
            Sample record dict or None if not found.

        Example:
            >>> record = store.load_samples(100)
            >>> print(record["samples"][0]["original"])
            "Original input text..."
        """
        save_key = f"step_{global_step}"
        return self.store.load(save_key)

    def load_all_samples(self) -> Dict[str, Dict[str, Any]]:
        """Load all reconstruction samples.

        WARNING: Loads all records into memory. Use sparingly.

        Returns:
            Dict mapping save_key to sample record.
        """
        return self.store.load_all("step")

    def get_sample_count(self) -> int:
        """Get total number of sample records stored."""
        all_samples = self.load_all_samples()
        return len(all_samples)
