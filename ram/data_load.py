"""Data loading utilities for RAM framework.

This module provides a simple, extensible data loader for RAM training.
The RamDataLoaderRegistry class handles dataset loading from lmbase and
allows customization of target text formatting via subclassing.

Usage:
    >>> from ram.data_load import RamDataLoaderRegistry
    >>> from ram.generic import RamSample
    >>>
    >>> # Basic usage - question + cot as target (default)
    >>> loader = RamDataLoaderRegistry({
    ...     "data_name": "gsm8k",
    ...     "split": "train",
    ...     "batch_size": 32,
    ...     "num_workers": 4,
    ...     "shuffle": True,
    ...     "drop_last": True,
    ... })
    >>>
    >>> for sample in loader:
    ...     # sample: RamSample with original sample and formatted target
    ...     model_output = model(sample.target_text)

Custom target formatting:
    >>> class MyDataLoader(RamDataLoaderRegistry):
    ...     def format_target(self, sample) -> str:
    ...         return f"Q: {sample.question}\nA: {sample.cot_answer}"
    >>>
    >>> loader = MyDataLoader({
    ...     "data_name": "gsm8k",
    ...     "split": "train",
    ...     "batch_size": 32,
    ...     "num_workers": 4,
    ...     "shuffle": True,
    ...     "drop_last": True,
    ... })
"""

from typing import Any, Dict, List

from lmbase.dataset import registry
from lmbase.dataset.base import TextSample
from torch.utils.data import DataLoader, Dataset

from ram.generic import RamSample


class RamDataLoaderRegistry:
    """Standardized dataloader for RAM framework.

    A simple, extensible class that wraps lmbase datasets and provides
    customizable target text formatting. Inherit and override
    `format_target()` to customize how samples are converted to text.

    All configuration parameters must be explicitly provided in data_config.
    No default values are set within the class.

    Args:
        data_config: Dataset configuration dict containing:
            - data_name: Name of the dataset (e.g., "gsm8k")
            - data_dir: Data directory path (can be empty string)
            - split: Dataset split (e.g., "train", "test")
            - batch_size: Batch size for training
            - num_workers: Number of dataloader workers
            - shuffle: Whether to shuffle data
            - drop_last: Whether to drop incomplete batches
        **kwargs: Additional arguments passed to DataLoader

    Example:
        >>> loader = RamDataLoaderRegistry({
        ...     "data_name": "gsm8k",
        ...     "data_dir": "",
        ...     "split": "train",
        ...     "batch_size": 32,
        ...     "num_workers": 4,
        ...     "shuffle": True,
        ...     "drop_last": True,
        ... })
        >>> for batch in loader:
        ...     assert isinstance(batch, list)
        ...     assert isinstance(batch[0], str)
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        **kwargs,
    ):
        self.data_config = data_config
        self.dataloader_kwargs = kwargs

        # Extract required parameters from data_config
        # All values must be explicitly provided - no defaults
        self.split = data_config["split"]
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config["num_workers"]
        self.shuffle = data_config["shuffle"]
        self.drop_last = data_config["drop_last"]

        # Load base dataset from lmbase
        self.base_dataset = registry.get(data_config, split=self.split)

        # Build the dataloader
        self._dataloader = self._build_dataloader()

    def _build_dataloader(self) -> DataLoader:
        """Build the PyTorch DataLoader."""
        return DataLoader(
            self.base_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_fn,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            **self.dataloader_kwargs,
        )

    def _collate_fn(self, samples: List[TextSample]) -> List[RamSample]:
        """Collate function that wraps samples into RamSample objects.

        Args:
            samples: List of TextSample from lmbase dataset

        Returns:
            List of RamSample with formatted target_text
        """
        return [
            RamSample(
                original=s,
                target_text=self.format_target(s),
                sample_id=getattr(s, "id", None),
            )
            for s in samples
        ]

    def format_target(self, sample: TextSample) -> str:
        """Format a sample into target text.

        This is the main customization point. Override this method
        to define how samples are converted to text strings.

        Default implementation: combines question + cot_answer.

        Args:
            sample: A TextSample from the lmbase dataset

        Returns:
            Formatted text string
        """
        # Direct attribute access - lmbase TextSample has these fields
        question = sample.question
        cot = sample.cot_answer

        # Combine question and CoT
        parts = [p for p in [question, cot] if p]
        return "\n".join(parts)

    def __iter__(self):
        """Iterate over batches."""
        return iter(self._dataloader)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._dataloader)

    @property
    def dataset(self) -> Dataset:
        """Access the underlying dataset."""
        return self.base_dataset
