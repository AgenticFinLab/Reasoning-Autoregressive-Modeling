"""NLCP V3 DataLoader: Wraps lmbase datasets to produce structured BuilderInput batches.

DESIGN SOURCE:
    The user's requirement:
    1. The data part should only contain raw text, NO tokenization.
    2. The builder receives text and handles tokenization internally.

    This module provides the data pipeline that produces text batches
    with explicit Q/CoT/Solution boundaries. All tokenization happens
    inside ConceptPyramidBuilder.forward().

USAGE:
    from nlcpV3.data_loader import NLCPV3DataLoader, BuilderInput

    loader = NLCPV3DataLoader(
        data_cfg={"data_name": "gsm8k", "split": "train"},
        batch_size=4,
        include_solution=True,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    for batch in loader:
        # batch: BuilderInput with raw text (Q, CoT, Solution)
        pyramid = builder(batch)  # forward() tokenizes internally
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from torch.utils.data import DataLoader

from lmbase.dataset import registry


# =========================================================================
# BuilderInput — structured text input to ConceptPyramidBuilder.forward()
# =========================================================================


@dataclass
class BuilderInput:
    """Structured text input for ConceptPyramidBuilder with Q/CoT/Solution boundaries.

    PURPOSE:
        Encapsulate raw text for Q, CoT, and Solution so that the Builder's
        forward() can internally understand which text belongs to which part,
        and perform all tokenization inside the model.

    DESIGN:
        Each part is stored as a list of text strings:
        - questions: Question text strings [B]
        - cot_answers: Chain-of-thought text strings [B]
        - solutions: Groundtruth solution text strings [B]

        The Builder's forward() will:
        1. Tokenize cot_answers -> encode_cot() -> H_CoT
        2. Build concept pyramid from H_CoT
        3. Tokenize questions and solutions -> attach to PyramidOutput
           for downstream compute_reasoning_loss()

    ATTRIBUTES:
        questions: Question text strings [B]
        cot_answers: Chain-of-thought text strings [B]
        solutions: Groundtruth solution text strings [B]
    """

    questions: List[str]
    cot_answers: List[str]
    solutions: List[str]

    @property
    def batch_size(self) -> int:
        """Batch size: B."""
        return len(self.questions)

    @property
    def has_solution(self) -> bool:
        """Whether this input includes non-empty solution text."""
        return len(self.solutions) > 0


# =========================================================================
# NLCPV3DataLoader — wraps lmbase.dataset.registry for Builder training
# =========================================================================


class NLCPV3DataLoader:
    """DataLoader for NLCP V3 Builder training.

    PURPOSE:
        Wraps lmbase datasets (e.g., GSM8K) and produces BuilderInput batches
        with clearly separated Q, CoT, and Solution as raw text strings.

        The collate_fn only extracts text from samples — NO tokenization.
        Tokenization is handled entirely by the Builder's forward() method.

    USAGE:
        >>> loader = NLCPV3DataLoader(
        ...     data_cfg={"data_name": "gsm8k", "split": "train"},
        ...     batch_size=4,
        ...     include_solution=True,
        ...     shuffle=True,
        ...     drop_last=True,
        ...     num_workers=0,
        ... )
        >>> for batch in loader:
        ...     assert isinstance(batch, BuilderInput)
        ...     pyramid = builder(batch)  # forward() tokenizes internally

    Args:
        data_cfg: Dataset configuration dict for lmbase registry.
            Must contain: data_name, split (optional: data_path, subset)
        batch_size: Number of samples per batch.
        include_solution: If True, also extract groundtruth solutions.
        shuffle: Whether to shuffle the dataset.
        drop_last: Whether to drop incomplete final batches.
        num_workers: Number of dataloader workers.
        kwargs: Additional arguments passed to DataLoader.
    """

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        batch_size: int,
        include_solution: bool,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        **kwargs,
    ):
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.include_solution = include_solution
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.extra_kwargs = kwargs

        # Load dataset from lmbase registry
        self.dataset = registry.get(data_cfg, split=data_cfg["split"])

    def _collate_fn(self, raw_samples: List[Any]) -> BuilderInput:
        """Collate raw lmbase samples into a BuilderInput batch (text only).

        PROCESS:
            1. Extract question, cot_answer, groundtruth from each sample
            2. Return BuilderInput with raw text strings
            3. NO tokenization here — forward() handles it internally

        Args:
            raw_samples: List of lmbase TextSample objects.

        Returns:
            BuilderInput with raw text for Q, CoT, Solution.
        """
        questions: List[str] = []
        cot_answers: List[str] = []
        solutions: List[str] = []

        for sample in raw_samples:
            questions.append(sample.question)
            cot_answers.append(sample.cot_answer)
            if self.include_solution:
                solutions.append(sample.groundtruth)

        return BuilderInput(
            questions=questions,
            cot_answers=cot_answers,
            solutions=solutions,
        )

    def __iter__(self):
        """Iterate over batches as BuilderInput objects."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            **self.extra_kwargs,
        )
        for batch in dataloader:
            yield batch

    def __len__(self) -> int:
        """Number of batches per epoch."""
        dataset_len = len(self.dataset)
        if self.drop_last:
            return dataset_len // self.batch_size
        return (dataset_len + self.batch_size - 1) // self.batch_size

    @property
    def dataset_size(self) -> int:
        """Total number of samples in the underlying dataset."""
        return len(self.dataset)
