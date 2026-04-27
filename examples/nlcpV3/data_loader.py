"""NLCP V3 DataLoader: raw text batches for Builder training.

Tokenization happens inside ConceptPyramidBuilder.forward().
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from torch.utils.data import DataLoader

from lmbase.dataset import registry


@dataclass
class BuilderInput:
    """Raw text input: questions, CoT answers, solutions."""

    questions: List[str]
    cot_answers: List[str]
    solutions: List[str]

    @property
    def batch_size(self) -> int:
        return len(self.questions)

    @property
    def has_solution(self) -> bool:
        return len(self.solutions) > 0


class NLCPV3DataLoader:
    """Wraps lmbase registry to yield BuilderInput batches."""

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
        questions, cot_answers, solutions = [], [], []
        for sample in raw_samples:
            questions.append(sample.question)
            cot_answers.append(sample.cot_answer)
            if self.include_solution:
                solutions.append(sample.groundtruth)
        return BuilderInput(
            questions=questions, cot_answers=cot_answers, solutions=solutions
        )

    def __iter__(self):
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
        dataset_len = len(self.dataset)
        if self.drop_last:
            return dataset_len // self.batch_size
        return (dataset_len + self.batch_size - 1) // self.batch_size

    @property
    def dataset_size(self) -> int:
        return len(self.dataset)
