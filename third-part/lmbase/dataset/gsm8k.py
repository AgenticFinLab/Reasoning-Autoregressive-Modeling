"""
Interface of the GSM8K dataset.

Dataset Source: https://huggingface.co/datasets/openai/gsm8k

Description:
    A collection of ~8,500 high-quality grade school math word problems designed
    for question-answering tasks involving multi-step reasoning. Problems require
    2 to 8 steps to solve using elementary operations.

Size: ~4.68 MB

Configurations:
    - main: Default configuration with train/test splits
    - socratic: Socratic variant with guided reasoning
    Config setting in code: subset="main" or subset="socratic"

Splits:
    - train: ~7,473 examples
    - test: ~1,319 examples

Features:
    - question: The math word problem
    - answer: Full solution with final answer marked by ####

License: MIT
Language: English
"""

from datasets import load_dataset

from lmbase.utils import re_extractor
from lmbase.dataset.base import TextSample, VisualTextBase


class GSM8KDataset(VisualTextBase):
    """A consistent interface for the GSM8k dataset."""

    def map_dataset(self):
        """Map the dataset to the desired format."""

        self.hf_dataset = load_dataset(self.hf_dataname, "main", split=self.split)

        super().map_dataset()

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        groundtruth_sol = re_extractor.extract_content(
            sample["answer"],
            marker="####",
        )
        groundtruth_sol = "" if groundtruth_sol is None else groundtruth_sol
        problem = sample["question"]
        question = f"{problem}{self.SOLUTION_FORMAT_PROMPT}"
        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=sample["answer"],
            groundtruth=groundtruth_sol,
            sample_info={"dataset": self.hf_dataname},
        )
