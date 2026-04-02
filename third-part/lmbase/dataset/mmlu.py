"""
Interface of the MMLU dataset.

Dataset Source: https://huggingface.co/datasets/cais/mmlu

Description:
    Massive Multitask Language Understanding (MMLU) benchmark for evaluating
    large language models across 57 diverse subjects including math, history,
    science, law, and more. Multiple-choice questions covering humanities,
    social sciences, STEM, and professional knowledge.

Size: ~57,303 MB, ~116,000 questions

Configurations:
    - all: All subjects combined
    - Individual subjects: abstract_algebra, anatomy, astronomy, business_ethics,
      clinical_knowledge, college_biology, college_chemistry, etc.
    Config setting in code: subset="all" or subset="<subject_name>"

Splits:
    - auxiliary_train: Auxiliary training set (~99.8k examples)
    - test: Test set
    - validation: Validation set
    - dev: Development set

Features:
    - question: The question text
    - choices: Multiple choice options (A, B, C, D)
    - answer: Correct answer

License: MIT
Language: English
Paper: Measuring Massive Multitask Language Understanding
"""

from datasets import load_dataset

from lmbase.identifier import OPTION_SOLUTION_PROMPT
from lmbase.dataset.base import TextSample, VisualTextBase


class MMLUDataset(VisualTextBase):
    """A consistent interface for the MMLU dataset."""

    def map_dataset(self):
        """Map the dataset to the desired format."""

        self.hf_dataset = load_dataset(
            self.hf_dataname,
            "all",
            split=self.split,
        )

        super().map_dataset()

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""

        self.idx += 1

        # Create the sample
        question = sample["question"]
        question = f"{question} {OPTION_SOLUTION_PROMPT}."

        # Get the list of choices
        options = sample["choices"]

        if options is None or len(options) == 0:
            question = f"{question}\n"
        else:
            option_letters = [chr(ord("A") + num) for num in range(len(options))]
            options_str = [
                f"({letter}): {choice}"
                for choice, letter in zip(options, option_letters)
            ]
            options_str = "\n".join(options_str)
            question = f"{question}\nOptions:\n{options_str}"

        groundtruth = chr(ord("A") + sample["answer"])

        return TextSample(
            main_id=f"ID{self.idx}",
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "subject": sample["subject"],
            },
        )
