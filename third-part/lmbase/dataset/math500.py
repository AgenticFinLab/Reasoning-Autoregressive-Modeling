"""
Interface of the MATH-500 dataset.

Dataset Source: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

Description:
    A curated subset of 500 challenging math problems from the MATH benchmark,
    used in OpenAI's "Let's Verify Step by Step" research. Designed for evaluating
    mathematical reasoning with step-by-step verification.

Size: ~500 rows

Splits:
    - test: 500 examples (test set only)

Features:
    - problem: The math problem statement
    - solution: Step-by-step solution
    - answer: Final answer

License: Not specified (see dataset repository)
Language: English
Paper: Let's Verify Step by Step (OpenAI)
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class Math500Dataset(VisualTextBase):
    """A consistent interface for the MATH-500 dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        self.idx += 1

        # Create the question
        question = sample["problem"]
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # extract the groundtruth
        groundtruth = sample["answer"]

        # extract the cot_answer
        cot_answer = sample["solution"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "level": sample["level"],
                "subject": sample["subject"],
                "unique_id": sample["unique_id"],
            },
        )
