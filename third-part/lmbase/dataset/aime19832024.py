"""
Interface of the AIME19832024 dataset.

Dataset Source: https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024

Description:
    A comprehensive collection of AIME (American Invitational Mathematics Examination)
    problems spanning from 1983 to 2024 (Part 2). Contains competition math problems
    with integer answers.

Size: ~933 rows

Splits:
    - train: Training set (all examples)

Features:
    - ID: Unique identifier
    - Question: The math problem statement
    - Answer: Final answer (integer)
    - Year: Competition year
    - Problem Number: Problem number in that year's exam

License: MIT
Language: English
Note: Intended for benchmarking, not for training
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class AIME19832024Dataset(VisualTextBase):
    """A consistent interface for the AIME19832024 dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        cot_answer = ""
        # opt = re_utility.extract_format_equations(cot_answer,
        groundtruth_sol = sample["Answer"]
        problem = sample["Question"]
        question = f"{problem}{self.SOLUTION_FORMAT_PROMPT}"

        return TextSample(
            main_id=sample["ID"],
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            sample_info={
                "dataset": self.hf_dataname,
                "year": sample["Year"],
                "problem_number": sample["Problem Number"],
            },
        )
