"""
Interface of the AIME2024 dataset.

Dataset Source: https://huggingface.co/datasets/HuggingFaceH4/aime_2024

Description:
    A collection of 30 math problems from the 2024 AIME I and II tests.
    AIME (American Invitational Mathematics Examination) is a prestigious
    math competition for high school students.

Size: ~81.7 KB, 30 rows

Splits:
    - train: 30 examples (single split)

Features:
    - id: Unique identifier
    - problem: The math problem statement
    - solution: Step-by-step solution
    - answer: Final answer (integer 0-999)
    - url: Source URL
    - year: Competition year

License: Not specified (see dataset repository)
Language: English
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class AIME2024Dataset(VisualTextBase):
    """A consistent interface for the AIME2024 dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        cot_answer = sample["solution"]
        # opt = re_utility.extract_format_equations(cot_answer,
        groundtruth_sol = sample["answer"]
        problem = sample["problem"]
        question = f"{problem}{self.SOLUTION_FORMAT_PROMPT}"

        return TextSample(
            main_id=sample["id"],
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            sample_info={
                "dataset": self.hf_dataname,
                "url": sample["url"],
                "year": sample["year"],
            },
        )
