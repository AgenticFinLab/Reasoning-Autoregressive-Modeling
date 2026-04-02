"""
Interface of the MATH dataset.

Dataset Source: https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval

Description:
    A collection of ~12,500 challenging competition math problems with detailed
    step-by-step solutions. Covers various math topics from AMC, AIME, and other
    math competitions.

Size: ~12,500 rows

Splits:
    - train: ~7,500 examples
    - test: ~5,000 examples

Features:
    - problem: The math problem statement
    - solution: Full step-by-step solution
    - answer: Final answer (extracted from solution)
    - level: Difficulty level (1-5)
    - type: Math topic/category

License: MIT
Language: English
"""

from math_verify import LatexExtractionConfig, parse

from lmbase.dataset.base import TextSample, VisualTextBase


class MATHDataset(VisualTextBase):
    """A consistent interface for the MATH dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        cot_answer = sample["solution"]
        # opt = re_utility.extract_format_equations(cot_answer, target_format="\\boxed")
        # groundtruth_sol = "" if opt is None else opt[-1]
        # The parsed item will be a list holding a value and a str value
        groundtruth_sol = parse(
            cot_answer,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        groundtruth_sol = "" if len(groundtruth_sol) == 0 else groundtruth_sol[-1]
        problem = sample["problem"]
        question = f"{problem}{self.SOLUTION_FORMAT_PROMPT}"

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            sample_info={
                "dataset": self.hf_dataname,
                "level": sample["level"],
                "type": sample["type"],
            },
        )
