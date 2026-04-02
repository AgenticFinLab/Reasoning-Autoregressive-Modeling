"""
Interface of the Finance Agent Benchmark dataset.

Dataset Source: https://huggingface.co/datasets/vals-ai/finance_agent_benchmark

Description:
    A benchmark dataset with 537 expert-authored financial research questions
    across 9 task categories. Evaluates AI models on real-world financial analyst
    tasks including data retrieval, market research, and financial projections.
    Developed by Vals AI in collaboration with Stanford and financial experts.

Size: ~537 questions

Splits:
    - train: All examples (single split)

Features:
    - question: The financial research question
    - answer: Expert answer
    - rubric: Evaluation rubric
    - task_category: Type of financial task

License: CC BY 4.0
Language: English
Paper: arXiv:2508.00828
Note: Requires tool use (search, SEC filings access) for best performance
"""

from datasets import load_dataset

from lmbase.utils import re_extractor
from lmbase.dataset.base import TextSample, VisualTextBase


class FinAgentDataset(VisualTextBase):
    """A consistent interface for the Finance Agent Benchmark dataset."""

    def map_dataset(self):
        """Map the dataset to the desired format."""
        # Load the dataset - only train split available
        self.hf_dataset = load_dataset(self.hf_dataname, split=self.split)

        super().map_dataset()

    def to_format(self, sample):
        """Convert a raw sample to the standard format."""
        self.idx += 1

        # Extract fields from the dataset
        question = sample["Question"]
        answer = sample["Answer"]
        question_type = sample["Question Type"] if "Question Type" in sample else ""
        expert_time = (
            sample["Expert time (mins)"] if "Expert time (mins)" in sample else ""
        )
        rubric = sample["Rubric"] if "Rubric" in sample else ""

        # Create the formatted question with solution prompt
        formatted_question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # Extract ground truth if available (though this dataset may not have standard markers)
        groundtruth_sol = re_extractor.extract_content(answer, marker="####")
        groundtruth_sol = "" if groundtruth_sol is None else groundtruth_sol

        return TextSample(
            main_id=f"FINAGENT_ID{self.idx}",
            split=self.split,
            question=formatted_question,
            cot_answer=answer,
            groundtruth=groundtruth_sol,
            sample_info={
                "dataset": self.hf_dataname,
                "question_type": question_type,
                "expert_time_mins": expert_time,
                "rubric": rubric,
            },
        )
