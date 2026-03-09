"""
Interface of the Finance Agent Benchmark dataset.

The dataset is from https://huggingface.co/datasets/vals-ai/finance_agent_benchmark
It contains financial research tasks with expert answers and rubrics.
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
