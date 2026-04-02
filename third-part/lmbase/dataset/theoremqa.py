"""
Interface of the TheoremQA dataset.

Dataset Source: https://huggingface.co/datasets/TIGER-Lab/TheoremQA

Description:
    A dataset of 800 high-quality question-answer pairs covering over 350 theorems
    across Math, EE&CS, Physics, and Finance. Designed to evaluate LLMs' ability
    to apply theorems to solve complex university-level questions.

Size: ~800 rows

Splits:
    - train: Training set
    - validation: Validation set
    - test: Test set

Features:
    - Question: The question text
    - Answer: Final answer
    - Picture: Associated image (if any)
    - Answer_type: Type of answer (numerical, text, etc.)
    - Theorem: Related theorem

License: Not specified (see dataset repository)
Language: English
Paper: arXiv:2305.12524
"""

import os

from lmbase.dataset.base import VisualTextSample, VisualTextBase


class TheoremQADataset(VisualTextBase):
    """A consistent interface for the TheoremQA dataset."""

    def __init__(
        self, split: str = "train", hf_dataname: str = None, config: dict = None
    ):

        self.data_path = config["data_path"]
        self.image_path = f"{self.data_path}/images"
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        question = sample["Question"]
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"
        image_data = sample["Picture"]
        q_image = None

        filename = f"{self.split}-Image-ID{self.idx}"
        filepath = f"{self.image_path}/{filename}.jpg"
        if os.path.exists(filepath):
            q_image = filepath
        else:
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                q_image = save_path

        groundtruth = sample["Answer"]
        cot_answer = ""

        return VisualTextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=[("image", q_image)],
            sample_info={
                "dataset": self.hf_dataname,
                "answer_type": sample["Answer_type"],
            },
        )
