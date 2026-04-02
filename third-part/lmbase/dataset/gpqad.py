"""
Interface of the GPQA-Diamond dataset.

Dataset Source: https://huggingface.co/datasets/fingertap/GPQA-Diamond

Description:
    Graduate-Level Google-Proof Q&A (GPQA) Diamond subset - a challenging benchmark
    with 198 expert-validated multiple-choice questions in biology, chemistry, and
    physics. Questions are "Google-proof" meaning even skilled non-experts with
    internet access perform poorly (~34%), while PhD-level experts score ~65-70%.

Size: ~198 rows

Splits:
    - train: Training set (small, intended for few-shot)

Features:
    - question: Question text with multiple choice options
    - answer: Correct answer (A, B, C, or D)

License: Not specified (see dataset repository)
Language: English
Paper: arXiv:2311.12022
Note: Questions are difficult by design - even experts find them challenging
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class GPQADiamondDataset(VisualTextBase):
    """A consistent interface for the GPQA-Diamond dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        # The question field in GPQA-Diamond usually contains the question and options
        question = sample["question"]
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # The answer is the correct option letter (e.g., "A", "B", "C", "D")
        groundtruth = sample["answer"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",  # GPQA-Diamond typically doesn't provide CoT in the main subset
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
            },
        )
