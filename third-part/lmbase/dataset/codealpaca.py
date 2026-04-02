"""
Interface of the CodeAlpaca dataset.

Dataset Source: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k

Description:
    A collection of ~20,000 code instruction-following examples generated using
    the Stanford Alpaca approach. Contains programming tasks with instructions
    and corresponding code solutions.

Size: ~20,000 rows

Splits:
    - train: ~20,000 examples (single split)

Features:
    - instruction: Natural language instruction
    - input: Optional input context
    - output: Code solution

License: CC BY 4.0
Language: English (instructions), Multiple programming languages (code)
"""

from lmbase.dataset.base import TextCodeSample, VisualTextBase


class CodeAlpacaDataset(VisualTextBase):
    """A consistent interface for the CodeAlpaca dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        problem = sample["instruction"]
        question = f"{problem}"
        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=sample["output"],
            sample_info={"dataset": self.hf_dataname},
        )
