"""
Interface of the hfcodealpaca dataset.

Dataset Source: https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K

Description:
    HuggingFace's version of CodeAlpaca with ~20,000 code instruction-following
    examples. Split into train (18,000) and test (2,000) sets.

Size: ~20,000 rows

Splits:
    - train: ~18,000 examples
    - test: ~2,000 examples

Features:
    - instruction/prompt: Natural language instruction
    - output/completion: Code solution

License: Creative Commons
Language: English (instructions), Multiple programming languages (code)
"""

from lmbase.dataset.base import TextCodeSample, VisualTextBase


class CodeAlpacaDataset(VisualTextBase):
    """A consistent interface for the hfCodeAlpaca dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample; handle both instruction/output and prompt/completion schemas.
        if "instruction" in sample:
            problem = sample["instruction"]
            groundtruth = sample.get("output", "")
        else:
            problem = sample.get("prompt", "")
            groundtruth = sample.get("completion", "")
        question = f"{problem}"
        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            sample_info={"dataset": self.hf_dataname},
        )
