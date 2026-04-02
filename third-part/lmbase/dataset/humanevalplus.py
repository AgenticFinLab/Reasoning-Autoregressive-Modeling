"""
Interface of the HumanEvalPlus dataset.

Dataset Source: https://huggingface.co/datasets/evalplus/humanevalplus

Description:
    An enhanced version of HumanEval with significantly more test cases
    (averaging 764.1 tests per problem vs 9.6 in original HumanEval).
    Same 164 problems as HumanEval but with more rigorous test coverage.

Size: ~2.9 MB, 164 rows

Splits:
    - test: 164 examples (test set only)

Features:
    - task_id: Unique task identifier
    - prompt: Function signature and docstring
    - canonical_solution: Reference solution
    - test: Enhanced test cases (764+ per problem)
    - entry_point: Function entry point name

License: Apache-2.0
Language: Python
Paper: EvalPlus: Rigorous Evaluation of LLM-based Code Generation
"""

from lmbase.dataset.base import TextCodeSample, VisualTextBase


class HumanEvalPlusDataset(VisualTextBase):
    """A consistent interface for the HumanEvalPlus dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        problem = sample["prompt"]
        question = (
            "Please complete the following function according to the given "
            "requirements and test examples.\n" + f"{problem}"
        )
        solution = sample["canonical_solution"]
        test_str = sample["test"]
        index = test_str.find("def")
        test_cases = test_str[index:] if index != -1 else test_str

        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=solution,
            groundtruth=solution,
            test_cases=test_cases,
            sample_info={
                "dataset": self.hf_dataname,
                "task_id": sample["task_id"],
                "entry_point": sample["entry_point"],
            },
        )
