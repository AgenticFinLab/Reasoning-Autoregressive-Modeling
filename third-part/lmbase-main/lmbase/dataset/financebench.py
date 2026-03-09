"""
Interface of the FinanceBench dataset.

The dataset is from https://huggingface.co/datasets/PatronusAI/financebench
It contains financial document QA tasks with SEC filings as source documents.
Split: train
"""

import os
import json
import logging
import requests
from urllib.parse import urlparse

from datasets import load_dataset

from lmbase.dataset.base import TextSample, VisualTextBase


class FinanceBenchDataset(VisualTextBase):
    """A consistent interface for the FinanceBench dataset."""

    def __init__(
        self,
        split: str = "train",
        hf_dataname: str = None,
        config: dict = None,
    ):
        self.data_path = config["data_path"]
        self.document_path = os.path.join(self.data_path, "documents")
        os.makedirs(self.document_path, exist_ok=True)

        # Track failed PDF downloads
        self.failed_downloads = []

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            self.hf_dataset = load_dataset(self.hf_dataname, split=self.split)

        logging.info(
            "   - Mapping samples to lmbase format, i.e., lmbase.dataset.base.TextSample"
        )

        # Check if hf_dataset is a DatasetDict (contains multiple splits)
        if hasattr(self.hf_dataset, "keys"):
            if self.split in self.hf_dataset:
                split_dataset = self.hf_dataset[self.split]
                column_names = split_dataset.column_names

                self.hf_dataset = split_dataset.map(
                    self.batch_format,
                    batched=True,
                    batch_size=100,
                    load_from_cache_file=True,
                    remove_columns=column_names,
                )
            else:
                raise ValueError(
                    f"Split '{self.split}' not found in dataset. Available splits: {list(self.hf_dataset.keys())}"
                )
        else:
            column_names = self.hf_dataset.column_names

            self.hf_dataset = self.hf_dataset.map(
                self.batch_format,
                batched=True,
                batch_size=100,
                load_from_cache_file=True,
                remove_columns=column_names,
            )

        # Save some demo samples to the dataset folder (check if file exists first)
        demo_file = os.path.join(self.data_path, f"{self.split}-demo-samples.json")
        if not os.path.exists(demo_file):
            self.save_example_samples(num_samples=20)

        # Save failed downloads to JSON file
        if len(self.failed_downloads) > 0:
            failed_file = os.path.join(self.data_path, "failed-downloaded-sample.json")
            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(self.failed_downloads, f, ensure_ascii=False, indent=2)
            logging.info(
                "   - Saved %d failed downloads to %s",
                len(self.failed_downloads),
                failed_file,
            )

    def download_pdf(self, url: str, sample_id: str) -> str:
        """
        Download a PDF document from URL and save to local path.

        Args:
            url (str): The URL of the PDF document.
            sample_id (str): Sample identifier for naming the file.

        Returns:
            str: Local path to the saved PDF, or None if download failed.
        """
        # Extract filename from URL or use sample_id
        parsed_url = urlparse(url)
        url_filename = os.path.basename(parsed_url.path)

        if url_filename.endswith(".pdf"):
            filename = f"{sample_id}_{url_filename}"
        else:
            filename = f"{sample_id}.pdf"

        save_path = os.path.join(self.document_path, filename)

        # Skip if already downloaded
        if os.path.exists(save_path):
            return save_path

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(response.content)

            logging.info("   - Downloaded PDF: %s", filename)
            return save_path

        except Exception as e:
            logging.warning(
                "Failed to download PDF for sample %s from %s: %s",
                sample_id,
                url,
                e,
            )
            # Record failed download
            self.failed_downloads.append(
                {
                    "sample_id": sample_id,
                    "url": url,
                    "error": str(e),
                }
            )
            return None

    def to_format(self, sample: dict):
        """Convert raw sample to standardized format."""
        sample_id = sample["financebench_id"]
        self.idx += 1

        question = sample["question"].strip()

        # Download the PDF document
        doc_link = sample["doc_link"]
        local_doc_path = None
        if doc_link:
            local_doc_path = self.download_pdf(doc_link, sample_id)

        # Format question
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}\n"

        # Use answer as groundtruth
        groundtruth = str(sample["answer"]).strip()

        # Use justification as chain-of-thought answer (ensure not None)
        cot_answer = sample["justification"]
        if cot_answer is None:
            cot_answer = ""

        return TextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "company": sample["company"],
                "doc_name": sample["doc_name"],
                "question_type": sample["question_type"],
                "question_reasoning": sample["question_reasoning"],
                "domain_question_num": sample["domain_question_num"],
                "dataset_subset_label": sample["dataset_subset_label"],
                "evidence": sample["evidence"],
                "gics_sector": sample["gics_sector"],
                "doc_type": sample["doc_type"],
                "doc_period": sample["doc_period"],
                "doc_link": doc_link,
                "local_doc_path": local_doc_path,
            },
        )
