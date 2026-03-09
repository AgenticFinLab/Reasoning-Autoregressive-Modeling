"""
Independent test for FinanceBench dataset loading and RAG evaluation.

This script demonstrates a minimal RAG pipeline using LlamaIndex with DeepSeek LLM
and HuggingFace embeddings on FinanceBench financial document QA tasks.

Usage:
    # Install dependencies
    pip install llama-index llama-index-llms-openai-like sentence-transformers

    # Set environment variable
    export DEEPSEEK_API_KEY="your-deepseek-api-key"

    # Run the test
    python examples/uTEST/test_data/test_financebench.py

Requirements:
    - DEEPSEEK_API_KEY environment variable must be set
    - PDF documents will be downloaded to EXPERIMENT/data/financebench/documents/
"""

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.embeddings import resolve_embed_model

from lmbase.dataset import registry as dataset_registry


def run():
    """
    Load FinanceBench and run RAG evaluation using LlamaIndex.
    """
    # Load dataset
    ds = dataset_registry.get(
        {
            "data_name": "financebench",
            "data_path": "EXPERIMENT/data/financebench",
        },
        "train",
    )
    print(f"Dataset loaded with {len(ds)} samples")

    # Configure LlamaIndex settings with DeepSeek
    Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        is_chat_model=True,
        temperature=0,
    )
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # Process first sample as demo
    sample = ds[0]
    print(f"\nProcessing sample: {sample['main_id']}")
    print(f"Question: {sample['question'][:200]}...")
    print(f"Ground truth: {sample['groundtruth']}")

    # Get local PDF path
    local_doc_path = sample["sample_info"]["local_doc_path"]
    if local_doc_path is None or not os.path.exists(local_doc_path):
        print(f"PDF not available for sample {sample['main_id']}")
        return

    # Create index from the PDF document
    doc_dir = os.path.dirname(local_doc_path)
    doc_filename = os.path.basename(local_doc_path)

    # Load only the specific PDF for this sample
    reader = SimpleDirectoryReader(input_files=[local_doc_path])
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document chunks from {doc_filename}")

    # Build vector index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=5)

    # Query with the question
    question = sample["question"].strip()
    response = query_engine.query(question)

    print(f"\n--- RAG Response ---")
    print(f"Answer: {response.response}")
    print(f"\n--- Ground Truth ---")
    print(f"Expected: {sample['groundtruth']}")
    print(f"\n--- Justification ---")
    print(f"{sample['cot_answer'][:500]}...")


if __name__ == "__main__":
    run()
