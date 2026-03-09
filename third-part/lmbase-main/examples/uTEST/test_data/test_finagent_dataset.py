"""
Test script for the FinAgent dataset integration.
"""

import os
from lmbase.dataset.registry import get


def test_finagent_dataset():
    """Test the FinAgent dataset integration."""
    print("Testing FinAgent dataset integration...")

    # Configuration for the dataset
    config = {
        "data_name": "finagent",
        "data_path": "./EXPERIMENT/finagent_test",  # Temporary path for test samples
    }

    try:
        # Load the dataset
        print("Loading FinAgent dataset (train split)...")
        dataset = get(config, split="train")

        print(f"Successfully loaded dataset with {len(dataset)} samples")

        # Print info about the first sample
        if len(dataset) > 0:
            first_sample = dataset[0]
            print("\nFirst sample info:")
            print(f"  ID: {first_sample.main_id}")
            print(f"  Question: {first_sample.question[:100]}...")  # First 100 chars
            print(f"  Answer: {first_sample.cot_answer[:100]}...")  # First 100 chars
            print(f"  Ground Truth: {first_sample.groundtruth}")
            print(f"  Sample Info: {first_sample.sample_info}")

        print("\n✓ FinAgent dataset integration test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Error loading FinAgent dataset: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_finagent_dataset()
    if success:
        print(
            "\nIntegration successful! The FinAgent dataset is now available in lmbase."
        )
    else:
        print("\nIntegration failed!")
    exit(0 if success else 1)
