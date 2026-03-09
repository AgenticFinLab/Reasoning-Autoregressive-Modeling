"""
Test script for the FinAgent evaluation framework.

This script allows evaluation of different models on the Finance Agent Benchmark dataset.
It supports both API-based models (like OpenAI, Anthropic, DeepSeek) and HuggingFace models.

Usage Examples:
    # Evaluate with API model (DeepSeek)
    python examples/uTEST/test_finagent_eval.py -m deepseek/deepseek-chat -t api -s EXPERIMENT/FinAgent

    # Evaluate with HuggingFace model (Qwen/Qwen2.5-1.5B-instruct)
    python examples/uTEST/test_finagent_eval.py -m Qwen/Qwen2.5-1.5B-instruct -t huggingface -s EXPERIMENT/FinAgent

Arguments:
    -m, --model: Model name to use for evaluation
    -t, --type: Model type ('api' for API-based models or 'huggingface' for HuggingFace models)
    -s, --save-dir: Directory to save evaluation logs and trajectories
"""

# Python built-in packages
import asyncio
import argparse
import traceback

# Third-party packages
from dotenv import load_dotenv

# Internal imports
from lmbase.eval.finagent import FinAgentEvaluator
from lmbase.dataset.registry import get


def test_evaluation(
    model_name="deepseek/deepseek-chat",
    model_type="api",
    save_dir="./logs",
    max_samples=5,
):
    """Test the evaluation functionality by iterating through dataset samples."""
    load_dotenv()
    print(f"\nTesting evaluation with model: {model_name} and type: {model_type}...")

    try:
        evaluator = FinAgentEvaluator(model_name=model_name, model_type=model_type)

        # Get the finagent dataset
        config = {"data_name": "finagent", "data_path": "./EXPERIMENT/data/finagent"}
        dataset = get(config, split="train")

        print(f"Dataset loaded with {len(dataset)} samples")

        # Iterate through all samples (or a subset for testing)
        samples_to_test = min(len(dataset), max_samples)
        print(f"Evaluating {samples_to_test} samples...")

        results = []
        for i in range(samples_to_test):
            sample = dataset[i]
            print(f"Evaluating sample {i+1}/{samples_to_test}: {sample['main_id']}")

            try:
                # Test that the evaluation method can be called
                result = evaluator.evaluate_single_sample(sample, save_dir=save_dir)
                results.append(result)
                print(f"  ✓ Completed evaluation for sample: {sample['main_id']}")

                # Print model output for debugging
                print(
                    f"  Generated answer: {result.get('generated_answer', 'No answer found')[:200]}..."
                )

            except Exception as e:
                print(f"  ✗ Failed to evaluate sample {sample['main_id']}: {str(e)}")
                traceback.print_exc()  # Print full traceback for debugging
                continue

        print(f"\nCompleted evaluation of {len(results)} samples successfully")
        print(
            f"Results keys in first result: {list(results[0].keys()) if results else 'No results'}"
        )

        # Calculate and display overall metrics
        if results:
            metrics = evaluator.calculate_metrics(results)
            print(f"\nOverall Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Test FinAgent Evaluation Framework")
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        default="./logs",
        help="Directory to save evaluation logs and trajectories",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek/deepseek-chat",
        help="Model name to use for evaluation (e.g., deepseek/deepseek-chat, openai/gpt-4o)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="api",
        choices=["api", "huggingface"],
        help="Model type: 'api' for API-based models (openai/, anthropic/, etc.) or 'huggingface' for HuggingFace models",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Maximum number of samples to evaluate (useful for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Testing FinAgent Evaluation Framework")
    print("=" * 50)

    args = parse_args()

    success = test_evaluation(
        model_name=args.model,
        model_type=args.type,
        save_dir=args.save_dir,
        max_samples=args.max_samples,
    )

    if success:
        print(
            f"\n✓ Tests completed! The FinAgent evaluation framework processed samples successfully."
        )
        print(f"✓ Model used: {args.model}")
        print(f"✓ Model type: {args.type}")
        print(f"✓ Samples evaluated: {args.max_samples}")
        print(f"\nLogs and trajectories are saved to: {args.save_dir}")
    else:
        print("\n✗ Tests failed!")

    exit(0 if success else 1)
