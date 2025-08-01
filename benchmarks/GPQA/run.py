#!/usr/bin/env python3
"""
Run a GPQA evaluation with ReasonCompletionSampler.

Only the essentials are kept:
* single sampler  – ReasonCompletionSampler
* single eval     – GPQAEval
* minimal CLI     – just what the sampler/eval need
"""

import argparse
import os
from reason_completion_sampler import ReasonCompletionSampler
from gpqa_eval import GPQAEval


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run sampling and evaluations using different samplers and evaluations.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--examples", type=int, help="Number of examples to use (overrides default)")
    parser.add_argument("--api_url", type=str, required=True, help="The API endpoint of the model to evaluate (e.g., http://127.0.0.1:8000/v1)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "EMPTY"), help="API key for the model. If not provided, reads from environment variable API_KEY.")
    parser.add_argument("--model", type=str, required=True, help="Model name, used for logging and API payload.")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of concurrent workers")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens the model can generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature to control randomness")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold.")
    parser.add_argument("--presence_penalty", type=float, default=None, help="Penalty for repeating tokens (encourages diversity)")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds for each API call. Default: 3600.")
    parser.add_argument("--n", type=int, default=8, help="repeat times for each question,")
    parser.add_argument("--evaluation_save_dir", type=str, default=None,  help="Path where this run’s results will be stored. If omitted, the script will create a temporary directory. If an existing `evaluation.json` is found, the run **automatically resumes** from previously evaluated samples.",
)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(args)

    #Build sampler – assume the class accepts these named params
    sampler = ReasonCompletionSampler(
        args=args
    )

    # Number of questions and repeats are trimmed in debug mode
    num_examples = 5 if args.debug else args.examples
    n_repeats = 1 if args.debug else args.n

    # Instantiate the evaluator    
    gpqa_eval = GPQAEval(n_repeats=n_repeats, num_examples=num_examples, args=args)
    gpqa_eval(sampler)
         

if __name__ == "__main__":
    main()
