import os
import json
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from scripts import extract_boxed
from scripts import equation_equivilancy
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-base_output_dir", type=str)

    return parser.parse_args()
class VLLMPhysicsEvaluator:
    def __init__(self, base_output_dir, dataset_dir, num_workers=4, timeout=30):
        self.base_output_dir = base_output_dir  # Directory storing all LLM outputs
        self.dataset_dir = dataset_dir  # Directory storing all datasets
        self.num_workers = num_workers  # Number of processes for parallel processing
        self.timeout = timeout  # Timeout for a single data entry (seconds)

    def evaluate_entry(self, entry,judge_api_url, judge_api_key,judge_model):
        """Evaluate a single data entry for parallel processing. If stuck or failed, return default error data."""
        try:
            gen_data, dataset_data = entry
            entry_id = gen_data.get("id")

            llm_answers = gen_data.get("llm_answers")
            dataset_answers = dataset_data.get("final_answers", [])
            # Extract the final answers from the LLM
            llm_final_answers = extract_boxed.extract_final_answer_allform(llm_answers, answer_type='list')

            if not llm_final_answers:
                flattened_answers = []
            else:
                flattened_answers = (
                    [item for sublist in llm_final_answers for item in sublist]
                    if isinstance(llm_final_answers[0], list)
                    else llm_final_answers
                )

            equivalency_results = []
            correct_count = 0
            sympy_errors_correct_llm = 0
            sympy_errors = 0

            for llm_answer in flattened_answers:
                matched = False
                for dataset_answer in dataset_answers:
                    # print(llm_answer, dataset_answer)
                    equivalency_data = equation_equivilancy.is_equiv(llm_answer, dataset_answer ,llm_answers, dataset_answers,judge_api_url, judge_api_key,judge_model, verbose=False)
                    equivalency_results.append(equivalency_data)

                    sympy_result = equivalency_data.get("sympy_result")
                    llm_result = equivalency_data.get("llm_result")

                    if sympy_result is False and llm_result is True:
                        sympy_errors_correct_llm += 1
                    if sympy_result is not None:
                        sympy_errors += 1

                    if equivalency_data.get("final_result") == True:
                        correct_count += 1
                        matched = True
                        break
                if matched:
                    continue

            total_comparisons = len(flattened_answers)
            accuracy = correct_count / total_comparisons if total_comparisons > 0 else 0.0

            return {
                "id": entry_id,
                "solution": llm_answers,
                "final_answers": flattened_answers,
                'gold':dataset_answers,
                "equivalency_results": equivalency_results,
                "accuracy": accuracy,
                'question':gen_data.get('questions'),
                'reasoning_content':gen_data.get('reason_content'),
                'llm_answers':gen_data.get('llm_answers'),
                'finish_reason':gen_data.get('finish_reason'),
                'completion_tokens':gen_data.get('completion_tokens')
            }, sympy_errors_correct_llm, sympy_errors

        except Exception as e:
            print(f"Error processing entry: {e}")
            return {
                "id": entry_id if 'entry_id' in locals() else "unknown",
                "solution": None,
                "final_answers": [],
                "equivalency_results": [],
                "accuracy": 0,  # Set accuracy to 0 in case of an error
                'question':gen_data.get('questions'),
                'reasoning_content':gen_data.get('reason_content'),
                'llm_answers':gen_data.get('llm_answers'),
                'finish_reason':gen_data.get('finish_reason'),
                'completion_tokens':gen_data.get('completion_tokens')
            }, 0, 0

    def process_single_dataset(self, llm_folder, dataset_folder,judge_api_url, judge_api_key,judge_model):
        """Process a single dataset's response.jsonl and perform evaluation."""
        dataset_output_path = llm_folder
        # os.path.join(self.base_output_dir, llm_folder, dataset_folder)
        # print("dataset_output_path:"+dataset_output_path)

        response_file = os.path.join(dataset_output_path, "response.jsonl")
        # print(response_file)
        if not os.path.exists(response_file):
            print(f"Skipping {dataset_folder} in {llm_folder}: response.jsonl not found.")
            return

        dataset_file = './text_only_dataset/'+dataset_folder+'.jsonl'
        # os.path.join(self.dataset_dir, f"{dataset_folder}.jsonl")
        # print('dataset_file:'+dataset_file)
        if not os.path.exists(dataset_file):
            print(f"Skipping {dataset_folder} in {llm_folder}: Corresponding dataset not found.")
            return

        output_jsonl = os.path.join(dataset_output_path, "final_evaluation.jsonl")
        summary_csv = os.path.join(dataset_output_path, "accuracy.csv")

        try:
            with open(response_file, "r") as gen_file, open(dataset_file, "r") as dataset_file:
                gen_lines = [json.loads(line.strip()) for line in gen_file]
            entries = list(zip(gen_lines, gen_lines))
            results = []
            accuracies = []
            with open(output_jsonl, "w") as outfile:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = {executor.submit(self.evaluate_entry, entry,judge_api_url, judge_api_key,judge_model): entry for entry in entries}
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {llm_folder}/{dataset_folder}"):
                        try:
                            result, sympy_errors_correct_llm, sympy_errors = future.result(timeout=self.timeout)
                        except TimeoutError:
                            print(f"Timeout error on dataset {dataset_folder} in {llm_folder}")
                            result = {
                                "id": "timeout_error",
                                "solution": None,
                                "final_answers": [],
                                "equivalency_results": [],
                                "accuracy": 0 , # Set to 0
                                'question':None,
                                'reasoning_content':None,
                                'llm_answers':None,
                                'finish_reason':None,
                                'completion_tokens':None
                            }
                        except Exception as e:
                            print(f"Unexpected error in {dataset_folder} of {llm_folder}: {e}")
                            result = {
                                "id": "error",
                                "solution": None,
                                "final_answers": [],
                                "equivalency_results": [],
                                "accuracy": 0 , # Set to 0
                                'question':None,
                                'reasoning_content':None,
                                'llm_answers':None,
                                'finish_reason':None,
                                'completion_tokens':None
                            }

                        
                        # json.dump(result, outfile, ensure_ascii=False)
                        outfile.write(json.dumps(result, ensure_ascii=False)+'\n')
                        results.append(result)
                        accuracies.append(result["accuracy"])

            overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            
            # with open(output_jsonl, "w") as outfile:
            #     for result in results:
            #         json.dump(result, outfile, ensure_ascii=False)
            #         outfile.write("\n")

            with open(summary_csv, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([overall_accuracy])

            print(f"Evaluation complete for {llm_folder}/{dataset_folder}. Results saved in {dataset_output_path}")

        except Exception as e:
            print(f"Critical error processing {dataset_folder} in {llm_folder}: {e}")

    def process_all_llm_outputs(self, task,judge_api_url, judge_api_key,judge_model):
        """Iterate through the base_output_dir directory and process all LLM-generated response.jsonl files in bulk."""

        self.process_single_dataset(self.base_output_dir, task,judge_api_url, judge_api_key,judge_model)  # Process each dataset sequentially

def eval_main(base_output_dir,judge_api_url, judge_api_key,judge_model):
    sub = base_output_dir.split('/')[-1]
    print(f'start evaluation task :{sub}')
    evaluator = VLLMPhysicsEvaluator(
        base_output_dir=base_output_dir,
        dataset_dir=sub,
        num_workers=64,  # Parallel evaluation of data entries
        timeout=60  # Set maximum execution time per entry
    )
    evaluator.process_all_llm_outputs(sub,judge_api_url, judge_api_key,judge_model)
# if __name__ == "__main__":
#     args = parse_args()
    
#     sub = args.base_output_dir.split('/')[-1]
#     print(f'start evaluation task :{sub}')
#     evaluator = VLLMPhysicsEvaluator(
#         base_output_dir=args.base_output_dir,
#         dataset_dir=sub,
#         num_workers=64,  # Parallel evaluation of data entries
#         timeout=1  # Set maximum execution time per entry
#     )
#     evaluator.process_all_llm_outputs(task=sub)
