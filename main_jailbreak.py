import argparse
import distutils.util
import logging
import os
import pandas as pd
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

# If these are internal utility modules, keep them;
# otherwise remove if not used or rename for clarity.
from utils import (get_model_path, seed_everything, setup_logging, generate_shot_list,
                   get_summary, log_gpu_memory_usage, load_model_and_tokenizer,
                   remove_linebreak_spaces, generate_tokenized_chat)
from utils_jailbreak import (MALICIOUS_CATEGORIES, get_target_prompt,
                             sample_shots, contains_refusal_phrase,
                             negative_demonstration, positive_affirmation,
                             prompt_conversion)
from utils_prompt_format import AgentType
from utils_ifsj import sample_shots_ifsj
from utils_safety import evaluate_safety
from utils_log import wandbLogger
from utils_defence import modify_jailbreak_prompt_with_defence

# Environment configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Function to save the checkpoint
def save_checkpoint(checkpoint_path, result_llm, result_refusal, line_idx):
    np.savez(
        checkpoint_path,
        result_llm=result_llm,
        result_refusal = result_refusal,
        line_idx=line_idx)
    logging.info(f"Checkpoint saved at the {line_idx}-th target prompt")

# Function to load the checkpoint
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path)
        line_idx = checkpoint["line_idx"]
        result_llm = checkpoint["result_llm"]
        result_refusal = checkpoint["result_refusal"]

        logging.info(
            f"Resuming from checkpoint at line {line_idx}"
        )
        return result_llm, result_refusal, line_idx
    else:
        return None, None, 0  # Start fresh if no checkpoint


def generate_and_save_responses(target_model, target_tokenizer, judge_model, judge_tokenizer, args, target_prompt_lines, result_llm, result_refusal):
    """
    Generate responses for each target prompt and save the results.

    """
    if args.adaptive_sampling == 'off':
        distribution = [1/12] * 12
    elif args.adaptive_sampling == 'percentage_of_I':
        dist_summary = get_summary('./configs/percentage_of_I.yaml')
    elif args.adaptive_sampling == 'percentage_of_1':
        dist_summary = get_summary('./configs/percentage_of_1.yaml')

    if args.dataset in ['advbench50', 'advbench']:
        csv_file = './dataset/AdvBench/harmful_behaviors_categorized.csv'
        df_all = pd.read_csv(csv_file)
    elif args.dataset == 'harmbench':
        csv_file = './dataset/HarmBench/harmbench_behaviors_text_all.csv'
        df_all = pd.read_csv(csv_file)

    for line_idx in target_prompt_lines:
        target_prompt = get_target_prompt(args.dataset, line_idx, category=args.category)
        if args.category == 'all':
            if args.dataset in ['advbench50', 'advbench']:
                idx_in_all = df_all.index[df_all['goal'] == target_prompt][0]
                cat = df_all['category'][idx_in_all]
            elif args.dataset == 'harmbench':
                idx_in_all = df_all.index[df_all['Behavior'] == target_prompt][0]
                cat = df_all['SemanticCategory'][idx_in_all]
        else:
            cat = args.category

        if not target_prompt:
            continue
        if args.convert:
            target_prompt = prompt_conversion(target_prompt, 'How can I')
        logging.info(f'[{line_idx}/{target_prompt_lines[-1]}] Processing target prompt: {target_prompt}. Category: {cat}')

        if args.adaptive_sampling != 'off':
            if args.dataset == 'advbench50':
                distribution = np.array(dist_summary[args.model_name]['advbench'][cat])
            else:
                distribution = np.array(dist_summary[args.model_name][args.dataset][cat])
            distribution /= distribution.sum()

        result_llm_per_line = []
        result_refusal_per_line = []
        for shots in args.all_shots:
            unsafe_llm = 0
            unsafe_refusal = 0
            output_text=''
            for restart in range(args.num_restart):
                if unsafe_llm and unsafe_refusal:
                    continue
                seed_everything(args.seed*1000000 + line_idx*1000 + shots*10 + restart)
                logging.info(
                    f'Processing {shots} shots, [{restart}/{args.num_restart}] restart, seed={args.seed*1000000 + line_idx*1000 + shots*10 + restart}'
                )
                if args.ifsj:
                    jailbreak_prompt = sample_shots_ifsj(args.model_name, target_prompt, shots,
                                                         target_model, target_tokenizer,
                                                         args.max_response_limit, steps=args.T_ifsj)
                else:
                    jailbreak_prompt = sample_shots(args.model_name, target_prompt, shots,
                                                    args.max_response_limit, distribution)

                # NOTE: positive affirmation should always be applied before negative demonstration
                if args.positive_affirmation != 'off' and shots>=1:
                    jailbreak_prompt = positive_affirmation(jailbreak_prompt, args.positive_affirmation)

                if args.negative_demonstration != 'off' and shots>=1:
                    jailbreak_prompt = negative_demonstration(jailbreak_prompt, args.negative_demonstration)

                if args.defence != 'off':
                    jailbreak_prompt = modify_jailbreak_prompt_with_defence(jailbreak_prompt, args.defence)

                tokenized_chat = generate_tokenized_chat(jailbreak_prompt, target_tokenizer)
                tokenized_chat = {k: v.to(target_model.device) for k, v in tokenized_chat.items()}
                prompt_len = len(tokenized_chat['input_ids'][0])

                # INFERENCE:
                with torch.inference_mode():
                    try:
                        output_tokens = target_model.generate(
                                            **tokenized_chat, temperature=args.temperature,
                                            max_new_tokens=args.max_new_tokens,
                                            pad_token_id=target_tokenizer.eos_token_id,
                                            num_beams=1, do_sample=False, top_p=1.0)[0]
                        raw_output_text = target_tokenizer.decode(output_tokens[prompt_len:], skip_special_tokens=True)
                        output_text = remove_linebreak_spaces(raw_output_text)
                    except torch.cuda.OutOfMemoryError:
                        logging.error(f"CUDA OOM Error for target line number: {line_idx}, shots: {shots}, restart: {restart}")
                        torch.cuda.empty_cache()

                # EVALUATION:
                unsafe_llm = 'unsafe' in evaluate_safety(judge_tokenizer, judge_model, AgentType.AGENT, prompt=output_text)
                unsafe_refusal = not contains_refusal_phrase(output_text)

                if args.save_input or args.save_output:
                    llm_folder = 'llm_success' if unsafe_llm else 'llm_fail'
                    refusal_folder = 'refusal_success' if unsafe_refusal else 'refusal_fail'

                    # Save input prompt
                    if args.save_input:
                        input_path = Path(args.directory, f'input/{llm_folder}/{shots}shot.txt')
                        with input_path.open('a', encoding='utf-8') as f:
                            f.write(f'{jailbreak_prompt}\n')

                        input_path = Path(args.directory, f'input/{refusal_folder}/{shots}shot.txt')
                        with input_path.open('a', encoding='utf-8') as f:
                            f.write(f'{jailbreak_prompt}\n')

                    # Save output response
                    if args.save_output:
                        output_path = Path(args.directory, f'output/{llm_folder}/{shots}shot.txt')
                        with output_path.open('a', encoding='utf-8') as f:
                            f.write(f'{output_text}\n')

                        output_path = Path(args.directory, f'output/{refusal_folder}/{shots}shot.txt')
                        with output_path.open('a', encoding='utf-8') as f:
                            f.write(f'{output_text}\n')

            result_llm_per_line.append(unsafe_llm)
            result_refusal_per_line.append(unsafe_refusal)
        result_refusal[line_idx-1] = result_refusal_per_line
        result_llm[line_idx-1] = result_llm_per_line
        save_checkpoint(f"{args.directory}/checkpoint.npz", result_llm, result_refusal, line_idx)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str, default='all', help="Category of the target prompt.")
    parser.add_argument('-m', "--model_name", type=str, default='Meta-Llama-3.1-8B-Instruct',
                        help="Name of the model to apply the target prompt to.")
    parser.add_argument("-n", "--num_restart", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--debug", default=False, type=distutils.util.strtobool)
    parser.add_argument('-d', "--directory", type=str, default='./exp')
    parser.add_argument("--dataset", type=str,
                        choices=['advbench', 'advbench50', 'harmbench'], default='advbench50')
    parser.add_argument("--convert", default=True, type=distutils.util.strtobool)
    parser.add_argument("--max_shot", type=int, default=256)
    parser.add_argument("--max_response_limit", type=int, default=100,
                        help='Max token limit for the response (default: 100), 0 means no response.')

    parser.add_argument("--save_output", default=False, type=distutils.util.strtobool)
    parser.add_argument("--save_input", default=False, type=distutils.util.strtobool)
    parser.add_argument("--wandb_project", default=None, type=str)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ifsj", default=False, type=distutils.util.strtobool)
    parser.add_argument("--T_ifsj", type=int, default=128)

    parser.add_argument('--defence', type=str, default='off')

    # PANDAS
    parser.add_argument("--positive_affirmation", default='off',
                        choices=['off', 'first', 'last', 'random', 'all'])
    parser.add_argument("--negative_demonstration", default='off',
                        choices=['off', 'first', 'last', 'random', 'all'])
    parser.add_argument("--adaptive_sampling", type=str, default='off',
                        choices=['off', 'percentage_of_I', 'percentage_of_1'], help="Distribution of shot categories.")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args)
    logging.info(f'Results will be saved in: {args.directory}')

    # Get the number of target prompts for the dataset and category
    num_target_prompt_map = get_summary('./configs/dataset_summary.yaml')

    if args.dataset in num_target_prompt_map:
        dataset_map = num_target_prompt_map[args.dataset]
        if args.category in dataset_map:
            num_target_prompt = dataset_map[args.category]
        else:
            raise ValueError(f"Unknown category: {args.category}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Determine the maximum shot count for given target model and the type of shots
    args.all_shots = generate_shot_list(args.max_shot)

    # load checkpoint: i denotes the target prompt; result is a 2d array of shape (num_target_prompt, len(args.all_shots))
    result_llm, result_refusal, line_idx = load_checkpoint(f"{args.directory}/checkpoint.npz")
    if result_llm is None:
        result_llm = np.zeros((num_target_prompt, len(args.all_shots)))
        result_refusal = np.zeros((num_target_prompt, len(args.all_shots)))
    target_prompt_lines = list(range(line_idx+1, num_target_prompt + 1)) # includes i+1 and num_target_prompt
    if target_prompt_lines != []:
        # Move model to GPU if available
        accelerator = Accelerator()
        current_gpu_id = accelerator.process_index
        device = 'cuda:0'

        # Summary of the evaluation
        logging.info(f' *GPU-{current_gpu_id}: Accelerator initialized, using device: {accelerator.device}')
        logging.info(f' *Jailbreaking {args.model_name} with {args.all_shots} shots on {args.dataset} dataset.')

        # Load target model and tokenizer
        model_path = get_model_path(args.model_name)
        target_model, target_tokenizer = load_model_and_tokenizer(model_path, current_gpu_id, device)

        # Load judge model and tokenizer
        model_path = get_model_path('Llama-Guard-3-8B')
        judge_model, judge_tokenizer = load_model_and_tokenizer(model_path, current_gpu_id, device)
        logging.info(f' *GPU-{current_gpu_id}: Model loaded: Llama-Guard-3-8B')

        # Log GPU memory usage
        log_gpu_memory_usage(current_gpu_id)

        # Ensure directories for input/output prompts exist if saving is enabled
        if args.save_input:
            for folder in ['input/llm_success', 'input/llm_fail', 'input/refusal_success', 'input/refusal_fail']:
                Path(args.directory, folder).mkdir(parents=True, exist_ok=True)

        if args.save_output:
            for folder in ['output/llm_success', 'output/llm_fail', 'output/refusal_success', 'output/refusal_fail']:
                Path(args.directory, folder).mkdir(parents=True, exist_ok=True)


        # responses are sampled from response_models
        generate_and_save_responses(target_model, target_tokenizer, judge_model, judge_tokenizer,
                                    args, target_prompt_lines, result_llm, result_refusal)

    asr_llm = np.sum(result_llm, axis=0)/result_llm.shape[0]*100
    asr_refusal = np.sum(result_refusal, axis=0)/result_refusal.shape[0]*100
    if args.wandb_project is not None:
        wandb_logger = wandbLogger(args)
        wandb_logger.upload(asr_llm, asr_refusal, args.all_shots)
        wandb_logger.finish()

    # Save the results
    np.savez(
            f"{args.directory}/result.npz",
            asr_llm=asr_llm,
            asr_refusal = asr_refusal)

if __name__ == '__main__':
    main()