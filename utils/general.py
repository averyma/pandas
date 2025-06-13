import torch
import numpy as np
from datetime import datetime
import os, time, random, re, yaml, gc, logging

from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_logging(args):
    """Sets up logging to both a file and the console."""

    os.makedirs(args.directory, exist_ok=True)
    log_filename = f"{args.directory}/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")
    logging.info("Logging arguments...")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model_path(model_name):
    model_weight = './model-weights'
    model_paths = {
        'Llama-Guard-3-8B': f'{model_weight}/Llama-Guard-3-8B',
        'Meta-Llama-3.1-8B-Instruct': f'{model_weight}/Meta-Llama-3.1-8B-Instruct',
        'GLM-4-9B-Chat': f'{model_weight}/glm-4-9b-chat',
        'Qwen2.5-7B-Instruct': f'{model_weight}/Qwen2.5-7B-Instruct',
        'openchat-3.6-8b': f'{model_weight}/openchat-3.6-8b-20240522',
        'OLMo-2-1124-7B-Instruct': f'{model_weight}/OLMo-2-1124-7B-Instruct',
    }

    return model_paths.get(model_name, "Model name not recognized")

def log_gpu_memory_usage(current_gpu_id):
    """Log the current GPU memory usage."""
    t_in_gb = torch.cuda.get_device_properties(current_gpu_id).total_memory / (1024 ** 3)
    r_in_gb = torch.cuda.memory_reserved(current_gpu_id) / (1024 ** 3)
    a_in_gb = torch.cuda.memory_allocated(current_gpu_id) / (1024 ** 3)
    torch.cuda.empty_cache()
    gc.collect()
    logging.info(f" *GPU-{current_gpu_id}: Total memory: {t_in_gb:.2f} GB, Reserved memory: {r_in_gb:.2f} GB, Allocated memory: {a_in_gb:.2f} GB, Free memory: {(r_in_gb - a_in_gb):.2f} GB")

def load_model_and_tokenizer(model_path, current_gpu_id, device):
    """Load the tokenizer and model."""
    if current_gpu_id ==0:
        logging.info(f" *Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if current_gpu_id ==0:
        logging.info(f" *GPU-{current_gpu_id}: tokenizer loaded")

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device).eval()

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if current_gpu_id ==0:
        logging.info(f" *GPU-{current_gpu_id}: model loaded in {time.time() - start:.2f}s")

    return model, tokenizer

def load_tokenizer(model_path):
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def generate_shot_list(n):
    result = [0]  # Start with 0
    i = 0
    while 2**i <= n:
        result.append(2**i)
        i += 1
    if result[-1] != n:  # Ensure the number 'n' is only added once
        result.append(n)
    return result

def get_summary(yaml_path):
    with open(yaml_path, 'r') as file:
        summary = yaml.safe_load(file)
    return summary

def remove_linebreak_spaces(text):
    # remove linebreaks (\r and \n) from the output text
    removed_lb = re.sub(r'\s*\n\s*', ' ', text)
    # remove multiple spaces and add a newline at the end
    output_text = re.sub(r'\s+', ' ', removed_lb.strip()) + '\n'
    return output_text

def generate_tokenized_chat(jailbreak_prompt, tokenizer):
    tokenized_chat = tokenizer.apply_chat_template(jailbreak_prompt,
                                                    add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt")
    return tokenized_chat