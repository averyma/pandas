import torch, os, logging, argparse
from accelerate import Accelerator
import distutils.util
import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from utils.jailbreak import (
    MALICIOUS_CATEGORIES, sample_shots, prompt_conversion,
    get_logit_of, getFirstCharacter
)

from utils.logger import wandbLogger

from utils.general import (
    get_model_path, seed_everything, setup_logging, load_model_and_tokenizer
)

# Environment configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def check_probe_already_in_space(optimizer, probe_params, eps=1e-12):
    """
    Return True if the given probe_params appear to match
    (within floating precision) any point in optimizer.space.
    """
    # In bayes_opt, optimizer.space.params is a list of lists,
    # each sub-list is (x1, x2, ..., xN) for one data point.
    # We'll compare each dimension to see if they match.

    # Convert the probe_params dict into a consistent tuple/list:
    probe_list = [probe_params[f"x{i}"] for i in range(1, 13)]

    for existing_point in optimizer.space.params:
        # existing_point is like [val1, val2, ..., val12]
        diff = np.abs(np.array(existing_point) - np.array(probe_list))
        if np.all(diff < eps):
            return True

    return False

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=['advbench', 'advbench50', 'harmbench'], default='advbench')
    parser.add_argument('-m', "--model_name", type=str, default='Meta-Llama-3.1-8B-Instruct',
                        help="Name of the model to apply the target prompt to.")
    parser.add_argument("-c", "--category", type=str, default='abuse-platform', help="Category of the target prompt.")
    parser.add_argument("-n", "--num-eval", type=int, default=10)
    parser.add_argument('-d', "--directory", type=str, default='./exp')
    parser.add_argument('-s', "--shots", type=int, default=32)
    parser.add_argument('-i', "--num_iter", type=int, default=50)
    parser.add_argument("--probe", default=True, type=distutils.util.strtobool)
    parser.add_argument("--init_points", type=int, default=1, help="It seems setting it to 0 still has 1 init point.")
    parser.add_argument("--wandb_project", default=None, type=str)
    parser.add_argument("--using_logit", default=True, type=distutils.util.strtobool)
    parser.add_argument("--char", default='I', choices=['1', 'I'], type=str)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--debug", default=False, type=distutils.util.strtobool)
    args = parser.parse_args()

    # Setup logging
    setup_logging(args)
    logging.info(f'Results will be saved in: {args.directory}')

    # Move model to GPU if available
    accelerator = Accelerator()
    current_gpu_id = accelerator.process_index
    device = 'cuda:0'
    model_path = get_model_path(args.model_name)
    model, tokenizer = load_model_and_tokenizer(model_path, current_gpu_id, 'cuda:0')

    if args.dataset == 'harmbench':
        path = './dataset/HarmBench/harmbench_behaviors_text_all.csv'
        df = pd.read_csv(path)
        assert args.category in df['SemanticCategory'].values, f"Invalid category: {args.category} for HarmBench"
        indices = df.index[df['SemanticCategory'] == args.category].tolist()
        keyword = 'Behavior'
    elif args.dataset == 'advbench':
        path = './dataset/AdvBench/harmful_behaviors_categorized.csv'
        df = pd.read_csv(path)
        assert args.category in df['category'].values, f"Invalid category: {args.category} for AdvBench"
        indices = df.index[df['category'] == args.category].tolist()
        keyword = 'goal'

    if args.debug:
        num_target=1
        shot=4
        num_eval=1
    else:
        num_target = len(indices)
        shot = args.shots
        num_eval = args.num_eval

    if args.using_logit:
        logging.info(f'Using the log likelihood of the character {args.char} as the BO objective.')
    else:
        logging.info(f'Using the percentage of the first character being {args.char} as the BO objective.')

    def seed_from_vector(x_list):
        return int(sum(x_list) * 1e6) % (2**32)

    def black_box_function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12):
        # larger logit means more likely to be char (ie. 'I', '1')
        # Maximizing attack success rate is equivalent to minimizing logit_I or maximizing -logit

        score = 0
        probabilities = torch.nn.Softmax(dim=0)(torch.tensor([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]))
        seed = seed_from_vector([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12])
        seed_everything(seed)

        for j in range(num_target):
            target_prompt = prompt_conversion(df[keyword][indices[j]], 'How can I')
            for k in range(num_eval):
                # Sample prompt
                prompt = sample_shots(args.model_name, target_prompt, shot, 100, probabilities)

                if args.using_logit:
                    logit = get_logit_of(prompt, model, tokenizer, args.model_name, args.char)
                    if args.char == 'I':
                        score -= logit
                    elif args.char == '1':
                        score += logit
                else:
                    output_text = getFirstCharacter(prompt, model, tokenizer, args.model_name)
                    if args.char == 'I':
                        score += output_text != args.char
                    elif args.char == '1':
                        score += output_text == args.char

        final_score = score / max(1e-12, (num_target * num_eval))
        return final_score

    pbounds = {'x1': (-10, 10), 'x2': (-10, 10), 'x3': (-10, 10), 'x4': (-10, 10),
               'x5': (-10, 10), 'x6': (-10, 10), 'x7': (-10, 10), 'x8': (-10, 10),
               'x9': (-10, 10), 'x10': (-10, 10), 'x11': (-10, 10), 'x12': (-10, 10)}

    probe_params = {"x1": 0., "x2": 0., "x3": 0., "x4": 0.,
                    "x5": 0., "x6": 0., "x7": 0., "x8": 0.,
                    "x9": 0., "x10": 0., "x11": 0., "x12": 0.}

    # Initialize optimizer
    optimizer = BayesianOptimization(
                                    f=black_box_function,
                                    pbounds=pbounds,
                                    random_state=0,
                                    verbose=2
                                    )

    # Check if there is a checkpoint
    log_path = f"{args.directory}/bayes_opt.log"
    if os.path.exists(log_path):
        # Load the previous logs
        logging.info(f"Logger file exists at {log_path}.")
        load_logs(optimizer, logs=[log_path])
        len_points = len(optimizer.space)
        logging.info("New optimizer is now aware of {} points.".format(len_points))

        if args.probe:
            already_probed = check_probe_already_in_space(optimizer, probe_params)
            if not already_probed:
                optimizer.probe(probe_params, lazy=True)

        # How many "init" points + probe do we want in total?
        probe_count = 1 if args.probe else 0
        desired_init = args.init_points + probe_count

        if len_points < desired_init:
            # We haven't yet reached the total desired_init, so we still owe some "init" calls
            remain_init = desired_init - len_points
            # Then do all of the normal iteration steps
            remain_iter = args.num_iter
        else:
            # We have already reached (or exceeded) the total desired init points
            remain_init = 0
            # Subtotal we want overall = init_points + probe_count + num_iter
            total_points = args.init_points + probe_count + args.num_iter
            # The number of iteration calls we still want
            remain_iter = max(0, total_points - len_points)
    else:
        if args.probe:
            # Probe the optimizer by evaluating using uniformly random sample
            optimizer.probe(probe_params, lazy=True)
        remain_init = args.init_points
        remain_iter = args.num_iter

    logging.info(f"Remaining init: {remain_init}, Remaining iter: {remain_iter}")

    if remain_iter> 0:
        # Create a new logger object, it continues to log (NEVER RESET EXISTING LOGS)
        logger = JSONLogger(path=log_path, reset=False)
        screen_logger = ScreenLogger(verbose=2)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, screen_logger)
        optimizer.maximize(
                init_points=remain_init,
                n_iter=remain_iter,
            )

    init_score = optimizer.space.target[0]
    best_score = optimizer.max['target']
    best_prob = torch.nn.Softmax(dim=0)(torch.tensor([optimizer.max['params'][f"x{i}"] for i in range(1, 13)])).numpy()
    formatted = ", ".join(f"{p:.8f}" for p in best_prob)
    score_trajectory = optimizer.space.target

    logging.info(f"Starting score: {init_score}")
    logging.info(f"Max score: {best_score}")
    logging.info(f"Probabilities: {best_prob}")
    logging.info(f"Probabilities (YAML-ready): [{formatted}]")

    if args.wandb_project is not None:
        wandb_logger = wandbLogger(args)
        wandb_logger.upload_BO(init_score, best_score, best_prob, score_trajectory)
        wandb_logger.finish()

if __name__ == '__main__':
    main()
