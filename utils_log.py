import wandb
import numpy as np

class wandbLogger(object):
    def __init__(self, args):
        self.wandb_log = wandb.init(name=args.directory.split("/")[-1], project=args.wandb_project,
                                    dir=args.directory,
                                    resume=True,
                                    reinit=True)
        self.wandb_log.config.update(args, allow_val_change=True)

    def upload(self, asr_llm, asr_refusal, all_shots):
        for i, shots in enumerate(all_shots):
            wandb_dict = {
                f'ASR_L/{shots}shot': asr_llm[i],
                f'ASR_R/{shots}shot': asr_refusal[i],
            }
            wandb.log(wandb_dict)

    def upload_BO(self, init_score, best_score, best_prob, score_trajectory):

        for i, score in enumerate(score_trajectory):
            wandb.log({'score': score}, step=i+1)

        wandb_dict = {
            f'init_score': init_score,
            f'best_score': best_score,
            f'best_prob': best_prob,
        }
        wandb.log(wandb_dict, step=len(score_trajectory))

    def finish(self):
        wandb.finish()