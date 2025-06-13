# PANDAS: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling

This repository contains the implementation of our ICML 2024 paper *PANDAS: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling* ([paper](https://arxiv.org/pdf/2502.01925)).

## Overview:
![PANDAS Overview](figures/overview.png)
> Many-shot jailbreaking prefixes the malicious target prompt with hundreds of fabricated conversational exchanges, making it appear as though the model has already complied with harmful instructions.

**PANDAS** improves many-shot jailbreaking using:
1. **Positive Affirmation** — Inserted before the next malicious question to reinforce instruction-following behaviors (green).
2. **Negative Demonstration** — A refusal followed by a correction phrase, steering the model away from refusal (red).
3. **Adaptive Sampling** — Malicious demonstrations are selected based on the topics of the target prompt.

---

## Getting Started:
### 1. Request Dataset Access
Submit a request to access the [ManyHarm dataset](https://huggingface.co/datasets/avery-ma/ManyHarm). We typically respond within 1–2 days. After approval, place the downloaded `.csv` files in: `./dataset/ManyHarm`
### 2. Model Weights
We use HuggingFace-compatible model weights such as [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). Ensure your model supports chat templates and correct role names.

---

## Usage
### Run vanilla many-shot jailbreaking:
```
python3 main_jailbreak.py --dataset harmbench -m Meta-Llama-3.1-8B-Instruct -c all \
--positive_affirmation off --negative_demonstration off --adaptive_sampling off \
--seed 0 --num_restart 3 --max_shot 256 -d <path to the result directory>
```

### Run PANDAS:
```
python3 main_jailbreak.py --dataset harmbench -m Meta-Llama-3.1-8B-Instruct -c all \
--positive_affirmation random --negative_demonstration first --adaptive_sampling percentage_of_1 \
--seed 0 --num_restart 3 --max_shot 256 -d <path to the result directory>
```

### Run Bayesian Optimization (e.g. on `abuse-platform` prompts):
```
python3 main_bayes_opt.py -c abuse-platform -s 64 -n 5 -i 50 --dataset advbench -m Meta-Llama-3.1-8B-Instruct \
--init_points 10 --debug 0 --using_logit 0 --char 1 -d <path to the result directory>
```

## Citation
If you find this useful in your research, please consider citing:

```
@inproceedings{ma2025pandas,
  title={{PANDAS}: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling},
  author={Ma, Avery and Pan, Yangchen and Farahmand, Amir-massoud},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2025},
}
```


## License
MIT License
