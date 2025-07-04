import random
import string

def random_insert_updated(text, insert_pct):
    """Randomly insert new chars into text after selected characters."""
    num_inserts = int(len(text) * insert_pct)
    indices = random.sample(range(len(text)), num_inserts)
    for idx in sorted(indices, reverse=True):
        new_char = random.choice(string.printable)
        text = text[:idx + 1] + new_char + text[idx + 1:]
    return text


def random_swap_updated(text, swap_pct):
    """Randomly swap chars within the text with new characters."""
    num_swaps = int(len(text) * swap_pct)
    indices = random.sample(range(len(text)), num_swaps)
    for i in indices:
        new_char = random.choice(string.printable)
        text = text[:i] + new_char + text[i+1:]
    return text


def random_patch(text, patch_pct):
    """Replace a random contiguous patch."""
    patch_len = int(len(text) * patch_pct)
    start_idx = random.randint(0, len(text)-patch_len)
    patch_str = ''.join(random.choice(string.printable) for _ in range(patch_len))
    text = text[:start_idx] + patch_str + text[start_idx+patch_len:]
    return text


def adaptive_perturb_pct(text, base_pct, min_len=10, max_len=100):
    """Adapt perturbation percentage based on text length."""
    text_len = len(text)
    if text_len <= min_len:
        return base_pct / 2
    elif text_len >= max_len:
        return base_pct * 2
    else:
        return base_pct


# def smooth(prompts, perturb_pct=0.1, n=10):
#     smoothed = []
#     for prompt in prompts:
#         perturbed = [prompt]
#         for _ in range(n - 1):
#             func = random.choice([random_insert_updated, random_swap_updated, random_patch])
#             adaptive_pct = adaptive_perturb_pct(prompt, perturb_pct)
#             perturbed.append(func(prompt, adaptive_pct))
#         smoothed.append(perturbed)
#     return smoothed
def smooth(prompt, perturb_pct=0.1, n=10):
    smoothed = []
    perturbed = [prompt]
    for _ in range(n - 1):
        func = random.choice([random_insert_updated, random_swap_updated, random_patch])
        adaptive_pct = adaptive_perturb_pct(prompt, perturb_pct)
        perturbed.append(func(prompt, adaptive_pct))
    smoothed.append(perturbed)
    return smoothed