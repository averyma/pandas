import torch
from utils.general import generate_tokenized_chat
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def remove_rows_and_columns(A, indices):
    # Convert the list of indices to a set for faster lookup
    indices_set = set(indices)

    # Create masks to identify which rows and columns to keep
    mask = np.array([i not in indices_set for i in range(A.shape[2])])

    # Apply the mask to remove specified rows and columns
    A_reduced = A[:,:,mask, :][:,:,:, mask]

    return A_reduced

def return_raw_attn(model, tokenizer, prompt):
    tokenized_chat = generate_tokenized_chat(prompt, tokenizer)
    tokenized_chat = {k: v.to(model.device) for k, v in tokenized_chat.items()}
    with torch.no_grad():
        torch.manual_seed(0)
        output_tokens = model.forward(**tokenized_chat, output_attentions=True)
    attn_raw = output_tokens.attentions

    return attn_raw

def return_special_token_and_role_index(tokenized_chat, model_name = 'Meta-Llama-3.1-8B-Instruct'):
    removed = 0
    raw_role_idx = []
    removed_idx = []
    if model_name == 'Meta-Llama-3.1-8B-Instruct':
        for i, token in enumerate(tokenized_chat['input_ids'][0]):
            if token >=128000:
                removed_idx.append(i)
                removed +=1
            if token == 128006:
                raw_role_idx.append(i+1-removed) # added one becasue
    elif model_name == 'GLM-4-9B-Chat':
        for i, token in enumerate(tokenized_chat['input_ids'][0]):
            if token >=151329:
                removed_idx.append(i)
                removed +=1
            if token in [151336, 151337]:
                raw_role_idx.append(i+1-removed)
    elif model_name == 'Qwen2.5-7B-Instruct':
        for i, token in enumerate(tokenized_chat['input_ids'][0]):
            if i <= 20:
                removed_idx.append(i)
                removed +=1
            else:
                if token >=151643:
                    removed_idx.append(i)
                    removed +=1
                if token == 151644:
                    raw_role_idx.append(i+1-removed)
    elif model_name == 'OLMo-2-1124-7B-Instruct':
        for i, token in enumerate(tokenized_chat['input_ids'][0]):
            if token in [27, 91, 397, 100257]:
                removed_idx.append(i)
                removed +=1
            if token in [882, 78191]:
                raw_role_idx.append(i-removed)
    elif model_name == 'openchat-3.6-8b':
        i = 0
        while i<len(tokenized_chat['input_ids'][0]):
            token = tokenized_chat['input_ids'][0][i]
            if token >=128000:
                removed_idx.append(i)
                removed +=1
            if token == 128006:
                removed_idx.append(i+1)
                removed_idx.append(i+2)
                removed_idx.append(i+3)
                removed_idx.append(i+4)
                removed += 4
                raw_role_idx.append(i+5-removed)
                i+=5
            else:
                i+=1

    return removed_idx, raw_role_idx

def return_grouped_attn(model, tokenizer, prompt, reduction='first', model_name='Meta-Llama-3.1-8B-Instruct'):
    assert model_name in ['Meta-Llama-3.1-8B-Instruct', 'OLMo-2-1124-7B-Instruct', 'Qwen2.5-7B-Instruct', 'openchat-3.6-8b'], f"{model_name} not supported"
    raw_attn = return_raw_attn(model, tokenizer, prompt)

    tokenized_chat = generate_tokenized_chat(prompt, tokenizer)
    tokenized_chat = {k: v.to(model.device) for k, v in tokenized_chat.items()}
    removed_idx, raw_role_idx = return_special_token_and_role_index(tokenized_chat, model_name)
    attn_maps = ()
    for attn in raw_attn:
        attn_maps += (remove_rows_and_columns(attn, removed_idx), )

    role_idx = raw_role_idx[::2]
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    dim = attn_maps[0].shape[-1]
    sum_attn =torch.zeros([dim, dim])
    if reduction == 'sum':
        for i, attn in enumerate(attn_maps):
            if model_name == 'Qwen2.5-7B-Instruct' and i == len(attn_maps)-1:
                break
            else:
                sum_attn += (attn.sum(dim=[0,1]).cpu()/num_heads)
    elif reduction == 'first':
        sum_attn = (attn_maps[0].sum(dim=[0,1]).cpu()/num_heads)

    # normalize again so that it sums to one(because we removed some col/row):
    sum_attn = sum_attn / sum_attn.sum(dim=1, keepdim=True)

    # Suppose we have Q,A,Q,A,Q, we reduce this into QA, QA, QA, Q is a demo (grouped), and the last Q is the target
    attn_grouped = torch.zeros([len(role_idx), len(role_idx)])

    for i, query_idx in enumerate(role_idx):
        for j, key_idx in enumerate(role_idx):

            if i == len(role_idx)-1: # target prompt
                query_end = dim # set query end to the end of the prompt
            else:
                query_end = role_idx[i+1] # set query end to idx of token right before the next the "human" token

            if j == len(role_idx)-1:
                key_end = dim
            else:
                key_end = role_idx[j+1]
            sub_block = sum_attn[query_idx:query_end, key_idx:key_end]
            num_non_zero = (sub_block != 0).sum()
            sub_block_sum = sub_block.sum()

            # If there are any non-zero elements, compute the average and update
            if num_non_zero > 0:
                average_value = sub_block_sum / (query_end-query_idx)
                attn_grouped[i, j] = average_value
            else:
            # If no non-zero elements, keep the sub-block unchanged or set to zero
                attn_grouped[i, j] = 0

    return attn_grouped

def return_diag_attn(model, tokenizer, prompt, reduction='first'):
    raw_attn = return_raw_attn(model, tokenizer, prompt)

    tokenized_chat = generate_tokenized_chat(prompt, tokenizer)
    tokenized_chat = {k: v.to(model.device) for k, v in tokenized_chat.items()}
    removed_idx, raw_role_idx = return_special_token_and_role_index(tokenized_chat)

    attn_maps = ()
    for attn in raw_attn:
        attn_maps += (remove_rows_and_columns(attn, removed_idx), )
    role_idx = raw_role_idx[::2]
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    dim = attn_maps[0].shape[-1]

    sum_attn =torch.zeros([dim, dim])
    if reduction == 'sum':
        for attn in attn_maps:
            sum_attn += (attn.sum(dim=[0,1]).cpu()/num_heads)
    elif reduction == 'first':
        sum_attn = (attn_maps[0].sum(dim=[0,1]).cpu()/num_heads)

    role_idx.append(sum_attn.shape[0]+1)
    diag_attn = [torch.diag(sum_attn)[role_idx[i]:role_idx[i+1]].sum()/(role_idx[i+1]-role_idx[i]) for i in range(len(role_idx)-1)]

    return diag_attn

def visualize_attn_grid(sum_attn, role_idx):
    """
    Visualize the grouped attention map with bounding boxes overlaid and an optional axis range.
    Completely remove rectangles (and small triangles) above the diagonal.

    Args:
        attn_grouped (torch.Tensor): The grouped attention values.
        sum_attn (torch.Tensor): The original summed attention map.
        role_idx (list): The list of indices representing the role boundaries.
        x_range (tuple): Optional range for the x-axis (min, max).
        y_range (tuple): Optional range for the y-axis (min, max).
    """
    # Convert tensors to numpy for visualization
    sum_attn = sum_attn.numpy()

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Show the summed attention map
    im = ax.imshow(sum_attn, cmap='coolwarm', interpolation='nearest', norm=LogNorm())


    # Add bounding boxes for groups, only below the diagonal
    for i, query_idx in enumerate(role_idx):
        for j, key_idx in enumerate(role_idx):
            if i == len(role_idx) - 1:  # Last block in rows
                query_end = sum_attn.shape[0]
            else:
                query_end = role_idx[i + 1]

            if j == len(role_idx) - 1:  # Last block in columns
                key_end = sum_attn.shape[1]
            else:
                key_end = role_idx[j + 1]

            if j > i:  # Skip rectangles on or above the diagonal
                continue
            if j<i:
                rect = plt.Rectangle(
                        (key_idx, query_idx),  # Bottom-left corner
                        key_end - key_idx,  # Width
                        query_end - query_idx,  # Height
                        linewidth=3, edgecolor='red', facecolor='none',
                        alpha=0.7, ls='--'
                )
                ax.add_patch(rect)

            ax.text(
                    key_idx + (key_end - key_idx) / 2.9,
                    query_idx + (query_end - query_idx) / 1.3,
                    rf"$P_{{{i+1},{j+1}}}$",
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=18,
                    weight=1000
                )

    ax.plot([0, sum_attn.shape[1]], [0, sum_attn.shape[0]], color='blue', linewidth=5, linestyle='--', alpha=0.7)
    # Set axis ranges if specified
    ax.set_xlim([-1, sum_attn.shape[1]+1])
    ax.set_ylim([sum_attn.shape[0]+1, 0])

    r_role_idx = role_idx.copy()
    r_role_idx.append(sum_attn.shape[0]-1)
    for i, query_idx in enumerate(r_role_idx[:-1]):
        ax.plot([query_idx, query_idx],  # Triangle edges
                [query_idx, r_role_idx[i+1]],
                color='blue', linestyle='--', linewidth=5, alpha=0.7
                )
    for i, query_idx in enumerate(r_role_idx[1:]):
        ax.plot([r_role_idx[i], query_idx],  # Triangle edges
                [query_idx, query_idx],
                color='blue', linestyle='--', linewidth=5, alpha=0.7
                )

    # Set axis labels and title
    ax.set_xlabel("Key Tokens", fontsize=22)
    ax.set_ylabel("Query Tokens", fontsize=22)
    ax.set_xticks(role_idx, [r'$N_1$', r'$N_2$', r'$N_3$', r'$N_4$' , r'$N_5$'], fontsize=22)
    ax.set_yticks(role_idx, [r'$N_1$', r'$N_2$', r'$N_3$', r'$N_4$' , r'$N_5$'], fontsize=22)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # Show the plot
    plt.savefig("./figure/attn_grid.pdf", format="pdf", bbox_inches="tight")
