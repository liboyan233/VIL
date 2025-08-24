import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attn_mask(attn_mask, layer_idx=0):
    """
    attn_mask: Tensor of shape (num_heads, seq_len, seq_len) or (seq_len, seq_len)
    default setting the attn_mask already averaged over heads
    """
    print('attn_mask shape:', attn_mask.shape)
    if attn_mask.dim() == 3:
        # first batch 
        mask = attn_mask[0].detach().cpu().numpy()
    elif attn_mask.dim() == 4:
        # first batch sample
        mask = attn_mask[0, 0].detach().cpu().numpy()
    else:
        mask = attn_mask.detach().cpu().numpy()
    mask = mask[:, :600]  # only keep attn rela to img_embeddings
    # change the permutation to make feature from same image close to each other

    mask = np.average(mask, axis=0, keepdims=True)  # average over predictions!
    mask_new = np.zeros_like(mask)
    for i in range(0, mask.shape[1], 40):
        mask_new[:, i//2:i//2+20] = mask[:, i:i+20]
    for i in range(20, mask.shape[1], 40):
        mask_new[:, 290+ i//2:290+i//2+20] = mask[:, i:i+20]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(mask_new, cmap='viridis', cbar=True)
    plt.savefig(f'attention_mask_layer_{layer_idx}.png')
    plt.close()

