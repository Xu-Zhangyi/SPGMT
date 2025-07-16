import torch
import numpy as np
from tqdm import tqdm


def test_spa_model(s_emb, label, device):
    batch_size = 100
    label_r = []
    for i in tqdm(range(0, s_emb.size(0), batch_size)):
        c_batch = s_emb[i:i + batch_size]
        diff = c_batch.unsqueeze(1) - s_emb.unsqueeze(0)
        distances_batch = torch.norm(diff, p=2, dim=2)
        label_r.append(distances_batch)
    label_r = torch.cat(label_r, dim=0)
    label_r = torch.argsort(label_r, dim=-1, descending=False).cpu().numpy()[:, 1:51]#升序
    # %================
    recall = torch.zeros((s_emb.shape[0], 6))
    label = label.numpy()
    for idx, la in tqdm(enumerate(label)):
        recall[idx, 0] += len(list(set(label_r[idx, :10]).intersection(set(la[:10]))))  # HR-10
        recall[idx, 1] += len(list(set(label_r[idx, :50]).intersection(set(la[:50]))))  # HR-50
        recall[idx, 2] += len(list(set(label_r[idx, :50]).intersection(set(la[:10]))))  # R10@50
        recall[idx, 3] += len(list(set(label_r[idx, :1]).intersection(set(la[:1]))))  # R1@1
        recall[idx, 4] += len(list(set(label_r[idx, :10]).intersection(set(la[:1]))))  # R1@10
        recall[idx, 5] += len(list(set(label_r[idx, :50]).intersection(set(la[:1]))))  # R1@50

    return recall
