import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1
    batch_size = x_size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixup_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixup_x, y_a, y_b, lam
