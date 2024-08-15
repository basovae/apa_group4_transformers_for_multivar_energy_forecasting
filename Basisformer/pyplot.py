import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def plot_seq_feature(pred_, true_, history_, label="train", error=False, input='', wv=''):
    assert(pred_.shape == true_.shape)
    
    # Convert to numpy if they are PyTorch tensors
    if isinstance(pred_, torch.Tensor):
        pred_ = pred_.detach().cpu().numpy()
    if isinstance(true_, torch.Tensor):
        true_ = true_.detach().cpu().numpy()
    if isinstance(history_, torch.Tensor):
        history_ = history_.detach().cpu().numpy()
    
    index = -1
    if pred_.shape[2] > 800:
        index = 840
    pred = pred_[..., index].reshape(pred_.shape[0], pred_.shape[1], 1)
    true = true_[..., index].reshape(true_.shape[0], true_.shape[1], 1)
    history = history_[..., index].reshape(history_.shape[0], history_.shape[1], 1)
    
    if len(pred.shape) == 3:  #BLD
        if not error:
            pred = pred[0]
            true = true[0]
            history = history[0]
        else:
            largest_loss = 0
            largest_index = 0
            criterion = nn.MSELoss()
            for i in range(pred.shape[0]):
                loss = criterion(torch.tensor(pred[i]), torch.tensor(true[i]))
                if loss > largest_loss:
                    largest_loss = loss
                    largest_index = i
            pred = pred[largest_index]
            true = true[largest_index]
            history = history[largest_index]
            if isinstance(input, (torch.Tensor, np.ndarray)):
                input_error = input[largest_index]
    
    L, D = pred.shape
    L_h, D_h = history.shape
    pic_row, pic_col = D, 1
    fig = plt.figure(figsize=(8*pic_row, 8*pic_col))
    for i in range(1):
        ax = plt.subplot(pic_row, pic_col, i+1)
        ax.plot(np.arange(L_h), history[:, i], label="history")
        ax.plot(np.arange(L_h, L_h+L), pred[:, i], label="pred")
        ax.plot(np.arange(L_h, L_h+L), true[:, i], label="true")
        ax.set_title(f"dimension = {i},  {label}")
        ax.legend()
    return fig