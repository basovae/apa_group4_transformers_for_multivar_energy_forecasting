import numpy as np

def metric(pred, true, epsilon=1e-5):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true + epsilon)) * 100
    mspe = np.mean(((pred - true) / true) ** 2) * 100
    return mae, mse, rmse, mape, mspe
