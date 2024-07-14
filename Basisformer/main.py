import argparse
import torch
import os
import logging
import numpy as np
import time
from adabelief_pytorch import AdaBelief
from Basisformer.model import Basisformer  # Ensure the correct import path
from Basisformer.evaluate_tool import metric
from Basisformer.pyplot import plot_seq_feature

def log_and_print(message):
    logging.info(message)
    print(message)

def create_model(args, device):
    model = Basisformer(args.seq_len, args.pred_len, args.d_model, args.heads, args.N, args.block_nums, args.bottleneck, args.map_bottleneck, device, args.tau)
    model.to(device)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Time series prediction - Basisformer')
    parser.add_argument('--is_training', type=bool, default=True, help='train or test')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--root_path', type=str, default='data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='all_countries.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
    parser.add_argument('--heads', type=int, default=16, help='head in attention')
    parser.add_argument('--d_model', type=int, default=100, help='dimension of model')
    parser.add_argument('--N', type=int, default=10, help='number of learnable basis')
    parser.add_argument('--block_nums', type=int, default=2, help='number of blocks')
    parser.add_argument('--bottleneck', type=int, default=2, help='reduction of bottleneck')
    parser.add_argument('--map_bottleneck', type=int, default=20, help='reduction of mapping bottleneck')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=24 , help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--tau', type=float, default=0.07, help='temperature of infonce loss')
    parser.add_argument('--loss_weight_prediction', type=float, default=1.0, help='weight of prediction loss')
    parser.add_argument('--loss_weight_infonce', type=float, default=1.0, help='weight of infonce loss')
    parser.add_argument('--loss_weight_smooth', type=float, default=1.0, help='weight of smooth loss')
    parser.add_argument('--check_point', type=str, default='checkpoint', help='check point path, relative path')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    
    # Use parse_known_args to ignore unrecognized arguments
    args, unknown = parser.parse_known_args()
    return args

def get_device(args):
    return torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, test_loader, args, device, record_dir):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_and_print('[Info] Number of parameters: {}'.format(num_params))

    para1 = [param for name, param in model.named_parameters() if 'map_MLP' in name]
    para2 = [param for name, param in model.named_parameters() if 'map_MLP' not in name]

    optimizer = AdaBelief(
        [{'params': para1, 'lr': 5e-3}, {'params': para2, 'lr': args.learning_rate}],
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=True
    )

    criterion = torch.nn.MSELoss()
    train_steps = len(train_loader)
    best_loss = float('inf')
    count = 0

    for epoch in range(args.train_epochs):
        train_loss = []
        loss_pred = []
        loss_of_ce = []
        l_s = []
        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            outputs, loss_infonce, loss_smooth, attn_x1, attn_x2, attn_y1, attn_y2 = model(batch_x, batch_x_mark, batch_y, True, batch_y_mark)

            loss_p = criterion(outputs, batch_y)
            lam1 = args.loss_weight_prediction
            lam2 = args.loss_weight_infonce
            lam3 = args.loss_weight_smooth
            loss = lam1 * loss_p + lam2 * loss_infonce + lam3 * loss_smooth
            train_loss.append(loss.item())
            loss_pred.append(loss_p.item())
            loss_of_ce.append(loss_infonce.item())
            l_s.append(loss_smooth.item())
            loss.backward()
            optimizer.step()

            if (i + 1) % (train_steps // 5) == 0:
                log_and_print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

        log_and_print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        log_and_print('loss_pred:{0}'.format(np.average(loss_pred)))
        log_and_print('loss entropy:{0}'.format(np.average(loss_of_ce)))
        log_and_print('loss smooth:{0}'.format(np.average(l_s)))

        fig = plot_seq_feature(outputs, batch_y, batch_x)
        ckpt_path = os.path.join(record_dir, args.check_point)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if best_loss == float('inf'):
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
        else:
            if train_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
                best_loss = train_loss
                count = 0
            else:
                count += 1
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'final_checkpoint.pth'))
        if count >= args.patience:
            break

    return

def test(model, test_loader, record_dir, args, device, scaler=None, test=True):
    if test:
        log_and_print('loading model')
        model.load_state_dict(torch.load(os.path.join(record_dir, args.check_point, 'valid_best_checkpoint.pth')))

    preds = []
    trues = []

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            outputs, m, attn_x1, attn_x2, attn_y1, attn_y2 = model(batch_x, batch_x_mark, batch_y, train=False, y_mark=batch_y_mark)

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if scaler:
                outputs_rescaled = scaler.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
                batch_y_rescaled = scaler.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)
                preds.append(outputs_rescaled)
                trues.append(batch_y_rescaled)
            else:
                preds.append(outputs)
                trues.append(batch_y)

    t2 = time.time()
    log_and_print('total_time:{0}'.format(t2 - t1))
    log_and_print('avg_time:{0}'.format((t2 - t1) / len(test_loader.dataset)))

    preds = [item for sublist in preds for item in sublist]
    trues = [item for sublist in trues for item in sublist]

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    np.savez(os.path.join(record_dir, 'test_results.npz'), preds=preds, trues=trues)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    log_and_print('mse:{}, mae:{}'.format(mse, mae))
    return mae, mse, rmse, mape, mspe


if __name__ == "__main__":
    args = parse_args()
    device = get_device(args)

    record_dir = os.path.join('records', args.data_path.split('.')[0], 'features_' + args.features,
                              'seq_len' + str(args.seq_len) + ',' + 'pred_len' + str(args.pred_len))
    if not os.path.exists(record_dir):
       os.makedirs(record_dir)

    if args.is_training:
        logger_file = os.path.join(record_dir, 'train.log')
    else:
        logger_file = os.path.join(record_dir, 'test.log')

    if os.path.exists(logger_file):
        with open(logger_file, "w") as file:
            file.truncate(0)
    logging.basicConfig(filename=logger_file, level=logging.INFO)

    log_and_print('Args in experiment:')
    log_and_print(args)

    model = create_model(args, device)
    log_and_print(model)