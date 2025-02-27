import argparse
import torch
from torch import nn
import os
import logging
import numpy as np
import time
import matplotlib.pyplot as plt
from adabelief_pytorch import AdaBelief
from Basisformer.model import Basisformer  # Ensure the correct import path
from Basisformer.evaluate_tool import metric
from Basisformer.pyplot import plot_seq_feature

def log_and_print(message):
    logging.info(message)
    print(message)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_args():
        parser = argparse.ArgumentParser(description='Time series prediction - Basisformer')
        parser.add_argument('--is_training', type=bool, default=True, help='train or test')
        parser.add_argument('--data_path', type=str, default='data', help='root path of the data file')
        parser.add_argument('--device', type=int, default=0, help='gpu device')
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--features', type=str, default='M', help='forecasting task')
        parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
        parser.add_argument('--heads', type=int, default=16, help='head in attention')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--N', type=int, default=10, help='number of learnable basis')
        parser.add_argument('--block_nums', type=int, default=2, help='number of blocks')
        parser.add_argument('--bottleneck', type=int, default=2, help='reduction of bottleneck')
        parser.add_argument('--map_bottleneck', type=int, default=20, help='reduction of mapping bottleneck')
        parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=24 , help='batch size of train input data')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--tau', type=float, default=0.07, help='temperature of infonce loss')
        parser.add_argument('--loss_weight_prediction', type=float, default=1.0, help='weight of prediction loss')
        parser.add_argument('--loss_weight_infonce', type=float, default=1.0, help='weight of infonce loss')
        parser.add_argument('--loss_weight_smooth', type=float, default=1.0, help='weight of smooth loss')
        parser.add_argument('--check_point', type=str, default='checkpoint', help='check point path, relative path')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        args, _ = parser.parse_known_args()
        return args

def model_setup(args, device):
    model = Basisformer(args.seq_len, args.pred_len, args.d_model, args.heads, args.N, 
                        args.block_nums, args.bottleneck, args.map_bottleneck, device, args.tau)
    return model.to(device)


def train(model, train_loader, args, device, record_dir):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ##log_and_print('[Info] Number of parameters: {}'.format(num_params))
    
    para1 = [param for name, param in model.named_parameters() if 'map_MLP' in name]
    para2 = [param for name, param in model.named_parameters() if 'map_MLP' not in name]
    optimizer = AdaBelief([{'params':para1,'lr':5e-3},{'params':para2,'lr':args.learning_rate}], eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
    criterion = nn.MSELoss()

    train_steps = len(train_loader)

    ##writer = SummaryWriter(os.path.join(record_dir,'event'))

    best_loss = float('inf')
    count = 0

    for epoch in range(args.train_epochs):
        train_loss = []
        loss_pred = []
        loss_of_ce = []
        l_s = []
        model.train()
        epoch_time = time.time()
#################### Specific to Basisformer (index) ##########################################
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            outputs, loss_infonce, loss_smooth, *_ = model(batch_x, index.float().to(device), batch_y, y_mark=batch_y_mark)
            
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

            if (i+1) % (train_steps//5) == 0:
                log_and_print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

        log_and_print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        loss1 = np.average(loss_pred)
        log_and_print('loss_pred:{0}'.format(loss1))
        loss2 = np.average(loss_of_ce)
        log_and_print('loss entropy:{0}'.format(loss2))
        loss3 = np.average(l_s)
        log_and_print('loss smooth:{0}'.format(loss3))

        log_and_print("Epoch: {0} | Train Loss: {1:.7f}".format(epoch + 1, train_loss))

        fig = plot_seq_feature(outputs, batch_y, batch_x)
        plots_dir = os.path.join(record_dir, 'train_plots')
        ensure_dir(plots_dir)
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f'train_plot_epoch_{epoch}.png'))
        plt.close(fig)
        #writer.add_figure("figure_train", fig, global_step=epoch)
        #writer.add_scalar('train_loss', train_loss, global_step=epoch)
        
        ckpt_path = os.path.join(record_dir, args.check_point)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
                
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'best_checkpoint.pth'))
            count = 0
        else:
            count += 1

        torch.save(model.state_dict(), os.path.join(ckpt_path, 'final_checkpoint.pth'))
        
        if count >= args.patience:
            break
    return

def test(model, test_loader, args, device, record_dir):
    if test:
        log_and_print('loading model')
        model.load_state_dict(torch.load(os.path.join(record_dir, args.check_point, 'best_checkpoint.pth')))
    
    preds = []
    trues = []

    model.eval()
    t1 = time.time()

    test_plots_dir = os.path.join(record_dir, 'test_plots')
    ensure_dir(test_plots_dir)

    with torch.no_grad():
############################## Specific to Basisformer (index) ###################################
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            outputs, _, attn_x1, attn_x2, attn_y1, attn_y2 = model(batch_x, index.float().to(device), batch_y, train=False, y_mark=batch_y_mark)
                
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            fig = plot_seq_feature(outputs, batch_y, batch_x, label=f"test_batch_{i}")
            plt.savefig(os.path.join(test_plots_dir, f'test_plot_batch_{i}.png'))
            plt.close(fig)
            
    
            preds.append(outputs)
            trues.append(batch_y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
            
    t2 = time.time()
    log_and_print('total_time:{0}'.format(t2-t1))
    log_and_print('avg_time:{0}'.format((t2-t1)/len(test_loader)))

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    log_and_print('mse:{}, mae:{}, mape:{}, rmse:{}'.format(mse, mae, mape, rmse))

    return 