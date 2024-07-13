import argparse
import torch
import sys
sys.path.append('.')
from torch import optim
from model import Basisformer
from torch import nn
import time
import numpy as np
from evaluate_tool import metric
from torch.utils.tensorboard import SummaryWriter
from pyplot import plot_seq_feature
from adabelief_pytorch import AdaBelief
import logging
import random
import matplotlib.pyplot as plt
import os


def train():
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_and_print('[Info] Number of parameters: {}'.format(num_params))

    # data sets and their corresponding loaders
    train_set, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args,flag='val')
    test_data, test_loader = data_provider(args,flag='test')
    

    para1 = [param for name,param in model.named_parameters() if 'map_MLP' in name]
    para2 = [param for name,param in model.named_parameters() if 'map_MLP' not in name]

    # optimizer updates the model parameters during training
    # optimizer = AdaBelief(model.parameters(), lr=args.learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
    optimizer = AdaBelief([{'params':para1,'lr':5e-3},{'params':para2,'lr':args.learning_rate}], eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
    # optimizer = AdaBelief(model.parameters(), lr=args.learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
    

    criterion = nn.MSELoss()
    criterion_view = nn.MSELoss(reduction='none')

    # number of batches in the training set?
    train_steps = len(train_loader)
    # initializing the Tensor Board writer for logging training process
    writer = SummaryWriter(os.path.join(record_dir,'event'))

    # defining for early stopping
    best_loss = 0
    count_error = 0
    count = 0
    

    #training loop

    for epoch in range(args.train_epochs):
        #lists to store
        train_loss = []
        loss_pred = []
        loss_of_ce = []
        l_s = []
        #setting model to training mode
        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(train_loader):
            #clears the gradients of all optimized tensors
            optimizer.zero_grad()

            # loading data to the specified device (originally to cuda)
            batch_x = batch_x.float().to(device) # (B,L,C)
            batch_y = batch_y.float().to(device) # (B,L,C)
            batch_y_mark = batch_y_mark.float().to(device)
            
            #feature dimension
            f_dim = -1 if args.features == 'MS' else 0
            #matching the target sequence length required by the model
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            #forward pass through the model to get outputs and losses
            outputs,loss_infonce,loss_smooth,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index.float().to(device),batch_y,y_mark=batch_y_mark)
            
            # calculating loss
            loss_p = criterion(outputs, batch_y)
            lam1 = args.loss_weight_prediction
            lam2 = args.loss_weight_infonce
            lam3 = args.loss_weight_smooth
        
            # if loss_p > 5:
            #     count_error = count_error +1
            #     writer.add_scalar('error_loss', loss_p, global_step=count_error)
            #     fig = plot_seq_feature(outputs, batch_y,batch_x,error=True,input=batch_x)
            #     writer.add_figure("figure_error", fig, global_step=count_error)
            #     log_and_print(loss_p)

            # total loss  
            loss = lam1 * loss_p + lam2 * loss_infonce  + lam3 * loss_smooth
            train_loss.append(loss.item())
            loss_pred.append(loss_p.item())
            loss_of_ce.append(loss_infonce.item())
            l_s.append(loss_smooth.item())

            # greadient of the loss
            loss.backward()

            #updating model parameters
            optimizer.step()

            #logging every fifth step of the training process 
            if (i+1) % (train_steps//5) == 0:
                log_and_print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

        # every epoch logging
        log_and_print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        # losses of every epoch
        train_loss = np.average(train_loss)
        loss1 = np.average(loss_pred)
        log_and_print('loss_pred:{0}'.format(loss1))
        loss2 = np.average(loss_of_ce)
        log_and_print('loss entropy:{0}'.format(loss2))
        loss3 = np.average(l_s)
        log_and_print('loss smooth:{0}'.format(loss3))
        vali_loss = vali(vali_data, vali_loader, criterion_view, epoch, writer, 'vali')
        test_loss = vali(test_data, test_loader, criterion_view, epoch, writer, 'test')
        # logging
        log_and_print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
            epoch + 1, train_loss, vali_loss, test_loss))

        # figures to TensorBoard
        fig = plot_seq_feature(outputs, batch_y, batch_x)
        writer.add_figure("figure_train", fig, global_step=epoch)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('vali_loss', vali_loss, global_step=epoch)
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        
        #saving model chaeckpoints
        ckpt_path = os.path.join(record_dir,args.check_point)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        #saving in new folder if it is firs tepoch
        if best_loss == 0:
            best_loss = vali_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
        else:
            if vali_loss < best_loss: #updates the results if vali loss improves
                torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
                best_loss = vali_loss
                count = 0
            else:
                count = count + 1
        #final save at the end of each epoch
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'final_checkpoint.pth'))
        #stopping training if loss doesn't improve for a number of epochs
        if count >= args.patience:
            break
    return


def test(setting='setting',test=True):
    test_data, test_loader = data_provider(args,flag='test')
    if test:
        log_and_print('loading model')
        model.load_state_dict(torch.load(os.path.join(record_dir,args.check_point, 'valid_best_checkpoint.pth')))
    
    preds = []
    trues = []

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(test_loader):
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            outputs,m,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index.float().to(device),batch_y,train=False,y_mark=batch_y_mark)
                
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  
            true = batch_y  

            preds.append(pred)
            trues.append(true)
            
    t2 = time.time()
    log_and_print('total_time:{0}'.format(t2-t1))
    log_and_print('avg_time:{0}'.format((t2-t1)/len(test_data)))

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


    mae, mse, rmse, mape, mspe = metric(preds, trues)
    log_and_print('mse:{}, mae:{}'.format(mse, mae))
    log_and_print(f'Test - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, MSPE: {mspe}')

    return 

def test(model, test_loader, record_dir, args, device, scaler, country_names, test=True):
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

            outputs_rescaled = scaler.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
            batch_y_rescaled = scaler.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)

            preds.append(outputs_rescaled)
            trues.append(batch_y_rescaled)

    t2 = time.time()
    log_and_print('total_time:{0}'.format(t2 - t1))
    log_and_print('avg_time:{0}'.format((t2 - t1) / len(test_loader.dataset)))

    # Flatten lists
    preds = [item for sublist in preds for item in sublist]
    trues = [item for sublist in trues for item in sublist]

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    # Save the results to a file
    np.savez(os.path.join(record_dir, 'test_results.npz'), preds=preds, trues=trues, country_names=country_names)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    log_and_print('mse:{}, mae:{}'.format(mse, mae))
    return mae, mse, rmse, mape, mspe

def log_and_print(text):
    logging.info(text)
    print(text)
    return    

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
            torch.set_num_threads(max_threads)  # intraop
            if torch.get_num_interop_threads() != max_threads:
                torch.set_num_interop_threads(max_threads)  # interop
    try:
        # Import mkl and set the number of threads if available
        import mkl
        mkl.set_num_threads(max_threads)
    except ImportError:
        # If mkl is not available, proceed without it
        pass
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]
    
#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time series prediction - Basisformer')
    parser.add_argument('--is_training', type=bool, default=True, help='train or test')
    parser.add_argument('--device', type=int, default=0, help='gpu dvice')

    # data loader
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='all_countries.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                            'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Price (EUR/MWhe)', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=96, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str default='tanh'

    # model define
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--heads', type=int, default=16, help='head in attention')
    parser.add_argument('--d_model', type=int, default=100, help='dimension of model')
    parser.add_argument('--N', type=int, default=10, help='number of learnable basis')
    parser.add_argument('--block_nums', type=int, default=2, help='number of blocks')
    parser.add_argument('--bottleneck', type=int, default=2, help='reduction of bottleneck')
    parser.add_argument('--map_bottleneck', type=int, default=20, help='reduction of mapping bottleneck')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--tau', type=float, default=0.07, help='temperature of infonce loss')
    parser.add_argument('--loss_weight_prediction', type=float, default=1.0, help='weight of prediction loss')
    parser.add_argument('--loss_weight_infonce', type=float, default=1.0, help='weight of infonce loss')
    parser.add_argument('--loss_weight_smooth', type=float, default=1.0, help='weight of smooth loss')


    #checkpoint_path
    parser.add_argument('--check_point',type=str,default='checkpoint',help='check point path, relative path')

    args = parser.parse_args()
    
    record_dir = os.path.join('records',args.data_path.split('.')[0],'features_'+args.features,\
                              'seq_len'+str(args.seq_len)+','+'pred_len'+str(args.pred_len))
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    
    if args.is_training:
        logger_file = os.path.join(record_dir,'train.log')
    else:
        logger_file = os.path.join(record_dir,'test.log')
        
    if os.path.exists(logger_file):
        with open(logger_file, "w") as file:
            file.truncate(0)
    logging.basicConfig(filename=logger_file, level=logging.INFO)
    
    log_and_print('Args in experiment:')
    log_and_print(args)

    device = init_dl_program(args.device, seed=0,max_threads=8) if torch.cuda.is_available() else "cpu"
    # device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"
    model = Basisformer(args.seq_len,args.pred_len,args.d_model,args.heads,args.N,args.block_nums,args.bottleneck,args.map_bottleneck,device,args.tau)

    log_and_print(model)
    model.to(device)  ##
    if args.is_training:
        train()
    else:
        test()

def plot_predictions(true, pred, country_names, num_samples=24, samples_per_figure=6):
    num_figures = num_samples // samples_per_figure
    if num_samples % samples_per_figure:
        num_figures += 1
    
    for fig_num in range(num_figures):
        plt.figure(figsize=(20, 12))
        start_idx = fig_num * samples_per_figure
        end_idx = min((fig_num + 1) * samples_per_figure, num_samples)
        
        for i in range(start_idx, end_idx):
            plt.subplot(samples_per_figure // 2, 2, (i % samples_per_figure) + 1)
            plt.plot(true[i], label='True')
            plt.plot(pred[i], label='Pred')
            plt.legend()
            plt.title(f'{country_names[i]}')
        
        plt.tight_layout()
        plt.show()

# Load the results from the file
def load_and_plot_results(record_dir, num_samples=24, samples_per_figure=6):
    data = np.load(os.path.join(record_dir, 'test_results.npz'))
    preds = data['preds']
    trues = data['trues']
    country_names = data['country_names']
    plot_predictions(trues, preds, country_names, num_samples, samples_per_figure)