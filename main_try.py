from iTransformer.exp_basic import Exp_Basic
from iTransformer import Model
from iTransformer.utils.tools import EarlyStopping, adjust_learning_rate, visual
from iTransformer.utils.metrics import metric
from Basisformer.model import Basisformer
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import argparse

def log_and_print(message):
    logging.info(message)
    print(message)

def parse_args():
        parser = argparse.ArgumentParser(description='Time series prediction - Basisformer')
        parser.add_argument('--is_training', type=bool, default=True, help='train or test')
        parser.add_argument('--data_path', type=str, default='data', help='root path of the data file')
        parser.add_argument('--device', type=int, default=0, help='gpu device')
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length') 
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

 
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

        # iTransformer
        parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                            help='experiemnt name, options:[MTSF, partial_train]')
        parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
        parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
        parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
        parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
        parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
        parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                            'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

        args, _ = parser.parse_known_args()
        return args

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

def fit (curr_model):
    if curr_model == 'basis_former':
        def model_setup(args, device):
            model = Basisformer(args.seq_len, args.pred_len, args.d_model, args.heads, args.N, 
                        args.block_nums, args.bottleneck, args.map_bottleneck, device, args.tau)
            return model.to(device)

        def train(model, train_loader, args, device, record_dir):
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
            
                f_dim = 0
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
                    og_and_print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

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




    elif curr_model == 'itransformer':
        class Exp_Long_Term_Forecast(Exp_Basic):
            def __init__(self, args):
                super(Exp_Long_Term_Forecast, self).__init__(args)

            def _build_model(self):
                model = Model.Model(self.args).float()
                return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, train_loader, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                    
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test_loader,  setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, mape{}, rmse{}'.format(mse, mae, mape, rmse))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape{}, rmse{}'.format(mse, mae, mape, rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    
def parse_args_itrans():
        parser = argparse.ArgumentParser(description='iTransformer')

        # basic config
        parser.add_argument('--is_training', type=bool, default=True, help='status')
        parser.add_argument('--args.', type=str, default='test', help='model id')
        parser.add_argument('--model', type=str, default='iTransformer',
                            help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

        # data loader
        #parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='all_countries.csv', help='data csv file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        #parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

        # model define
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # iTransformer
        parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                            help='experiemnt name, options:[MTSF, partial_train]')
        parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
        parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
        parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
        parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
        parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
        parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                            'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

        args = parser.parse_args()
        return args
# Define the settings

        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len,
            args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
            args.factor, args.embed, args.distil, args.des, 0)
        
        if train_flag:
            Exp_Long_Term_Forecast.train(self=exp, train_loader=train_loader, setting=setting)
        
        if test_flag:
            Exp_Long_Term_Forecast.test(self=exp, test_loader=test_loader, setting=setting, test=0)
        return exp.model


def log_and_print(message):
    logging.info(message)
    print(message)


def parse_args():
        parser = argparse.ArgumentParser(description='Time series prediction - Basisformer')
        parser.add_argument('--is_training', type=bool, default=True, help='train or test')
        parser.add_argument('--data_path', type=str, default='data', help='root path of the data file')
        parser.add_argument('--device', type=int, default=0, help='gpu device')
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length') 
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

 
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

        # iTransformer
        parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                            help='experiemnt name, options:[MTSF, partial_train]')
        parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
        parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
        parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
        parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
        parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
        parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                            'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

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
            
            f_dim = 0
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