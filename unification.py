import importlib
import os

# device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

class unification:
    
    def fit_transformer(model, train_flag, test_flag, train_loader=None, test_loader=None, pretrained_model=None):
        '''Fits a transformer model to the train and/or test loaders
        
        model - "basis_former", "itransformer", "ns_autoformer"
        
        train_flag: typ(bool) - True: to train the model on train_loader, False: if pretrained_model is passed
        
        test_flag: typ(bool) - True: to test on test_loader, False: if only training
        
        pretrained_model - pass a pretrained model if available to be fitted on a test_loader. 
        eg. fit(basis_former, train_flag=False, test_flag=True, test_loader=test_loader, pretrained_model=model)
        '''
        
        if model == 'basis_former':             # Code for Basisforme

            import Basisformer.model
            importlib.reload(Basisformer.model)
            from Basisformer.model import Basisformer

            import Basisformer.main
            importlib.reload(Basisformer.main)
            from Basisformer.main import parse_args, model_setup
            importlib.reload(Basisformer.pyplot)

            args = parse_args()
            
            if pretrained_model == None:
                # Set up model
                model = model_setup(args, device)

            else:
                model = pretrained_model

            # Log arguments and model
            ##log_and_print('Args in experiment:')
            ##log_and_print(args)
            ##log_and_print(model)
            
            if train_flag:
                import Basisformer.model
                importlib.reload(Basisformer.model)
                from Basisformer.model import Basisformer

                import Basisformer.main
                importlib.reload(Basisformer.main)
                from Basisformer.main import train


                record_dir = os.path.join('records', args.data_path.split('.')[0], 'features_' + args.features,
                                        'seq_len' + str(args.seq_len) + ',' + 'pred_len' + str(args.pred_len))
                
                if train_loader == None:
                    return 'train_loader not found'

                # Call the train function
                train(model, train_loader, args, device, record_dir)
                
            else:
                if pretrained_model == None:
                    return 'model not found which is required for testing'
                
            if test_flag :
                import Basisformer.main
                importlib.reload(Basisformer.main)
                from Basisformer.main import test
                
                if test_loader == None:
                    return 'test_loader not found'

                test(model, test_loader, args, device, record_dir)
            return model
                
        
        elif model == 'itransformer':              # code for itransformer
            
            import iTransformer.experiment
            importlib.reload(iTransformer.experiment)
            from iTransformer.experiment import Exp_Long_Term_Forecast
            
            class Args:
                is_training = 1
                model_id = 'iTransformer_train'
                model = 'iTransformer'
                data = 'all_countries'
                features = 'M'
                target = 'OT'
                freq = 'h'
                checkpoints = './checkpoints/'
                seq_len = 96
                label_len = 48
                pred_len = 48
                enc_in = 24
                dec_in = 24
                c_out = 24
                d_model = 512
                n_heads = 8
                e_layers = 2
                d_layers = 1
                d_ff = 2048
                moving_avg = 25
                factor = 1
                distil = True
                dropout = 0.05
                embed = 'timeF'
                activation = 'gelu'
                output_attention = False
                do_predict = True
                num_workers = 10
                itr = 2
                train_epochs = 1
                batch_size = 24
                patience = 3
                learning_rate = 0.0001
                des = 'test'
                loss = 'mse'
                lradj = 'type1'
                use_amp = False
                use_gpu = True if torch.cuda.is_available() else False
                gpu = 0
                use_multi_gpu = False
                devices = '0,1,2,3'
                exp_name = 'MTSF'
                channel_independence = False
                inverse = False
                class_strategy = 'projection'
                target_root_path = './data'
                target_data_path = 'all_countries'
                efficient_training = False
                use_norm = True
                partial_start_index = 0
                seed = 2021
                p_hidden_dims = [128, 128]
                p_hidden_layers = 2

            args = Args()
            
            if pretrained_model == None:
                # Initialize the experiment
                exp = Exp_Long_Term_Forecast(args)

            else:
                return 'pretrained not valid for iTransformer and ns_autoformer'

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
        
        elif model == 'ns_autoformer':             # code for ns_transformer

            import ns_Autoformer.ns_Autoformer
            importlib.reload(ns_Autoformer.ns_Autoformer)
            from ns_Autoformer.main import Exp_Main

            class Args:
                is_training = 1
                model_id = 'ns_autoformer_train'
                model = 'ns_Autoformer'
                features = 'M'
                target = 'OT'
                freq = 'h'
                checkpoints = './checkpoints/'
                seq_len = 96
                label_len = 48
                pred_len = 48
                enc_in = 24
                dec_in = 24
                c_out = 24
                d_model = 512
                n_heads = 8
                e_layers = 2
                d_layers = 1
                d_ff = 2048
                moving_avg = 25
                factor = 1
                distil = True
                dropout = 0.05
                embed = 'timeF'
                activation = 'gelu'
                output_attention = False
                do_predict = True
                num_workers = 10
                itr = 2
                train_epochs = 1
                batch_size = 24
                patience = 3
                learning_rate = 0.0001
                des = 'test'
                loss = 'mse'
                lradj = 'type1'
                use_amp = False
                use_gpu = True if torch.cuda.is_available() else False
                gpu = 0
                use_multi_gpu = False
                devices = '0,1,2,3'
                seed = 2021
                p_hidden_dims = [128, 128]
                p_hidden_layers = 2

            args = Args()
            
            if pretrained_model == None:
                # Initialize the experiment
                exp = Exp_Main(args)

            # Define the setting string
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_id, args.model, args.features, args.seq_len, args.label_len,
                args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.factor, args.embed, args.distil, args.des, 0)
            
            if train_flag:
                Exp_Main.train(self=exp, train_loader=train_loader, setting=setting)
            
            if test_flag:
                Exp_Main.test(self=exp, test_loader=test_loader, setting=setting, test=0)
            return exp.model