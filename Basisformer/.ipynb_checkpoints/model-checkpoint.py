import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from Basisformer.utils import Coefnet, MLP_bottle


# The Basisformer model integrates all the components to perform time series forecasting.
#  It normalizes the input, generates basis functions, processes them through the coefficient network,
#  and combines the basis functions to produce the forecast output.
#  It also includes mechanisms for loss calculation during training,
#  using both smoothness and entropy losses to optimize the model.


class Basisformer(nn.Module):
    def __init__(self,seq_len,pred_len,d_model,heads,basis_nums,block_nums,bottle,map_bottleneck,device,tau):
        super().__init__()
        self.d_model = d_model # the dimensionality of the model's hidden state
        self.k = heads # number of attention heads
        self.N = basis_nums # number of basis functions
        self.coefnet = Coefnet(blocks=block_nums,d_model=d_model,heads=heads)

        self.pred_len = pred_len # prediction length
        self.seq_len = seq_len # sequence length

        # Multi-Layer Perceptron
        self.MLP_x = MLP_bottle(seq_len,heads * int(seq_len/heads),int(seq_len/bottle)) #processes the input sequence length to create a more compact representation
        self.MLP_y = MLP_bottle(pred_len,heads * int(pred_len/heads),int(pred_len/bottle)) #same for prediction
        self.MLP_sx = MLP_bottle(heads * int(seq_len/heads),seq_len,int(seq_len/bottle)) # re-expands the sequence length helping to restore some structure
        self.MLP_sy = MLP_bottle(heads * int(pred_len/heads),pred_len,int(pred_len/bottle)) # same for prediction


        # linear layers with weight normalization for projecting sequences into the model dimension
        # nn.Linear(seq_len, d_model) - fully connected linear layer that transforms the input from seq_len or pred_len dimensions to d_model dimensions
        # wn function applies weight normalization to the linear layer
        self.project1 = wn(nn.Linear(seq_len,d_model))
        self.project2 = wn(nn.Linear(seq_len,d_model))
        self.project3 = wn(nn.Linear(pred_len,d_model))
        self.project4 = wn(nn.Linear(pred_len,d_model))
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss(reduction='none')

        self.device = device # setting the device (CPU or GPU)

        # smooth array
        arr = torch.zeros((seq_len+pred_len-2,seq_len+pred_len))
        for i in range(seq_len+pred_len-2):
            arr[i,i]=-1
            arr[i,i+1] = 2
            arr[i,i+2] = -1
        self.smooth_arr = arr.to(device)

        # initializing basis function
        # MLP maps input to a higher dim space
        self.map_MLP = MLP_bottle(4, # input dim
                                  self.N*(self.seq_len+self.pred_len), #output dim
                                  map_bottleneck, # hidden layer size for the MLP
                                  bias=True)
        self.tau = tau # temperature parameter
        self.epsilon = 1E-5 # to avoid deletion by zero

    def forward(self,x,mark,y=None,train=True,y_mark=None):
        # normalization
        mean_x = x.mean(dim=1,keepdim=True)
        std_x = x.std(dim=1,keepdim=True)
        feature = (x - mean_x) / (std_x + self.epsilon)
        # reshaping
        B,L,C = feature.shape               # batch size, seq length, number of features
        feature = feature.permute(0,2,1)    # changing order of dimentions
        feature = self.project1(feature)    # (B,C,d)
                                            # linear transforrmation defined as wn(nn.Linear(seq_len, d_model))

        # creating basis function
        m = self.map_MLP(           # maps the input marker to a higher-dimensional space
           mark[:, 0].unsqueeze(1)  # selects the first marker and reshapes it for the MLP;
                                    # unsqueeze adds new dim at position in ()
                        ).reshape(B,self.seq_len + self.pred_len,self.N) #reshapes the output to have other dimensions


        # normalization
        m = m / torch.sqrt(torch.sum(m**2,dim=1,keepdim=True)+self.epsilon)

        # using basis functions in the model by splitting and projecting basis functions
        raw_m1 = m[:,:self.seq_len].permute(0,2,1)  #(B,L,N) # corresponding to the input sequence
        raw_m2 = m[:,self.seq_len:].permute(0,2,1)   #(B,L',N) #corresponding to the prediction sequence
        # permute(0,2,1) changes the order of dimensions for compatibility with other operations

        m1 = self.project2(raw_m1)    #(B,N,d) projects the input sequence basis functions into the model dimension

        # attention mechanism with basis functions
        score,attn_x1,attn_x2 = self.coefnet(m1,feature)    #(B,k,C,N)
        # applies the coefficient network to the projected basis functions and the features extracted from the input sequence
        # scores represent how much each basis function contributes to the final representation


        # combining basis functions
        base = self.MLP_y(raw_m2).reshape(B,self.N,self.k,-1).permute(0,2,1,3)   #(B,k,N,L/k)
        out = torch.matmul(score,base).permute(0,2,1,3).reshape(B,C,-1)  #(B,C,k * (L/k))
        out = self.MLP_sy(out).reshape(B,C,-1).permute(0,2,1)   #（BC,L）



        # reverse normalization
        output = out * (std_x + self.epsilon) + mean_x

        #loss calculation
        if train:
            l_smooth = torch.einsum('xl,bln->xbn',self.smooth_arr,m)
            l_smooth = abs(l_smooth).mean()
            # l_smooth = self.criterion1(l_smooth,torch.zeros_like(l_smooth))

            # #back
            mean_y = y.mean(dim=1,keepdim=True)
            std_y = y.std(dim=1,keepdim=True)
            feature_y_raw = (y - mean_y) / (std_y + self.epsilon)

            feature_y = feature_y_raw.permute(0,2,1)
            feature_y = self.project3(feature_y)   #(BC,d)
            m2 = self.project4(raw_m2)    #(N,d)

            score_y,attn_y1,attn_y2 = self.coefnet(m2,feature_y)    #(B,k,C,N)
            logit_q = score.permute(0,2,3,1) #(B,C,N,k)
            logit_k = score_y.permute(0,2,3,1) #(B,C,N,k)

            # l_pos = torch.bmm(logit_q.view(-1,1,self.k), logit_k.view(-1,self.k,1)).reshape(-1,1)  #(B*C*N,1,1)
            l_neg = torch.bmm(logit_q.reshape(-1,self.N,self.k), logit_k.reshape(-1,self.N,self.k).permute(0,2,1)).reshape(-1,self.N) # (B,C*N,N)

            labels = torch.arange(0,self.N,1,dtype=torch.long).unsqueeze(0).repeat(B*C,1).reshape(-1)

            labels = labels.to(self.device)

            cross_entropy_loss = nn.CrossEntropyLoss()
            l_entropy = cross_entropy_loss(l_neg/self.tau, labels)

            return output,l_entropy,l_smooth,attn_x1,attn_x2,attn_y1,attn_y2
        else:
            # #back
            mean_y = y.mean(dim=1,keepdim=True)
            std_y = y.std(dim=1,keepdim=True)
            feature_y_raw = (y - mean_y) / (std_y + self.epsilon)

            feature_y = feature_y_raw.permute(0,2,1)
            feature_y = self.project3(feature_y)   #(BC,d)
            m2 = self.project4(raw_m2)    #(N,d)

            score_y,attn_y1,attn_y2 = self.coefnet(m2,feature_y)    #(B,k,C,N)
            return output,m,attn_x1,attn_x2,attn_y1,attn_y2

