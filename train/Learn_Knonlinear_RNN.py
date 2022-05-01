import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import sys
import os
sys.path.append("../utility/")
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import odeint
from Utility import data_collecter
import time
        
#define network
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Network, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x,hidden=None):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).double()
        return hidden

def K_loss(data,net,u_dim=1,Nstate=4):
    steps,train_traj_num,Nstates = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_pred,hidden = net.forward(data[:steps-1,:,:])
    max_loss_list = []
    mean_loss_list = []
    for i in range(steps-1):
        X_current = X_pred[i,:,:]
        Y = data[i+1,:,u_dim:]
        Err = X_current-Y
        max_loss_list.append(torch.mean(torch.max(torch.abs(Err),axis=0).values).detach().cpu().numpy())
        mean_loss_list.append(torch.mean(torch.mean(torch.abs(Err),axis=0)).detach().cpu().numpy())
    return np.array(max_loss_list),np.array(mean_loss_list)


#loss function
def Klinear_loss(data,net,mse_loss,u_dim=1,gamma=0.99,Nstate=4,all_loss=0):
    steps,train_traj_num,Nstates = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_pred,hidden = net.forward(data[:steps-1,:,:])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(steps-1):
        X_current = X_pred[i,:,:]
        Y = data[i+1,:,u_dim:]
        beta_sum += beta
        loss += beta*mse_loss(X_current,Y)
        beta *= gamma
    loss = loss/beta_sum
    return loss


def train(env_name,train_steps = 200000,suffix="",augsuffix="",\
            layer_depth=3,obs_mode="theta",\
            activation_mode="ReLU",Ktrain_samples=50000):
    # Ktrain_samples = 1000
    # Ktest_samples = 1000    
    Ktrain_samples = Ktrain_samples
    Ktest_samples = 20000
    Ksteps = 15
    Kbatch_size = 100
    res = 1
    normal = 1
    gamma = 0.8
    #data prepare
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Ksteps)
    print("test data ok!")
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ksteps)
    print("train data ok!")
    in_dim = Ktest_data.shape[-1]-u_dim
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 128
    Elayers = [in_dim+u_dim]+[layer_width]*layer_depth+[in_dim]
    # Blayers = [in_dim]+[layer_width]*layer_depth+[in_dim*u_dim]
    print("layers:",Elayers)
    net = Network(input_size =in_dim+u_dim, output_size=in_dim, hidden_dim=128, n_layers=layer_depth-1)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)
    #train
    eval_step = 500
    best_loss = 1000.0
    best_state_dict = {}
    logdir = "../Data/"+suffix+"/KNonlinearRNN_"+env_name+augsuffix+"layer{}_AT{}_mode{}_samples{}".format(layer_depth,activation_mode,obs_mode,Ktrain_samples)
    if not os.path.exists( "../Data/"+suffix):
        os.makedirs( "../Data/"+suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    start_time = time.process_time()
    for i in range(train_steps):
        #K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:,Kindex[:Kbatch_size],:]
        Kloss = Klinear_loss(X,net,mse_loss,u_dim,gamma)
        optimizer.zero_grad()
        Kloss.backward()
        optimizer.step() 
        writer.add_scalar('Train/loss',Kloss,i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step ==0:
            #K loss
            with torch.no_grad():
                Kloss = Klinear_loss(Ktest_data,net,mse_loss,u_dim,gamma)
                # if auto_first and i<10000:
                #     loss = AEloss
                # else:
                writer.add_scalar('Eval/loss',Kloss,i)
                writer.add_scalar('Eval/best_loss',best_loss,i)
                if Kloss<best_loss:
                    best_loss = copy(Kloss)
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model':best_state_dict,'Elayer':Elayers}
                    torch.save(Saved_dict,logdir+".pth")
                    # print(logdir+".pth")
                print("Step:{} Eval K-loss:{} ".format(i,Kloss.detach().cpu().numpy()))
            # print("-------------END-------------")
        writer.add_scalar('Eval/best_loss',best_loss,i)
        # if (time.process_time()-start_time)>=210*3600:
        #     print("time out!:{}".format(time.clock()-start_time))
        #     break
    print("END-best_loss{}".format(best_loss))
    

def main():
    train(args.env,suffix=args.suffix,\
            layer_depth=args.layer_depth,obs_mode=args.obs_mode,\
            activation_mode=args.activation_mode,augsuffix=args.augsuffix,\
                Ktrain_samples=args.K_train_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="DampingPendulum")
    parser.add_argument("--suffix",type=str,default="4_30")
    parser.add_argument("--K_train_samples",type=int,default=50000)
    parser.add_argument("--augsuffix",type=str,default="")
    parser.add_argument("--obs_mode",type=str,default="theta")
    parser.add_argument("--activation_mode",type=str,default="ReLU")
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

