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
    def __init__(self,layers,u_dim,activation_mode="ReLU"):
        super(Network,self).__init__()
        ELayers = OrderedDict()
        for layer_i in range(len(layers)-1):
            ELayers["linear_{}".format(layer_i)] = nn.Linear(layers[layer_i],layers[layer_i+1])
            if layer_i != len(layers)-2:
                if activation_mode.startswith("tanh"):
                    ELayers["relu_{}".format(layer_i)] = nn.Tanh()
                else:
                    ELayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.Enet = nn.Sequential(ELayers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BLayers = OrderedDict()
        # for layer_i in range(len(B_layers)-1):
        #     BLayers["linear_{}".format(layer_i)] = nn.Linear(B_layers[layer_i],B_layers[layer_i+1])
        #     if layer_i != len(B_layers)-2:
        #         if activation_mode.startswith("tanh"):
        #             ELayers["relu_{}".format(layer_i)] = nn.Tanh()
        #         else:
        #             ELayers["relu_{}".format(layer_i)] = nn.ReLU()
        # self.Bnet = nn.Sequential(BLayers)
        # self.Nstates = layers[-1]
        self.u_dim = u_dim

    def forward(self,x):
        return self.Enet(x)
    # def forward(self,x,u):
    #     Bmat = self.Bnet(x).view(-1,self.Nstates,self.u_dim)
    #     U_out = torch.bmm(Bmat,u.view(-1,self.u_dim,1)).view(-1,self.Nstates)
    #     return self.Enet(x)+U_out+x
    

def K_loss(data,net,u_dim=1,Nstate=4):
    steps,train_traj_num,Nstates = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = data[0,:,u_dim:]
    max_loss_list = []
    mean_loss_list = []
    for i in range(steps-1):
        X_current = net.forward(torch.cat((X_current,data[i,:,:u_dim]),-1))
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
    X_current = data[0,:,u_dim:]
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(steps-1):
        X_current = net.forward(torch.cat((X_current,data[i,:,:u_dim]),-1))
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
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Ksteps,mode="eval")
    Ktest_samples = Ktest_data.shape[1]
    print("test data ok!,shape:",Ktest_data.shape)
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ksteps,mode="train")
    print("train data ok!,shape:",Ktrain_data.shape)
    Ktrain_samples = Ktrain_data.shape[1]
    in_dim = Ktest_data.shape[-1]-u_dim
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 128
    Elayers = [in_dim+u_dim]+[layer_width]*layer_depth+[in_dim]
    # Blayers = [in_dim]+[layer_width]*layer_depth+[in_dim*u_dim]
    print("layers:",Elayers)
    net = Network(Elayers,u_dim,activation_mode)
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
    logdir = "../Data/"+suffix+"/KNonlinear_"+env_name+augsuffix+"layer{}_AT{}_mode{}_samples{}".format(layer_depth,activation_mode,obs_mode,Ktrain_samples)
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
                writer.add_scalar('Eval/loss',Kloss,i)
                writer.add_scalar('Eval/best_loss',best_loss,i)
                if Kloss<best_loss:
                    best_loss = copy(Kloss)
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model':best_state_dict,'Elayer':Elayers}
                    torch.save(Saved_dict,logdir+".pth")
                print("Step:{} Eval K-loss:{} ".format(i,Kloss.detach().cpu().numpy()))
            # print("-------------END-------------")
        writer.add_scalar('Eval/best_loss',best_loss,i)
    print("END-best_loss{}".format(best_loss))
    

def main():
    train(args.env,suffix=args.suffix,\
            layer_depth=args.layer_depth,obs_mode=args.obs_mode,\
            activation_mode=args.activation_mode,augsuffix=args.augsuffix,
            Ktrain_samples=args.K_train_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="DampingPendulum")
    parser.add_argument("--suffix",type=str,default="4_28")
    parser.add_argument("--K_train_samples",type=int,default=50000)
    parser.add_argument("--augsuffix",type=str,default="")
    parser.add_argument("--obs_mode",type=str,default="theta")
    parser.add_argument("--activation_mode",type=str,default="ReLU")
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

