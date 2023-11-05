import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad, Variable

import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_preprocessing(data_name, data_dir, device):
    df = pd.read_csv(data_dir+data_name+".csv")
    
    time_column = df.columns[0]
    data_column = df.columns[1:]
    
    time_trace = df[data_column].to_numpy().T

    scaled_trace = time_trace.copy()
    ss = StandardScaler()
    scaled_trace = ss.fit_transform(scaled_trace)

    scaled_trace = torch.FloatTensor(scaled_trace).unsqueeze(1).to(device)
    mean_scaled_trace = scaled_trace.mean(axis=0).unsqueeze(1)

    time_trace = torch.FloatTensor(time_trace).unsqueeze(1).to(device)

    obs_time = torch.FloatTensor(df[time_column].values).unsqueeze(1).to(device)
    
    return obs_time, time_trace, scaled_trace, mean_scaled_trace 

def kl_divergence(mu, log_var):
    kl = torch.mean(- 0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp(), axis=2))
    return kl

def cal_der(y, x, device) :
    return grad(y, x, create_graph=True, grad_outputs=torch.ones(y.size()).to(device))[0]

def train(model, optimizer, loss_f, loss_kl, col_time, obs_time, time_trace, beta, device, s_min=0.5, s_max=2) :
    obs = obs_time.repeat(model.data.size()[0],1,1)
    col = col_time.repeat(model.data.size()[0],1,1)   
    t_v = Variable(col, requires_grad = True)

    mu, log_var = model.encoder()

    model.z = model.reparameterization(mu, log_var)
    
    s, tilde_g, tilde_y = model(t_v)

    if s_max == None:
        s_penalty = (torch.abs(s-((beta)/s_min**2))).mean()
    else:
        s_penalty = (torch.abs(s-((1-beta)/s_max**2+beta/s_min**2))).mean()
    
    lb, ld = model.pred_lmda()
    pred_lb = lb.repeat(1,1,len(col_time))
    pred_ld = ld.repeat(1,1,len(col_time))

    y_t = cal_der(tilde_y, t_v, device)[:,:,0].view(tilde_y.size())

    _, _, tilde_y_obs = model(obs)

    optimizer.zero_grad()
    physics_loss = loss_f(y_t, - pred_ld*tilde_y + pred_lb*tilde_g)

    data_loss = loss_f(tilde_y_obs, time_trace)
    
    reg_loss = loss_kl(mu, log_var)+s_penalty
    
    loss = physics_loss+data_loss+reg_loss
    
    loss.backward(retain_graph=True)
    optimizer.step()
        
    return physics_loss.item(), data_loss.item(), reg_loss.item()

class MDN(nn.Module):
    def __init__(self, d, final_time, device, K = 4, h = 16, M=80, s_min = 0.5, s_max = 2):
        super(MDN, self).__init__()
        
        self.d = d
        self.final_time = final_time
        self.d = d
        self.M = M
        self.latent_dim = K
        self.hidden_dim = h
        self.s_min = s_min
        self.s_max = s_max
        
        self.enc_hidden1 = nn.Linear(self.d, self.hidden_dim)
        self.enc_hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.latent_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.latent_logvar = nn.Linear(self.hidden_dim, self.latent_dim)       

        self.decoder_hidden3_1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_lb = nn.Linear(self.hidden_dim, 1)
        self.decoder_hidden3_2 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_ld = nn.Linear(self.hidden_dim, 1)
                
        self.decoder_hidden4 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_hidden5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_w = nn.Linear(self.hidden_dim, self.M)
        
        self.decoder_hidden6 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_hidden7 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_s = nn.Linear(self.hidden_dim, self.M)
        
        self.nn_hidden8 = nn.Linear(self.M, self.M)
        self.nn_wprime = nn.Linear(self.M, self.M)
        
        self.ELU = nn.ELU()
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(dim=2)

        self.device = device
        
    def encoder(self):
        enc = self.ELU(self.enc_hidden1(self.data))
        enc = self.ELU(self.enc_hidden2(enc))
        
        mean = self.latent_mu(enc)
        logvar = self.latent_logvar(enc)
        
        return mean, logvar
        
    def reparameterization(self, mean, logvar):
        var = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(var)                 # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def pred_lmda(self):
        lb = self.ELU(self.decoder_hidden3_1(self.z))
        lb = self.ELU(self.decoder_lb(lb))+1
        ld = self.ELU(self.decoder_hidden3_2(self.z))
        ld = self.ELU(self.decoder_ld(ld))+1
        return lb, ld                                 # return lambda_b, lambda_d
    
    def reconstruct(self):
        w = self.ELU(self.decoder_hidden4(self.z))
        w = self.ELU(self.decoder_hidden5(w))
        #w = self.Softmax(self.decoder_w(w))
        w = self.ELU(self.decoder_w(w))+1           # You can also use self.Softmax instead of ELU+1
        w = w/(w.sum(axis=2).unsqueeze(1))          # You can also use self.Softmax instead of ELU+1
        
        s = self.ELU(self.decoder_hidden6(self.z))
        s = self.ELU(self.decoder_hidden7(s))
        s = self.decoder_s(s)
        s = (1/self.s_min**2-1/self.s_max**2)*self.Sigmoid(s)+1/self.s_max**2

        wprime = self.ELU(self.nn_hidden8(w))
        wprime = self.ELU(self.nn_wprime(wprime))+1      
        return s, w, wprime
    
    def Rayleigh(self, t, w, mu, s):                            # mixture of CDF Rayleigh distributions
        MDN = -1*torch.max(torch.zeros_like(mu),1.1*s*(t-mu)-(1-torch.exp(-s*((t-mu)**2)/2)))
        MDN += torch.max(torch.zeros_like(t), 1.1*s*(t-mu))
        return (w*MDN).sum(axis=2).unsqueeze(1)
        
    def forward(self, t):
        s, w, wprime = self.reconstruct()               # scale factor is square-inversed for a computational convenience.
        s = s.repeat(1,t.size()[1],1)
        w = w.repeat(1,t.size()[1],1)
        wprime = wprime.repeat(1,t.size()[1],1)
        t = t.repeat(1,1,self.M)
        
        mu_g = torch.linspace(0,int(self.final_time),self.M).repeat(w.size()[0],w.size()[1],1).to(self.device)
        mu_y = torch.linspace(0,int(self.final_time),self.M).repeat(wprime.size()[0],wprime.size()[1],1).to(self.device)
        
        tilde_g = self.Rayleigh(t,w,mu_g,s)
        tilde_y = self.Rayleigh(t,wprime,mu_y,s)

        return s, tilde_g, tilde_y

def mean_w(model, mean_scaled_trace, time_trace, n_sample_traj=1000, is_scale = True):
    model.data = mean_scaled_trace.repeat(n_sample_traj,1,1)

    mu, log_var = model.encoder()

    model.z = model.reparameterization(mu, log_var)

    s, w, _ = model.reconstruct()

    model.data = time_trace
    
    if is_scale:
        w_mean = ten_to_npy(w*torch.sqrt(s)).mean(axis=0).reshape(-1)
        w_std = ten_to_npy(w*torch.sqrt(s)).std(axis=0).reshape(-1)
    else:
        w_mean = ten_to_npy(w).mean(axis=0).reshape(-1)
        w_std = ten_to_npy(w).std(axis=0).reshape(-1)

    return w_mean, w_std

def derivative_penalty(obs_y, target_size):
    target_size
    interpol_size = 1
    der_y = torch.zeros_like(obs_y)
    for k in range(len(obs_y)):
        der_y[k] = moving_average(obs_y[k],7).view(1,-1)
    der_y = torch.gradient(der_y, axis=2)[0]    
    der_y[torch.where(der_y < 0)] = 0
    der_y = torch.abs(der_y)
    
    while obs_y.size()[-1]*interpol_size <= target_size:
        interpol_size +=1
    der_penalty = (der_y-der_y.min(axis=2).values.unsqueeze(1))/(der_y.max(axis=2).values.unsqueeze(1)-der_y.min(axis=2).values.unsqueeze(1))
    if interpol_size != 1:
        der_diff = (der_penalty[:,:,1:]-der_penalty[:,:,:-1])/interpol_size
        der_penalty = der_penalty.repeat_interleave(interpol_size, dim=2)

    return der_penalty[:,:,:target_size]

def ten_to_npy(x):
    return x.detach().cpu().numpy()

def moving_average(x, n):
    x_roll = x.clone()
    x_tmp = torch.zeros_like(x)
    for k in range(-n+1,n):
        x_tmp += torch.roll(x,k, dims=0)
    x_tmp = x_tmp/(2*(n-1)+1)
    x_roll[n:-n] = x_tmp[n:-n]
    for k in range(1,n):
        x_roll[0][-k] = x[0][-n-k:-k].mean()
    return x_roll