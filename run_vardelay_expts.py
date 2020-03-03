# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os 
import sys
import argparse
import numpy as np
from vardelay_utils import build_generators, build_loss, build_performance
import generators, models
import scipy.io as sio
import torch

parser = argparse.ArgumentParser(description='Recurrent Memory Experiment (Variable Delay Condition -2AFC)')
parser.add_argument('--task', type=int, default=0, help='Task code')
parser.add_argument('--model', type=int, default=0, help='Model code')
parser.add_argument('--lambda_val', type=float, default=0.98, help='lambda (initialization for diagonal terms)')
parser.add_argument('--sigma_val', type=float, default=0.0, help='sigma (initialization for off-diagonal terms)')
parser.add_argument('--rho_val', type=float, default=0.0, help='rho (l2-norm regularization)')

args = parser.parse_args()

diag_val = args.lambda_val
offdiag_val = args.sigma_val
wdecay_coeff = args.rho_val
m_ind = args.model
t_ind = args.task

model_list = ['LeInitRecurrent','GRURecurrent']

# Task and model parameters
model = model_list[m_ind]

n_hid = 500  # number of hidden units

generator, test_generator = build_generators(t_ind)

# Define the input and expected output variable
input_var = None
mask_var = None
if torch.cuda.is_available():
    device='cuda:0'
else:
    device='cpu'

if model == 'LeInitRecurrent':
    model = models.LeInitRecurrent(input_var, mask_var=mask_var, batch_size=generator.batch_size,
                                          n_in=generator.n_in,  n_out=generator.n_out, n_hid=n_hid, diag_val=diag_val,
                                          offdiag_val=offdiag_val,  out_nlin='sigmoid')
elif model == 'GRURecurrent':
    model = models.GRURecurrent(input_var, mask_var=mask_var, batch_size=generator.batch_size, n_in=generator.n_in, n_out=generator.n_out, n_hid=n_hid)

# Build the model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=wdecay_coeff)

def l2_activation_regularization(activations):
    loss = 0
    for i in range(5):
        loss = loss + 1e-4 * torch.pow(activations[int(-1*i)], 2).mean()

    return loss

# TRAINING
s_vec, opt_vec, net_vec, frac_rmse_vec = [], [], [], []
for i, (_, example_input, example_output, example_mask, s, opt_s) in generator:
    example_input = torch.Tensor(example_input).requires_grad_(True)
    example_input = example_input.to(device)

    example_output = torch.Tensor(example_output).requires_grad_(True)
    example_output = example_output.to(device)

    example_mask = torch.Tensor(example_mask).requires_grad_(True)
    example_mask = example_mask.to(device)

    prediction, hiddens = model.forward(example_input)

    if t_ind==2 or t_ind==6 or t_ind==8:
        prediction = torch.clamp(1e-6, 1.0 - 1e-6, prediction)
    else:
        prediction = prediction
    loss = build_loss(prediction, example_output, generator.resp_dur, t_ind) + l2_activation_regularization(hiddens)
    loss.backward()
    optimizer.step()

    s_vec.append(s)
    opt_vec.append(opt_s)
    net_vec.append(np.squeeze(prediction.data.cpu().numpy()[:,-5,:]))
    score = loss.data.cpu()
    if i % 500 == 0:
        opt_vec = np.asarray(opt_vec)
        net_vec = np.asarray(net_vec)
        s_vec   = np.asarray(s_vec)
        infloss = build_performance(s_vec, opt_vec, net_vec, t_ind)
        frac_rmse_vec.append(infloss)
        print('Batch #%d; Absolute loss: %.6f; Fractional loss: %.6f' % (i, score, infloss))
        s_vec   = []
        opt_vec = []
        net_vec = []

# TESTING
model.eval()
delay_vec, s_vec, opt_vec, net_vec, ex_hid_vec, ex_inp_vec = [], [], [], [], [], []
for i, (delay_durs, example_input, example_output, example_mask, s, opt_s) in test_generator:
    with torch.no_grad():
        example_input = torch.Tensor(example_input)
        example_input = example_input.to(device)

        example_output = torch.Tensor(example_output)
        example_output = example_output.to(device)

        example_mask = torch.Tensor(example_mask).requires_grad_(True)
        example_mask = example_mask.to(device)

        prediction, hiddens = model.forward(example_input)

        if t_ind == 2 or t_ind == 6 or t_ind == 8:
            prediction = torch.clamp(1e-6, 1.0 - 1e-6, prediction)
        else:
            prediction = prediction
        s_vec.append(s)
        opt_vec.append(opt_s)
        net_vec.append(np.squeeze(prediction.data.cpu().numpy()[:,-5,:]))
        if i % 500 == 0:
            print("Iteration {}".format(i))
            hid = [hidden.data.cpu().numpy() for hidden in hiddens]
            ex_hid_vec.append(hid)
            ex_inp_vec.append(example_input.cpu().numpy())
            delay_vec.append(delay_durs)

opt_vec = np.asarray(opt_vec)
net_vec = np.asarray(net_vec)
s_vec   = np.asarray(s_vec)
infloss_test = build_performance(s_vec, opt_vec, net_vec, t_ind)
print('Test data; Fractional loss: %.6f' %infloss_test)

# Input and hidden layer activities
ex_hid_vec = np.asarray(ex_hid_vec)
ex_hid_vec = np.reshape(ex_hid_vec,(-1, generator.stim_dur + generator.delay_dur +
                                    generator.resp_dur, ExptDict["n_hid"]))

ex_inp_vec = np.asarray(ex_inp_vec)
ex_inp_vec = np.reshape(ex_inp_vec,(-1, generator.stim_dur + generator.delay_dur +
                                    generator.resp_dur, ExptDict["task"]["n_loc"] * generator.n_in))


torch.save({'all_params_list': model.state_dict(),
             'inpResps': ex_inp_vec,
             'hidResps': ex_hid_vec,
             'frac_rmse_test': infloss_test,
             'frac_rmse_vec': np.asarray(frac_rmse_vec),
             'delay_vec': np.asarray(delay_vec)}, 'vardelay_sigma%f_lambda%f_rho%f_model%i_task%i.pt'%(offdiag_val, diag_val, wdecay_coeff, m_ind, t_ind))


