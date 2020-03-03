# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os
import sys
import argparse
import numpy as np
from utils import build_generators, build_model, build_loss, build_performance

parser = argparse.ArgumentParser(description='Recurrent Memory Experiment (Tethered Condition)')
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

# Models and model-specific parameters
model_list = [{"model_id":'LeInitRecurrent',"diag_val":diag_val,"offdiag_val":offdiag_val},
              {"model_id":'OrthoInitRecurrent',"init_val":diag_val},
              {"model_id":'GRURecurrent',"diag_val":diag_val,"offdiag_val":offdiag_val}]

# Tasks and task-specific parameters
task_list = [{"task_id":'DE1', "n_out":1, "n_loc":1, "out_nonlin":'linear'},
             {"task_id":'DE2', "n_out":2, "n_loc":2, "out_nonlin":'linear'},
             {"task_id":'CD1', "n_out":1, "n_loc":1, "out_nonlin":'sigmoid'},
             {"task_id":'CD2', "n_out":1, "n_loc":2, "out_nonlin":'sigmoid'},
             {"task_id":'GDE2',"n_out":1, "n_loc":2, "out_nonlin":'linear'},
             {"task_id":'VDE1',"n_out":1, "n_loc":1, "max_delay":100, "out_nonlin":'linear'},
             {"task_id":'Harvey2012', "n_out":1, "sigtc":15.0, "stim_rate":1.0, "n_loc":1, "out_nonlin":'linear'},
             {"task_id":'SINE', "n_out":1, "n_loc":1, "alpha":0.25, "out_nonlin":'linear'},
             {"task_id":'COMP', "n_out":1, "n_loc":1, "out_nonlin": 'linear'}
             ]

# Task and model parameters
ExptDict = {"model": model_list[m_ind],
            "task": task_list[t_ind],
            "tr_cond": 'all_gains',
            "test_cond": 'all_gains',
            "n_hid": 500,
            "n_in": 50,
            "batch_size": 50,
            "stim_dur": 25,
            "delay_dur": 100,
            "resp_dur": 25,
            "kappa": 2.0,
            "spon_rate": 0.1,
            "tr_max_iter": 25001,
            "test_max_iter": 2501}

# Build task generators
generator, test_generator = build_generators(ExptDict)

# Define the input and expected output variable
input_var = None
if torch.cuda.is_available():
    device='cuda:0'
else:
    device='cpu'

# Build the model
model = build_model(input_var, ExptDict).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=wdecay_coeff)

def l2_activation_regularization(activations):
    loss = 0
    for i in range(5):
        loss = loss + 1e-1 * torch.pow(activations[int(-1*i)], 2).mean()

    return loss

# TRAINING
s_vec, opt_vec, net_vec, infloss_vec = [], [], [], []
for i, (example_input, example_output, s, opt) in generator:
    example_input = torch.Tensor(example_input).requires_grad_(True)
    example_input = example_input.to(device)

    example_output = torch.Tensor(example_output).requires_grad_(True)
    example_output = example_output.to(device)

    prediction, hiddens = model.forward(example_input)
    if ExptDict["task"]["task_id"] in ['DE1', 'DE2', 'GDE2', 'VDE1', 'SINE']:
        prediction = prediction
    elif ExptDict["task"]["task_id"] in ['CD1', 'CD2', 'Harvey2012', 'Harvey2012Dynamic', 'Harvey2016', 'COMP']:
        prediction = torch.clamp(1e-6, 1.0 - 1e-6, prediction)
    loss = build_loss(prediction, example_output, ExptDict) + l2_activation_regularization(hiddens)
    loss.backward()
    optimizer.step()

    s_vec.append(s)
    opt_vec.append(opt)
    net_vec.append(np.squeeze(prediction.data.cpu().numpy()[:,-5,:]))
    score = loss.data.cpu()
    if i % 500 == 0:
        opt_vec = np.asarray(opt_vec)
        net_vec = np.asarray(net_vec)
        s_vec   = np.asarray(s_vec)
        infloss = build_performance(s_vec,opt_vec,net_vec,ExptDict)
        infloss_vec.append(infloss)
        print('Batch #%d; Absolute loss: %.6f; Fractional loss: %.6f' % (i, score, infloss))
        s_vec   = []
        opt_vec = []
        net_vec = []

# TESTING
model.eval()
s_vec, opt_vec, net_vec, ex_hid_vec, ex_inp_vec = [], [], [], [], []
for i, (example_input, example_output, s, opt) in test_generator:
    with torch.no_grad():
        example_input = torch.Tensor(example_input)
        example_input = example_input.to(device)

        example_output = torch.Tensor(example_output)
        example_output = example_output.to(device)

        prediction, hiddens = model.forward(example_input)
        if ExptDict["task"]["task_id"] in ['DE1', 'DE2', 'GDE2', 'VDE1', 'SINE']:
            prediction = prediction
        elif ExptDict["task"]["task_id"] in ['CD1', 'CD2', 'Harvey2012', 'Harvey2012Dynamic', 'Harvey2016', 'COMP']:
            prediction = torch.clamp(1e-6, 1.0 - 1e-6, prediction)
        s_vec.append(s)
        opt_vec.append(opt)
        net_vec.append(np.squeeze(prediction.data.cpu().numpy()[:,-5,:]))
        if i % 500 == 0:
            print("Iteration {}".format(i))
            hid = [hidden.data.cpu().numpy() for hidden in hiddens]
            ex_hid_vec.append(hid)
            ex_inp_vec.append(example_input.cpu().numpy())

opt_vec = np.asarray(opt_vec)
net_vec = np.asarray(net_vec)
s_vec   = np.asarray(s_vec)
infloss_test = build_performance(s_vec,opt_vec,net_vec,ExptDict)
print('Test data; Fractional loss: %.6f' %infloss_test)

# Input and hidden layer activities
ex_hid_vec = np.asarray(ex_hid_vec)
ex_hid_vec = np.reshape(ex_hid_vec,(-1, generator.stim_dur + generator.delay_dur +
                                    generator.resp_dur, ExptDict["n_hid"]))

ex_inp_vec = np.asarray(ex_inp_vec)
ex_inp_vec = np.reshape(ex_inp_vec,(-1, generator.stim_dur + generator.delay_dur +
                                    generator.resp_dur, ExptDict["task"]["n_loc"] * generator.n_in))


torch.save({'all_params_list':model.state_dict(),
             'inpResps':ex_inp_vec,
             'hidResps':ex_hid_vec,
             'infloss_test':infloss_test,
             'infloss_vec':np.asarray(infloss_vec)}, 'tethered_sigma%f_lambda%f_rho%f_model%i_task%i.pt'%(offdiag_val, diag_val, wdecay_coeff, m_ind, t_ind))