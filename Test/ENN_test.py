# coding:utf-8

import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
import os

# 参数定义
device = 'cuda'

model_name = 'ENN_fem'

path_model = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/ENN/ENN_model/' + model_name + '.pth'

data_name = 'train_fem'

path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/ENN/data/' + data_name

dim = 1

dim_yield = 30

dim_q = 30

dim_evolution = 30

newton_num = 2

loss_weight_in = 1

loss_weight_b = 1

e_mod = 80

miu = 0.25

e_lam = torch.tensor(e_mod * miu / (1 + miu) / (1 - 2 * miu)).type(torch.FloatTensor).to(device)

e_g = torch.tensor(e_mod / 2 / (1 + miu)).type(torch.FloatTensor).to(device)

e_2g = torch.tensor(e_mod / (1 + miu)).type(torch.FloatTensor).to(device)

C = torch.tensor([[e_lam + e_2g, e_lam, 0, e_lam], [e_lam, e_lam + e_2g, 0, e_lam], [0, 0, e_g, 0], [e_lam, e_lam, 0, e_lam + e_2g]]).to(device)

C_inv = torch.linalg.inv(C).to(device)

I = torch.eye(dim).to(device)

# 导入数据
e11 = np.load(path + '/e11.npy', allow_pickle=True)
e22 = np.load(path + '/e22.npy', allow_pickle=True)
e12 = np.load(path + '/e12.npy', allow_pickle=True)
e33 = np.load(path + '/e33.npy', allow_pickle=True)
time = np.load(path + '/steptime.npy', allow_pickle=True).shape[0]
element = np.load(path + '/element.npy', allow_pickle=True)
s11 = np.load(path + '/s11.npy', allow_pickle=True)
s22 = np.load(path + '/s22.npy', allow_pickle=True)
s12 = np.load(path + '/s12.npy', allow_pickle=True)
s11 = torch.from_numpy(s11).type(torch.FloatTensor).to(device)
s22 = torch.from_numpy(s22).type(torch.FloatTensor).to(device)
s12 = torch.from_numpy(s12).type(torch.FloatTensor).to(device)


def increment(input):
    # input [batch, time, 4]
    time = input.shape[1]
    output = input
    for i in range(0, time - 1):
        output[:, time - i - 1, :] = input[:, time - i - 1, :] - input[:, time - i - 2, :]
    return output  # output [batch, time, 4]


def DDSDDE(input):
    # input [batch, time ,4] [e11, e22, e12, e33]
    output = np.zeros_like(input)
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            output[i][j][0] = (e_lam + e_2g) * input[i][j][0] + e_lam * input[i][j][1] + e_lam * input[i][j][3]
            output[i][j][1] = e_lam * input[i][j][0] + (e_lam + e_2g) * input[i][j][1] + e_lam * input[i][j][3]
            output[i][j][2] = e_g * input[i][j][2]
            output[i][j][3] = e_lam * input[i][j][0] + e_lam * input[i][j][1] + (e_lam + e_2g) * input[i][j][3]
    return output  # output time 4 [s11, s22, s12, s33]


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True,
                                   )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


class F_yield(nn.Module):
    def __init__(self, n_input_q=dim + 4, n_input_s=4):
        super(F_yield, self).__init__()

        self.z_1 = nn.Linear(n_input_s, dim_yield)
        self.z_2 = nn.Parameter(torch.randn(dim_yield, dim_yield))
        self.z_3 = nn.Parameter(torch.randn(dim_yield, dim_yield))
        self.z_4 = nn.Parameter(torch.randn(dim_yield, 1))

        self.sz_1 = nn.Linear(n_input_s, dim_yield, bias=False)
        self.sz_2 = nn.Linear(n_input_s, dim_yield, bias=False)
        self.sz_3 = nn.Linear(n_input_s, 1, bias=False)

        self.q_1 = nn.Linear(n_input_q, dim_q, bias=False)
        self.q_2 = nn.Linear(dim_q, dim_q, bias=False)
        self.q_3 = nn.Linear(dim_q, dim_q, bias=False)
        self.q_4 = nn.Linear(dim_q, 1, bias=False)

        self.act_s = nn.ELU()

        self.act_q = nn.LeakyReLU()

    def forward(self, input_pe, input_q, input_s):
        # inputs: [1, 1]
        input_q.requires_grad_(True)
        input_s.requires_grad_(True)

        input_pe.requires_grad_(True)

        z1 = self.act_s(self.z_1(input_s))
        z2 = self.act_s(torch.matmul(z1, torch.abs(self.z_2)) + self.sz_1(input_s))
        z3 = self.act_s(torch.matmul(z2, torch.abs(self.z_3)) + self.sz_2(input_s))
        z4 = torch.matmul(z3, torch.abs(self.z_4)) + self.sz_3(input_s)

        input_pe_q = torch.cat((input_pe, input_q), dim=1)
        q1 = self.act_q(self.q_1(input_pe_q))
        q2 = self.act_q(self.q_2(q1))
        q3 = self.act_q(self.q_3(q2))
        q4 = self.q_4(q3)

        output = z4 - q4  # batch 1
        r = gradients(output, input_s, 1)  # batch 4

        r_s = torch.zeros(input_pe.shape[0], 4, 4).to(device)
        for i in range(0, 4):
            r_s[:, 0:4, i] = gradients(r[:, i], input_s)

        r_q = torch.zeros((input_pe.shape[0], 4, dim)).to(device)

        f_d = torch.cat((r, gradients(output, input_q, 1)), dim=1)  # batch dim + 4

        f_pe = gradients(output, input_pe, 1)

        return output, r, r_s, r_q, f_d, f_pe  # [batch_size, seq_len, d_model]


class F_evolution(nn.Module):
    def __init__(self):
        super(F_evolution, self).__init__()
        self.fc = nn.Sequential(nn.Linear(dim + 8, dim_evolution, bias=True),
                                nn.LeakyReLU(),
                                nn.Linear(dim_evolution, dim_evolution, bias=True),
                                nn.LeakyReLU(),
                                nn.Linear(dim_evolution, dim_evolution, bias=True),
                                nn.LeakyReLU(),
                                nn.Linear(dim_evolution, dim, bias=True),
                                )

    def forward(self, input_pe, input_q, input_s):
        # inputs: [4]
        input_q.requires_grad_(True)
        input_s.requires_grad_(True)

        input_pe.requires_grad_(True)

        inputs = torch.cat((input_pe, input_q, input_s), dim=1)

        output = self.fc(inputs)

        h_s = torch.zeros(input_pe.shape[0], dim, 4).to(device)
        for i in range(0, dim):
            h_s[:, i, 0:4] = gradients(output[:, i], input_s)

        h_q = torch.zeros(input_pe.shape[0], dim, dim).to(device)
        for i in range(0, dim):
            h_q[:, i, 0:dim] = gradients(output[:, i], input_q)

        h_pe = torch.zeros(input_pe.shape[0], dim, 4).to(device)
        for i in range(0, dim):
            h_pe[:, i, 0:4] = gradients(output[:, i], input_pe)

        return output, h_s, h_q, h_pe  # [batch_size, seq_len, d_model]

class ENN(nn.Module):
    def __init__(self):
        super(ENN, self).__init__()

        self.f_yield = F_yield()

        self.f_evolution = F_evolution()

    def forward(self, x_in, d_stress, property):
        # x_in : batch, time, 4, d_stress : batch, time, 4
        x_stress = torch.zeros((x_in.shape[0], x_in.shape[1], 4)).type(torch.FloatTensor).to(
            device)  # time 4  [s11, s22, s12, s33]

        # 定义内变量
        pe = torch.zeros((x_in.shape[0], 4)).to(device)
        q = torch.zeros((x_in.shape[0], dim)).to(device)

        for i in range(1, time):
            stress_trial = x_stress[:, i - 1, :] + d_stress[:, i, :]

            x_stress[:, i, :] = stress_trial

            # batch 1; batch 4; batch 4 4; batch dim dim; batch 4+dim
            f_y, r, r_s, r_q, f_d, f_pe = self.f_yield(pe, q, stress_trial)

            # 寻找f_y
            yield_ele = f_y[:, 0] > 1e-6

            if yield_ele.sum() != 0:
                d_lambda = torch.zeros((x_in.shape[0], 1, 1)).to(device)  # yield_num, 1 1
                pe_new = pe.clone()
                q_new = q.clone()
                stress_trial_new = stress_trial.clone()
                for newton in range(0, newton_num):

                    # batch dim; batch dim 4; batch dim dim
                    h, h_s, h_q, h_pe = self.f_evolution(pe_new[yield_ele, :], q_new[yield_ele, :],
                                                         stress_trial_new[yield_ele, :])

                    # C_yield batch 4 4;
                    C_yield = C_inv.unsqueeze(0).repeat(h.shape[0], 1, 1)

                    # A_up batch 4 4+dim
                    A_up = torch.cat(((C_yield + torch.mul(d_lambda[yield_ele, :], r_s[yield_ele, :])),
                                      torch.mul(d_lambda[yield_ele, :], r_q[yield_ele, :])), dim=2)

                    I_yield = I.unsqueeze(0).repeat(h.shape[0], 1, 1)
                    # A_up batch dim 4+dim
                    # A_down = torch.cat((torch.mul(d_lambda[yield_ele, :], h_s), -I_yield + torch.mul(d_lambda[yield_ele, :], h_q)), dim=2)

                    # -----------------------------------------
                    A_down = torch.cat((torch.mul(d_lambda[yield_ele, :], h_s - h_pe @ C_inv),
                                        -I_yield + torch.mul(d_lambda[yield_ele, :], h_q)), dim=2)

                    A_inv = torch.cat((A_up, A_down), dim=1)

                    A = torch.linalg.inv(A_inv)

                    a = -pe_new[yield_ele, :] + pe[yield_ele, :] + torch.mul(d_lambda[yield_ele, :].squeeze(2),
                                                                             r[yield_ele, :])

                    b = -q_new[yield_ele, :] + q[yield_ele, :] + torch.mul(d_lambda[yield_ele, :].squeeze(2), h)

                    a_ = torch.cat((a, b), dim=1).unsqueeze(2)

                    # ----------------------------------
                    f_d[:, :4] -= f_pe @ C_inv

                    r_ = torch.cat((r[yield_ele, :], h), dim=1).unsqueeze(2)  # batch dim+4 1
                    # batch_yield, 1
                    derta_lambda = (f_y[yield_ele, :] - torch.matmul(torch.matmul(f_d[yield_ele, :].unsqueeze(1), A),
                                                                     a_).squeeze(2)) \
                                   / torch.matmul(torch.matmul(f_d[yield_ele, :].unsqueeze(1), A), r_).squeeze(2)
                    # batch_yield dim+4 1
                    derta_s_q = -torch.matmul(A, a_) - torch.mul(derta_lambda.unsqueeze(2), torch.matmul(A, r_))
                    # batch_yield 4 1
                    derta_s = derta_s_q[:, 0: 4, :]
                    # batch_yield dim 1
                    derta_q = derta_s_q[:, 4: 4 + dim, :]
                    # batch_yield 4 1
                    derta_pe = -torch.matmul(C_yield, derta_s)

                    pe_new[yield_ele, :] = pe_new[yield_ele, :] + derta_pe.squeeze(2)

                    q_new[yield_ele, :] = q_new[yield_ele, :] + derta_q.squeeze(2)

                    d_lambda[yield_ele, :] = d_lambda[yield_ele, :] + derta_lambda.unsqueeze(2)

                    x_stress[yield_ele, i, :] = x_stress[yield_ele, i, :] + derta_s.squeeze(2)

                    stress_trial_new = x_stress[:, i, :].clone()

                    f_y, r, r_s, r_q, f_d, f_pe = self.f_yield(pe_new, q_new, stress_trial_new)

                    # 寻找f_y
                    yield_ele = f_y[:, 0] > 1e-6

                    if (yield_ele.sum() == 0) | (newton == newton_num - 1):
                        pe = pe_new
                        q = q_new
                        break

        # x_stress batch time 4
        if property == 'test':
            return x_stress[:, :, 0], x_stress[:, :, 1], x_stress[:, :, 2], x_stress[:, :, 3]



enn = ENN()
original = torch.load(path_model)

enn.load_state_dict(original)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in enn.parameters())))
enn.eval()
enn.to(device)

data_in = np.zeros((element.shape[0], time, 4))
for i in range(0, element.shape[0]):
    for j in range(0, time):
        data_in[i][j][0] = e11[element[i][0] - 1][j]
        data_in[i][j][1] = e22[element[i][0] - 1][j]
        data_in[i][j][2] = e12[element[i][0] - 1][j]
        data_in[i][j][3] = e33[element[i][0] - 1][j]
data_in = data_in.astype(np.float32)
data_in_d_s = DDSDDE(increment(data_in))
data_in = torch.from_numpy(data_in).to(device)
data_in_d_s = torch.from_numpy(data_in_d_s).to(device)

s11_out, s22_out, s12_out, s33_out = enn(data_in, data_in_d_s, 'test')

loss_fn = torch.nn.SmoothL1Loss()

loss = loss_fn(s11_out, s11).data.item() + loss_fn(s22_out, s22).data.item() + loss_fn(s12_out, s12).data.item()

print(loss)