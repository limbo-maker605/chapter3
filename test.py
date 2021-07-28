import torch
import torch.nn as nn
import torch.optim as optim
import pylab as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cvxpy as cp
from gurobipy import *
import os
import cplex
from torch.autograd import Variable
import gurobipy as grb

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 展平操作
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

def bound_propagation(model, initial_bound):
    l, u = initial_bound
    bounds = []

    for layer in model:
        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)

        elif isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()
                  + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()
                  + layer.bias[:, None]).t()
        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)

        bounds.append((l_, u_))
        l, u = l_, u_
    return bounds

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dnn_2 = nn.Sequential(Flatten(), nn.Linear(784,200), nn.ReLU(),
                            nn.Linear(200,10)).to(device)

model_dnn_4 = nn.Sequential(Flatten(), nn.Linear(784,200), nn.ReLU(),
                            nn.Linear(200,100), nn.ReLU(),
                            nn.Linear(100,100), nn.ReLU(),
                            nn.Linear(100,10)).to(device)

model_cnn = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(8, 8, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(16, 16, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*16, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)
#

for X, y in test_loader:
    X, y = X.to(device), y.to(device)
    break

def plot_images(X, y, yp, M, N):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M * 1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1 - X[i * N + j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i * N + j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i * N + j].max(dim=0)[1] == y[i * N + j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    plt.show()

def form_milp(model, c, initial_bounds, bounds):
    linear_layers = [(layer, bound) for layer, bound in zip(model, bounds) if isinstance(layer, nn.Linear)]
    convolution_layers = [(layer, bound) for layer, bound in zip(model, bounds) if isinstance(layer, nn.Conv2d)]
    d = len(linear_layers) - 1
    d2 = len(convolution_layers) - 1
    # create cvxpy variables
    z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] +
         [cp.Variable(linear_layers[-1][0].out_features)])
    v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers[:-1]]

    # extract relevant matrices
    W = [layer.weight.detach().cpu().numpy() for layer, _ in linear_layers]
    b = [layer.bias.detach().cpu().numpy() for layer, _ in linear_layers]
    l = [l[0].detach().cpu().numpy() for _, (l, _) in linear_layers]
    u = [u[0].detach().cpu().numpy() for _, (_, u) in linear_layers]
    l0 = initial_bounds[0][0].view(-1).detach().cpu().numpy()
    u0 = initial_bounds[1][0].view(-1).detach().cpu().numpy()

    # # create cvxpy variables
    # z2 = ([cp.Variable(layer.in_channels) for layer, _ in convolution_layers] +
    #      [cp.Variable(convolution_layers[-1][0].out_channels)])
    # z2_total = ([cp.Variable(layer.in_channels*3*3) for layer, _ in convolution_layers] +
    #      [cp.Variable(convolution_layers[-1][0].out_channels*3*3)])
    # v2 = [cp.Variable(layer.out_channels, boolean=True) for layer, _ in convolution_layers[:-1]]
    # # extract relevant matrices
    # W2 = [layer.weight.detach().cpu().numpy() for layer, _ in convolution_layers]
    # b2 = [layer.bias.detach().cpu().numpy() for layer, _ in convolution_layers]
    # l2 = [l2[0].detach().cpu().numpy() for _, (l2, _) in convolution_layers]
    # u2 = [u2[0].detach().cpu().numpy() for _, (_, u2) in convolution_layers]
    # l02 = initial_bounds[0][0].view(-1).detach().cpu().numpy()
    # u02 = initial_bounds[1][0].view(-1).detach().cpu().numpy()
    # #
    # # add ReLU constraints
    # len_inchannels = [layer.in_channels*3*3 for layer, _ in convolution_layers] + [convolution_layers[-1][0].out_channels*3*3]
    # len_outchannels = [layer.out_channels for layer, _ in convolution_layers]
    # stride_list = [layer.stride for layer, _ in convolution_layers]
    # s = []
    # for i in range(len(len_outchannels)):
    #     s += list(stride_list[i])
    # num = [x * x for x in s][::2]
    constraints = []
    # count = 784
    # for i in range(len(convolution_layers) - 1):
    #     count = int(count/num[i])
    #     upper = np.reshape(u2[i], (len_outchannels[i], count))
    #     low = np.reshape(l2[i], (len_outchannels[i], count))
    #     constraints += [z2[i + 1] >= np.reshape(W2[i],(len_outchannels[i],len_inchannels[i])) @ z2_total[i] + b2[i],
    #                     z2[i + 1] >= 0,
    #                     cp.multiply(v2[i], upper.mean()) >= z2[i + 1],
    #                     np.reshape(W2[i],(len_outchannels[i],len_inchannels[i])) @ z2_total[i] + b2[i] >= z2[i + 1] + cp.multiply((1 - v2[i]), low.mean())]
    #
    #     # final linear constraint
    # constraints += [z2[d2 + 1] == np.reshape(W2[d2],(len_outchannels[d2],len_inchannels[d2])) @ z2_total[d2] + b2[d2]]
    # # initial bound constraints
    # constraints += [z2[0] >= l02, z2[0] <= u02]

    # add ReLU constraints
    for i in range(len(linear_layers) - 1):
        constraints += [z[i + 1] >= W[i] @ z[i] + b[i],
                        z[i + 1] >= 0,
                        cp.multiply(v[i], u[i]) >= z[i + 1],
                        W[i] @ z[i] + b[i] >= z[i + 1] + cp.multiply((1 - v[i]), l[i])]

    # final linear constraint
    constraints += [z[d + 1] == W[d] @ z[d] + b[d]]

    # initial bound constraints
    constraints += [z[0] >= l0, z[0] <= u0]
    return cp.Problem(cp.Minimize(c @ z[d + 1]), constraints), (z, v)


def form_milp2(model, c, initial_bounds, bounds):
    linear_layers = [(layer, bound) for layer, bound in zip(model, bounds) if isinstance(layer, nn.Linear)]
    d = len(linear_layers) - 1

    # create cvxpy variables
    z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] +
         [cp.Variable(linear_layers[-1][0].out_features)])
    v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers[:-1]]

    # extract relevant matrices
    W = [layer.weight.detach().cpu().numpy() for layer, _ in linear_layers]
    b = [layer.bias.detach().cpu().numpy() for layer, _ in linear_layers]
    l = [l[0].detach().cpu().numpy() for _, (l, _) in linear_layers]
    u = [u[0].detach().cpu().numpy() for _, (_, u) in linear_layers]
    l0 = initial_bounds[0][0].view(-1).detach().cpu().numpy()
    u0 = initial_bounds[1][0].view(-1).detach().cpu().numpy()

    # add ReLU constraints
    constraints = []
    for i in range(len(linear_layers) - 1):
        constraints += [z[i + 1] >= W[i] @ z[i] + b[i],
                        z[i + 1] >= 0,
                        cp.multiply(v[i], u[i]) >= z[i + 1],
                        W[i] @ z[i] + b[i] >= z[i + 1] + cp.multiply((1 - v[i]), l[i])]

    # final linear constraint
    constraints += [z[d + 1] == W[d] @ z[d] + b[d]]

    # initial bound constraints
    constraints += [z[0] >= l0, z[0] <= u0]

    return cp.Problem(cp.Minimize(c @ z[d + 1]), constraints), (z, v)


# 创建小神经网络
model_small = nn.Sequential(Flatten(), nn.Linear(784,50), nn.ReLU(),
                            nn.Linear(50,20), nn.ReLU(),
                            nn.Linear(20,10)).to(device)

def epoch(loader, model, opt=None):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        # print("err:",total_err)
        total_loss += loss.item() * X.shape[0]
        # print("shape",X.shape[0])
        # print("loss",total_loss)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# train dnn_2
# opt = optim.SGD(model_dnn_2.parameters(), lr=1e-1)
# for _ in range(10):
#     train_err, train_loss = epoch(train_loader, model_dnn_2, opt)
#     test_err, test_loss = epoch(test_loader, model_dnn_2)
#     print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
# torch.save(model_dnn_2.state_dict(), "model_dnn_2.pt")
model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))

# train dnn_4
# opt = optim.SGD(model_dnn_4.parameters(), lr=1e-1)
# for _ in range(10):
#     train_err, train_loss = epoch(train_loader, model_dnn_4, opt)
#     test_err, test_loss = epoch(test_loader, model_dnn_4)
#     print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
# torch.save(model_dnn_4.state_dict(), "model_dnn_4.pt")
model_dnn_4.load_state_dict(torch.load("model_dnn_4.pt"))

# train cnn
# opt = optim.SGD(model_cnn.parameters(), lr=1e-1)
# for t in range(10):
#     train_err, train_loss = epoch(train_loader, model_cnn, opt)
#     test_err, test_loss = epoch(test_loader, model_cnn)
#     if t == 4:
#         for param_group in opt.param_groups:
#             param_group["lr"] = 1e-2
    # print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
torch.save(model_cnn.state_dict(), "model_cnn.pt")
# 加载cnn model'
model_cnn.load_state_dict(torch.load("model_cnn.pt"))

# train small model and save to disk
# opt = optim.SGD(model_small.parameters(), lr=1e-1)
# for _ in range(10):
#     train_err, train_loss = epoch(train_loader, model_small, opt)
#     test_err, test_loss = epoch(test_loader, model_small)
#     # print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
# torch.save(model_small.state_dict(), "model_small.pt")

# load model from disk
model_small.load_state_dict(torch.load("model_small.pt"))

def linear_approximation(model, bounds):
    new_bounds = []
    lower_bounds = []
    upper_bounds = []
    gurobi_vars = []

    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', False)
    gurobi_model.setParam('Threads', 1)

    # input layer
    inp_lb = []
    inp_ub = []
    inp_gurobi_vars = []

    # 找到input layer的上下界
    l1 = bounds[0][0][0].view(784, 1)
    u1 = bounds[0][1][0].view(784, 1)
    # l2 = bounds[1][0][0].view(50, 1)
    # u2 = bounds[1][1][0].view(50, 1)
    # l3 = bounds[2][0][0].view(50, 1)
    # u3 = bounds[2][1][0].view(50, 1)
    # l4 = bounds[3][0][0].view(20, 1)
    # u4 = bounds[3][1][0].view(20, 1)
    # l5 = bounds[4][0][0].view(20, 1)
    # u5 = bounds[4][1][0].view(20, 1)
    # l6 = bounds[5][0][0].view(10, 1)
    # u6 = bounds[5][1][0].view(10, 1)

    # input_domain1 = torch.cat((l1, l2, l3, l4, l5, l6), dim=0)
    # input_domain2 = torch.cat((u1, u2, u3, u4, u5, u6), dim=0)
    input_domain = torch.cat((l1, u1), dim=1)
    # print(input_domain)
    for dim, (lb, ub) in enumerate(input_domain):
        v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                              vtype=grb.GRB.CONTINUOUS,
                              name=f'inp_{dim}')
        inp_gurobi_vars.append(v)
        inp_lb.append(lb)
        inp_ub.append(ub)
    gurobi_model.update()
    lower_bounds.append(inp_lb)
    upper_bounds.append(inp_ub)
    new_bounds.append((inp_lb, inp_ub))
    gurobi_vars.append(inp_gurobi_vars)
    # print("一开始的lower_bounds",lower_bounds)
    # print("一开始的upper_bounds",upper_bounds)
    # print("gurobi_vars", gurobi_vars)
    #other layers
    layer_idx = 1
    # layers = [layer for layer in model]
    for layer in model:
        new_layer_lb = []
        new_layer_ub = []
        new_layer_gurobi_vars = []

        if type(layer) is nn.Linear:
            for neuron_idx in range(layer.weight.size(0)):
                ub = layer.bias.data[neuron_idx]
                lb = layer.bias.data[neuron_idx]
                lin_expr = layer.bias.data[neuron_idx]
                ub = ub.item()
                lb = lb.item()
                lin_expr = lin_expr.item()
                for prev_neuron_idx in range(layer.weight.size(1)):
                    # print("prev_neuron_idx",prev_neuron_idx)
                    coeff = layer.weight.data[neuron_idx, prev_neuron_idx]
                    if coeff >= 0:
                        # print("coeff", coeff)
                        # print("upper_bounds[-1][", prev_neuron_idx, "]", upper_bounds[-1][prev_neuron_idx])
                        # print("lower_bounds[-1][", prev_neuron_idx, "]", lower_bounds[-1][prev_neuron_idx])
                        # print(ub)
                        # print(lb)
                        ub += coeff*upper_bounds[-1][prev_neuron_idx]
                        # print("ub[",prev_neuron_idx,"]",ub)
                        lb += coeff*lower_bounds[-1][prev_neuron_idx]
                        # print("lb[",prev_neuron_idx,"]",lb)
                    else:
                        # print("coeff", coeff)
                        # print("upper_bounds[-1][", prev_neuron_idx, "]", upper_bounds[-1][prev_neuron_idx])
                        # print("lower_bounds[-1][", prev_neuron_idx, "]", lower_bounds[-1][prev_neuron_idx])
                        # print(ub)
                        # print(lb)
                        ub += coeff*lower_bounds[-1][prev_neuron_idx]
                        # print("ub[-1][", prev_neuron_idx, "]", ub)
                        lb += coeff*upper_bounds[-1][prev_neuron_idx]
                        # print("lb[-1][", prev_neuron_idx, "]", lb)
                    lin_expr += coeff * gurobi_vars[-1][prev_neuron_idx]

                v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=f'lay{layer_idx}_{neuron_idx}')

                gurobi_model.addConstr(v == lin_expr)
                # print("v",v)
                # print("lin",lin_expr)
                gurobi_model.update()

                # print("这是v", v)
                gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                gurobi_model.optimize()
                # print("gurobi_model.status", gurobi_model.status)
                # print("3-0",v.X)
                # if gurobi_model.status == GRB.Status.INFEASIBLE:
                #     print('Optimization was stopped with status %d' % gurobi_model.status)
                #     # do IIS, find infeasible constraints
                #     gurobi_model.computeIIS()
                #     for c in gurobi_model.getConstrs():
                #         if c.IISConstr:
                #             print('我是%s' % c.constrName)

                assert gurobi_model.status == 2, "LP wasn't optimally solved"
                # We have computed a lower bound
                lb = v.X
                v.lb = lb
                # print("v",v)
                # print("lb",v.X)
                # Let's now compute an upper bound

                gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                gurobi_model.update()
                gurobi_model.reset()
                gurobi_model.optimize()
                # print("gurobi_model.status", gurobi_model.status)
                # 说明infeasible
                assert gurobi_model.status == 2, "LP wasn't optimally solved"
                ub = v.X
                v.ub = ub
                # print("ub-v",v)
                # print("ub",v.X)
                ub_tensor = torch.tensor(ub).clone().detach().requires_grad_(True)
                lb_tensor = torch.tensor(lb).clone().detach().requires_grad_(True)
                new_layer_lb.append(lb_tensor)
                new_layer_ub.append(ub_tensor)
                new_layer_gurobi_vars.append(v)

                # print("new_lb", new_layer_lb)
                # print("new_ub", new_layer_ub)
                # print("new_layer_gurobi_vars", new_layer_gurobi_vars)
        elif type(layer) == nn.ReLU:
            for neuron_idx, pre_var in enumerate(gurobi_vars[-1]):
                pre_lb = lower_bounds[-1][neuron_idx]
                pre_ub = upper_bounds[-1][neuron_idx]
                # print("我是pre_lb",pre_lb)
                # print("我是pre_ub",pre_ub)
                v = gurobi_model.addVar(lb=max(0, pre_lb),
                                      ub=max(0, pre_ub),
                                      obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=f'ReLU{layer_idx}_{neuron_idx}')

                if pre_lb >= 0 and pre_ub >= 0:
                    # The ReLU is always passing
                    gurobi_model.addConstr(v == pre_var)
                    lb = pre_lb
                    ub = pre_ub
                elif pre_lb <= 0 and pre_ub <= 0:
                    lb = 0
                    ub = 0
                    # No need to add an additional constraint that v==0
                    # because this will be covered by the bounds we set on
                    # the value of v.
                else:
                    lb = 0
                    ub = pre_ub
                    gurobi_model.addConstr(v >= pre_var)

                    slope = pre_ub / (pre_ub - pre_lb)
                    bias = - pre_lb * slope
                    gurobi_model.addConstr(v <= slope * pre_var + bias)

                ub_tensor = torch.tensor(ub).clone().detach().requires_grad_(True)
                lb_tensor = torch.tensor(lb).clone().detach()
                new_layer_lb.append(lb_tensor)
                new_layer_ub.append(ub_tensor)
                new_layer_gurobi_vars.append(v)

                # print("new_lb2", new_layer_lb)
                # print("new_ub2",new_layer_ub)
                # print("new_layer_gurobi_vars2",new_layer_gurobi_vars)
        elif type(layer) == nn.MaxPool1d:
            assert layer.padding == 0, "Non supported Maxpool option"
            assert layer.dilation == 1, "Non supported MaxPool option"
            nb_pre = len(gurobi_vars[-1])
            window_size = layer.kernel_size
            stride = layer.stride

            pre_start_idx = 0
            pre_window_end = pre_start_idx + window_size

            while pre_window_end <= nb_pre:
                lb = max(lower_bounds[-1][pre_start_idx:pre_window_end])
                ub = max(upper_bounds[-1][pre_start_idx:pre_window_end])

                neuron_idx = pre_start_idx // stride

                v = model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                      name=f'Maxpool{layer_idx}_{neuron_idx}')
                all_pre_var = 0
                for pre_var in gurobi_vars[-1][pre_start_idx:pre_window_end]:
                    model.addConstr(v >= pre_var)
                    all_pre_var += pre_var
                all_lb = sum(lower_bounds[-1][pre_start_idx:pre_window_end])
                max_pre_lb = lb
                gurobi_model.addConstr(all_pre_var >= v + all_lb - max_pre_lb)

                pre_start_idx += stride
                pre_window_end = pre_start_idx + window_size

                new_layer_lb.append(lb)
                new_layer_ub.append(ub)
                new_layer_gurobi_vars.append(v)
        elif type(layer) == Flatten:
            continue
        else:
            raise NotImplementedError

        lower_bounds.append(new_layer_lb)
        upper_bounds.append(new_layer_ub)
        new_bounds.append((new_layer_lb, new_layer_ub))
        gurobi_vars.append(new_layer_gurobi_vars)
        layer_idx += 1

        # Assert that this is as expected a network with a single output
    # assert len(gurobi_vars[-1]) == 1, "Network doesn't have scalar output"
    # print("lb", lower_bounds)
    # print("ub", upper_bounds)
    # print(new_bounds)
    gurobi_model.update()
    return new_bounds





epsilon = 0.5
initial_bound = ((X[0:1] - epsilon).clamp(min=0), (X[0:1] + epsilon).clamp(max=1))
# # 计算cnn 边界
bounds = bound_propagation(model_small, initial_bound)
new_bounds = linear_approximation(model_small, bounds)
c = np.zeros(10)
c[y[0].item()] = 1
c[2] = -1
# 使用gurobi去计算，负数说明存在对抗样本


prob, (z, v) = form_milp2(model_small, c, initial_bound, new_bounds)
prob.solve(solver=cp.GUROBI, verbose=True)



# 找到input layer的上下界
# l1 = bounds[0][0][0].view(784, 1)
# u1 = bounds[0][1][0].view(784, 1)
# l2 = bounds[1][0][0].view(50, 1)
# u2 = bounds[1][1][0].view(50, 1)
# input_domain1 = torch.cat((l1, l2), dim=0)
# input_domain2 = torch.cat((u1, u2), dim=0)
# input_domain = torch.cat((input_domain1, input_domain2), dim=1)
#
# for dim, (lb, ub) in enumerate(input_domain):
#     print(dim)
#     print(lb)
#     print(ub)
# input_domain = torch.cat(bounds[0][0][0],bounds[0][1][0])
# print(bounds[0])

# print(prob.value)
# print("Last layer values from MILP:", z[3].value)
# print("Last layer from model:",
#       model_small(torch.tensor(z[0].value).float().view(1,1,28,28).to(device))[0].detach().cpu().numpy())
# plt.imshow(1-z[0].value.reshape(28,28), cmap="gray")
# plt.show()

# epsilon = 0.1
# initial_bound = ((X[0:1] - epsilon).clamp(min=0), (X[0:1] + epsilon).clamp(max=1))
# bounds = bound_propagation(model_dnn_2, initial_bound)

# for y_targ in range(10):
#     if y_targ != y[0].item():
#         c = np.eye(10)[y[0].item()] - np.eye(10)[y_targ]
#         prob, _ = form_milp(model_cnn, c, initial_bound, bounds)
#         print("Targeted attack {} objective: {}".format(y_targ, prob.solve(solver=cp.GUROBI)))