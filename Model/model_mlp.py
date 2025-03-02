"""
This is Udacity actor and critic model mostly.

Optionally (NOT Used Currently):
    Actor may use differentiable clamp function instead of tanh on MLP output
    See link below for more details
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# --- torch.clamp set zero gradientGradient on out of boundary input ---
# ---                  Use    DifferentiableClamp                    ---
# ----------------------------------------------------------------------

# from torch.cuda.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """

    @staticmethod
    # @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """
    return DifferentiableClamp.apply(input, min, max)


# ---------------- Model -------------------------------


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Deterministic Policy Action) Model."""

    regularization_BN = 'BatchNormalization'
    regularization_DropOut = 'DropOut'

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300,
                 regularization=regularization_BN, drop_out_val=0.25):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            regularization (str): A type of regularization: Batch Normalization, Drop Out, ...
            drop_out_val ( float in [0, 1[ ): percent of neurons to drop out
        """
        super(Actor, self).__init__()
        self.state_size     = state_size
        self.action_size    = action_size
        self.fc1_units      = fc1_units
        self.fc2_units      = fc2_units
        self.regularization = regularization
        self.drop_out_val   = drop_out_val

        self.fc1 = nn.Linear(state_size, self.fc1_units)
        if self.regularization == self.regularization_BN:
            self.regular1 = nn.Sequential(nn.BatchNorm1d(self.fc1_units), nn.ReLU())
        elif self.regularization == self.regularization_DropOut:
            self.regular1 = nn.Sequential(nn.ReLU(), nn.Dropout(self.drop_out_val))
        else:
            assert False
        self.fc2 = nn.Linear(self.fc1_units, self.fc2_units)
        if self.regularization == self.regularization_BN:
            self.regular2 = nn.Sequential(nn.BatchNorm1d(self.fc2_units), nn.ReLU())
        elif self.regularization == self.regularization_DropOut:
            self.regular2 = nn.Sequential(nn.ReLU(), nn.Dropout(self.drop_out_val))
        else:
            assert False
        self.fc3 = nn.Linear(self.fc2_units, action_size)
        self.fc3_activation = nn.Tanh()  # This may move a gradient to zero
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    # TODO: Use torch.clamp instead of torch.tanh
    # TODO: If torch.clamp has not gradient 1 at boundaries then implement clamp with backward gradient is one
    #  as in the link below.
    #  https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        squeeze_output = False
        if state.dim() == 1:  # If it's a 1D tensor
            state = state.unsqueeze(0)  # Add a batch dimension
            squeeze_output = True
        x = self.regular1(self.fc1(state))
        x = self.regular2(self.fc2(x))
        x = self.fc3(x)
        out = self.fc3_activation(x)
        if squeeze_output:
            out = out.squeeze()
        return out


class Critic(nn.Module):
    """Critic (state-action value) Model."""

    regularization_No = 'No'
    regularization_BN = 'BatchNormalization'
    regularization_DropOut = 'DropOut'

    def __init__(self, state_size, action_size, fcs1_units=400, fc2_units=300,
                 regularization=regularization_No, drop_out_val=0.25):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            regularization (str): A type of regularization: Batch Normalization, Drop Out, ...
            drop_out_val ( float in [0, 1[ ): percent of neurons to drop out
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fcs1_units = fcs1_units
        self.fc2_units = fc2_units
        self.regularization = regularization
        self.drop_out_val   = drop_out_val

        self.fcs1 = nn.Linear(state_size, self.fcs1_units)
        if self.regularization == self.regularization_BN:
            self.regular1 = nn.Sequential(nn.BatchNorm1d(self.fcs1_units), nn.ReLU())
        elif self.regularization == self.regularization_DropOut:
            self.regular1 = nn.Sequential(nn.ReLU(), nn.Dropout(self.drop_out_val))
        elif self.regularization == self.regularization_No:
            self.regular1 = nn.ReLU()
        else:
            assert False
        self.fc2 = nn.Linear(self.fcs1_units+action_size, self.fc2_units)
        if self.regularization == self.regularization_BN:
            self.regular2 = nn.Sequential(nn.BatchNorm1d(self.fc2_units), nn.ReLU())
        elif self.regularization == self.regularization_DropOut:
            self.regular2 = nn.Sequential(nn.ReLU(), nn.Dropout(self.drop_out_val))
        elif self.regularization == self.regularization_No:
            self.regular2 = nn.ReLU()
        else:
            assert False
        self.fc3 = nn.Linear(self.fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        # --------- With ReLU only --------------------
        xs = self.regular1(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = self.regular2(self.fc2(x))

        # -------- With Batch Normalization --------------
        # xs = self.bn1(self.fcs1(state))
        # x = torch.cat((xs, action), dim=1)
        # x = self.bn2(self.fc2(x))

        return self.fc3(x)


# ----------------------------------------------------------------------
# ------------------------ Torch.Clamp Gradient ------------------------
# ----------------------------------------------------------------------

# https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def check_torch_clamp_gradient():
    x = torch.from_numpy(np.array(([1])).astype(np.float32))  # one scalar as input
    layer = nn.Linear(1, 1, bias=False)  # neural net with one weight
    optimizer = torch.optim.Adam(params=layer.parameters(), lr=1e-3)

    for i in range(101):
        w = list(layer.parameters())[0]  # weight before backprop
        y = layer(x)  # y = w * x
        f_y = torch.clamp(y, min=4, max=6)  # f(y) = clip(y)
        loss = torch.abs(f_y - 5)   # absolute error, zero if f(y) = 5

        optimizer.zero_grad()
        loss.backward()
        grad = w.grad

        if (i % 100 == 0) or (i == 0):
            print('iteration {}'.format(i))
            print('w: {:.2f}'.format(w.detach().numpy()[0][0]))
            print('y: {:.2f}'.format(y.detach().numpy()[0]))
            print('f_y: {:.2f}'.format(f_y.detach().numpy()[0]))
            print('loss: {:.2f}'.format(loss.detach().numpy()[0]))
            print('grad: {:.2f}\n'.format(grad.detach().numpy()[0][0]))

        optimizer.step()


def check_class_Clamp_grad():
    class Clamp(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            return input.clamp(min=4, max=6)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone()

    clamp_class = Clamp()

    x = torch.from_numpy(np.array(([1])).astype(np.float32))  # one scalar as input
    layer = nn.Linear(1, 1, bias=False)  # neural net with one weight
    optimizer = torch.optim.Adam(params=layer.parameters(), lr=1e-3)

    y_s = []
    f_y_s = []
    loss_s = []
    grad_s = []

    is_custom_clamp = True  # Set it to True to calculate the custom clamp.
    for i in range(10001):
        w = list(layer.parameters())[0]  # weight before backprop
        y = layer(x)  # y = w * x
        if is_custom_clamp:
            clamp = clamp_class.apply
            f_y = clamp(y)  # f(y) = clip(y)
        else:
            f_y = torch.clamp(y, min=4, max=6)
        loss = torch.abs(f_y - 5)  # absolute error, zero if f(y) = 2

        optimizer.zero_grad()
        loss.backward()
        grad = w.grad

        if (i % 100 == 0) or (i == 0):
            print('iteration {}'.format(i))
            print('w: {:.10f}'.format(w.detach().numpy()[0][0]))
            print('y: {:.10f}'.format(y.detach().numpy()[0]))
            print('f_y: {:.10f}'.format(f_y.detach().numpy()[0]))
            print('loss: {:.10f}'.format(loss.detach().numpy()[0]))
            # print('grad: {:.2f}\n'.format(grad.detach().numpy()[0][0]))
            print('grad: {:.10f}\n'.format(grad.detach().numpy()[0][0]))

            y_s.append(y.detach().numpy()[0])
            f_y_s.append(f_y.detach().numpy()[0])
            loss_s.append(loss.detach().numpy()[0])
            grad_s.append(grad.detach().numpy()[0][0])
        optimizer.step()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(y_s)), y_s, label='y')
    plt.plot(np.arange(len(f_y_s)), f_y_s, label='f_y')
    plt.plot(np.arange(len(loss_s)), loss_s, label='loss')
    plt.plot(np.arange(len(grad_s)), grad_s, label='grad')
    plt.ylabel(f'y, f_y, loss, grad')
    plt.xlabel('hundred of iterations')
    plt.legend(loc="center right")
    if is_custom_clamp:
        plt.title(f'Custom clamp')
    else:
        plt.title(f'Torch Clamp')

    plt.show()


def check_DifferentiableClamp_grad():
    # from torch.cuda.amp import custom_bwd, custom_fwd

    class DifferentiableClamp(torch.autograd.Function):
        """
        In the forward pass this operation behaves like torch.clamp.
        But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
        """

        @staticmethod
        # @custom_fwd
        def forward(ctx, input, min, max):
            return input.clamp(min=min, max=max)

        @staticmethod
        # @custom_bwd
        def backward(ctx, grad_output):
            return grad_output.clone(), None, None

    def dclamp(input, min, max):
        """
        Like torch.clamp, but with a constant 1-gradient.
        :param input: The input that is to be clamped.
        :param min: The minimum value of the output.
        :param max: The maximum value of the output.
        """
        return DifferentiableClamp.apply(input, min, max)

    # clamp_class = Clamp()

    x = torch.from_numpy(np.array(([1])).astype(np.float32))  # one scalar as input
    layer = nn.Linear(1, 1, bias=False)  # neural net with one weight
    optimizer = torch.optim.Adam(params=layer.parameters(), lr=1e-3)

    y_s = []
    f_y_s = []
    loss_s = []
    grad_s = []

    is_custom_clamp = True  # Set it to True to calculate the custom clamp.
    for i in range(10001):
        w = list(layer.parameters())[0]  # weight before backprop
        y = layer(x)  # y = w * x
        if is_custom_clamp:
            f_y = dclamp(y, min=4, max=6)  # f(y) = clip(y)
        else:
            f_y = torch.clamp(y, min=4, max=6)
        loss = torch.abs(f_y - 5)  # absolute error, zero if f(y) = 2

        optimizer.zero_grad()
        loss.backward()
        grad = w.grad

        if (i % 100 == 0) or (i == 0):
            print('iteration {}'.format(i))
            print('w: {:.10f}'.format(w.detach().numpy()[0][0]))
            print('y: {:.10f}'.format(y.detach().numpy()[0]))
            print('f_y: {:.10f}'.format(f_y.detach().numpy()[0]))
            print('loss: {:.10f}'.format(loss.detach().numpy()[0]))
            # print('grad: {:.2f}\n'.format(grad.detach().numpy()[0][0]))
            print('grad: {:.10f}\n'.format(grad.detach().numpy()[0][0]))

            y_s.append(y.detach().numpy()[0])
            f_y_s.append(f_y.detach().numpy()[0])
            loss_s.append(loss.detach().numpy()[0])
            grad_s.append(grad.detach().numpy()[0][0])
        optimizer.step()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(y_s)), y_s, label='y')
    plt.plot(np.arange(len(f_y_s)), f_y_s, label='f_y')
    plt.plot(np.arange(len(loss_s)), loss_s, label='loss')
    plt.plot(np.arange(len(grad_s)), grad_s, label='grad')
    plt.ylabel(f'y, f_y, loss, grad')
    plt.xlabel('hundred of iterations')
    plt.legend(loc="center right")
    if is_custom_clamp:
        plt.title(f'Custom clamp')
    else:
        plt.title(f'Torch Clamp')

    plt.show()


if __name__ == '__main__':
    # check_torch_clamp_gradient()
    # check_class_Clamp_grad()
    check_DifferentiableClamp_grad()
