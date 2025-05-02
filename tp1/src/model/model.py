import torch


class DDQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_dim, 128)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(128, 128)
        self.relu_2 = torch.nn.ReLU()
        self.output = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.output(x)
        return x


class DuelingDQN(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_dim, 128)
        self.relu_1 = torch.nn.ReLU()
        self.linear_v_1 = torch.nn.Linear(128, 64)
        self.relu_v_1 = torch.nn.ReLU()
        self.linear_v_2 = torch.nn.Linear(64, 1)
        self.linear_a_1 = torch.nn.Linear(128, 64)
        self.relu_a_1 = torch.nn.ReLU()
        self.linear_a_2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)

        v = self.linear_v_1(x)
        v = self.relu_v_1(v)
        v = self.linear_v_2(v)

        a = self.linear_a_1(x)
        a = self.relu_a_1(a)
        a = self.linear_a_2(a)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q
