import torch


class DDQN(torch.nn.Module):
    """
    A Double Deep Q-Network (DDQN) implementation using PyTorch.

    This model uses convolutional layers to process input states and fully connected layers
    to output Q-values for each action in the action space.
    """

    def __init__(self, output_dim):
        """
        Initialize the DDQN model.
        """
        super().__init__()
        self.cnn_1 = torch.nn.Conv2d(1, 32, kernel_size=4)
        self.relu_1 = torch.nn.ReLU()
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.cnn_2 = torch.nn.Conv2d(32, 64, kernel_size=2)
        self.relu_2 = torch.nn.ReLU()
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.linear_1 = torch.nn.Linear(192, 64)
        self.relu_3 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, output_dim)

    def forward(self, x1):
        """
        Forward pass of the DDQN model.

        Args:
            x1 (torch.Tensor): The input tensor representing the game state.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = self.cnn_1(x1)
        x = self.relu_1(x)
        x = self.max_pool_1(x)
        x = self.cnn_2(x)
        x = self.relu_2(x)
        x = self.max_pool_2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.relu_3(x)
        x = self.linear_2(x)
        return x


class DuelingDQN(torch.nn.Module):
    """
    A Dueling Deep Q-Network (Dueling DQN) implementation using PyTorch.

    This model separates the estimation of state-value (V) and advantage (A) to improve
    learning stability and performance in reinforcement learning tasks.
    """

    def __init__(self, output_dim):
        """
        Initialize the Dueling DQN model.
        """
        super().__init__()
        self.cnn_1 = torch.nn.Conv2d(1, 32, kernel_size=4)
        self.relu_1 = torch.nn.ReLU()
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.cnn_2 = torch.nn.Conv2d(32, 64, kernel_size=2)
        self.relu_2 = torch.nn.ReLU()
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.linear_v_1 = torch.nn.Linear(192, 64)
        self.relu_v_1 = torch.nn.ReLU()
        self.linear_v_2 = torch.nn.Linear(64, 1)
        self.linear_a_1 = torch.nn.Linear(192, 64)
        self.relu_a_1 = torch.nn.ReLU()
        self.linear_a_2 = torch.nn.Linear(64, output_dim)

    def forward(self, x1):
        """
        Forward pass of the Dueling DQN model.

        Args:
            x1 (torch.Tensor): The input tensor representing the game state.

        Returns:
            torch.Tensor: The Q-values for each action, computed as Q(s, a) = V(s) + (A(s, a) - mean(A)).
        """
        x = self.cnn_1(x1)
        x = self.relu_1(x)
        x = self.max_pool_1(x)
        x = self.cnn_2(x)
        x = self.relu_2(x)
        x = self.max_pool_2(x)
        x = x.view(x.size(0), -1)

        v = self.linear_v_1(x)
        v = self.relu_v_1(v)
        v = self.linear_v_2(v)

        a = self.linear_a_1(x)
        a = self.relu_a_1(a)
        a = self.linear_a_2(a)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q
