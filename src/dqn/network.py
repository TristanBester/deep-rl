import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size, checkpoint_path):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.checkpoint_path = checkpoint_path

        self.fc_1 = nn.Linear(input_size, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        # Don't apply activation function to state-action values
        x = self.fc_3(x)
        return x

    def serialise(self, checkpoint_name):
        file_name = f"{self.checkpoint_path}/{checkpoint_name}.pt"
        print(f"Saving model parameters to file {file_name} ...")
        torch.save(self.state_dict(), file_name)

    def deserialise(self, checkpoint_file):
        print(f"Loading model parameters from file {checkpoint_file} ...")
        self.load_state_dict(torch.load(checkpoint_file))
