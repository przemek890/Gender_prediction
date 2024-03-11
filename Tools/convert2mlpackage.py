import torch
import coremltools as ct
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,dropout_prob=0.5):
        super(Net, self).__init__()

        kernel_s = 3

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_s, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25*25*32, 256)

        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = x.reshape(-1, 25*25*32)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


loaded_dict = torch.load('../Models/gender_model.pth')
state_dict = loaded_dict['weights']

# Załadowanie modelu
model = Net()
model.load_state_dict(state_dict)
model.eval()

# Tworzenie przykładowego wejścia
example_input = torch.rand(1, 3, 52, 52)

# Generowanie wersji TorchScript
traced_model = torch.jit.trace(model, example_input)

# Konwersja do Core ML
model_coreml = ct.convert(
  traced_model,
  convert_to="mlprogram",
  inputs=[ct.TensorType(shape=example_input.shape)],

)

# Zapisz model Core ML
model_coreml.save('../Models/gender_model.mlpackage')
