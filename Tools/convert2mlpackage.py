import torch
import coremltools as ct
import torch.nn as nn
from Src.Genders_net import Net_1, Net_2, Net_3,Net_4,Net_5


loaded_dict = torch.load('../Models/gender_model_2.pth')
state_dict = loaded_dict['weights']

# Załadowanie modelu
model = Net_2()
model.load_state_dict(state_dict)
model.eval()

# Tworzenie przykładowego wejścia
example_input = torch.rand(1, 3, 100, 100)

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
