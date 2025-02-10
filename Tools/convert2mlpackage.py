import torch
import coremltools as ct
import torch.nn as nn
from Src.Genders_net import Net_1, Net_2, Net_3,Net_4,Net_5


loaded_dict = torch.load('../Models/gender_model_2.pth')
state_dict = loaded_dict['weights']

model = Net_2()
model.load_state_dict(state_dict)
model.eval()

example_input = torch.rand(1, 3, 100, 100)

traced_model = torch.jit.trace(model, example_input)

model_coreml = ct.convert(
  traced_model,
  convert_to="mlprogram",
  inputs=[ct.TensorType(shape=example_input.shape)],

)

model_coreml.save('../Models/gender_model.mlpackage')
