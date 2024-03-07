import torch
from Src.Gender_net import Net
import coremltools as ct
''''''''''''''''''''''''''

loaded_dict = torch.load('../Models/gender_model.pth')
state_dict = loaded_dict['weights']


model = Net()
model.load_state_dict(state_dict)
model.eval()


example_input = torch.rand(1, 3, 52, 52)
traced_model = torch.jit.trace(model, example_input)


mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
)

mlmodel.save('../Models/gender_model.mlpackage')