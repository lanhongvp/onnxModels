import torch
from torch import nn

# Module creation:
class IterativelyModifyTensor(nn.Module):
    """
    Creates a module that takes a 2D tensor/matrix (input_2d_tensor),
    and replaces each row of it with a 1D tensor (substitution_tensor)
    with a for loop.
    
    This is just for demonstration purposes; to see if we can iteratively
    modify matrix rows in Pytorch.
    """
    def __init__(self):
        super(IterativelyModifyTensor, self).__init__()
        
    def forward(self, input_2d_tensor, substitution_tensor):
        output_tensor = (torch
                         .zeros(1)
                         .repeat(input_2d_tensor.size(0) * input_2d_tensor.size(1))
                         .view(input_2d_tensor.size(0), input_2d_tensor.size(1))
                         .long()
                        )
        num_rows = input_2d_tensor.size(0)
        
        for row in range(num_rows):
            dim = 0
            index = torch.ones(1).repeat(input_2d_tensor.size(1)).unsqueeze(0).long() * row
            src = substitution_tensor.unsqueeze(0)
            output_tensor = output_tensor.scatter(dim, index, src)
        
        return output_tensor
    
# Module instantiation:
my_modifier = IterativelyModifyTensor()

# Generate inputs:
input_2d_tensor = torch.zeros(4, 3).long()
substitution_tensor = torch.arange(3)

print("input_2d_tensor:\n%s\n" % input_2d_tensor)
print("substitution_tensor:\n%s\n" % substitution_tensor)

# Call forward:
output = my_modifier(input_2d_tensor, substitution_tensor)

print("output:\n%s\n" % output)

# Scripting the module:
print("scripting the module...\n")

my_modifier_scripted = torch.jit.script(my_modifier)

# See if output from scripted module is correct (it is):
output_from_scripted = my_modifier_scripted(input_2d_tensor, substitution_tensor)

print("output_from_scripted:\n%s\n" % output_from_scripted)

# Exporting the scripted module to ONNX:
print("exporting to ONNX...\n")

torch.onnx.export(model=my_modifier_scripted,
                  args=(input_2d_tensor, substitution_tensor),
                  f="../../onnxTestModels/modifier.onnx",
                  example_outputs=[output_from_scripted],
                  input_names=["input_2d_tensor", "substitution_tensor"],
                  output_names=["output"])

# Loading the ONNX model with Microsoft's onnxruntime:
print("loading exported ONNX model with onnxruntime...\n")

import onnx
model = onnx.load('../../onnxTestModels/modifier.onnx')

onnx.checker.check_model(model)
print("onnx model check succeed")

import onnxruntime as ort

ort_session = ort.InferenceSession('../../onnxTestModels/modifier.onnx')

# See if output onnxruntime is correct (it is):
output_onnx = ort_session.run(None, 
                              {"input_2d_tensor": input_2d_tensor.numpy(),
                               "substitution_tensor": substitution_tensor.numpy()
                              })

print("output_onnx:\n%s\n" % output_onnx[0])