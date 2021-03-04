import torch
import onnx
import onnxruntime.tools.symbolic_shape_infer as sym
import numpy as np

@torch.jit.script
def loop(x, y):
    for i in range(int(y)):
        x = x + i
    return x

class LoopModel2(torch.nn.Module):
    def forward(self, x, y):
        return loop(x, y)

if __name__ == '__main__':
    model = LoopModel2()
    torch.onnx.export(model, (torch.tensor([[]]), torch.tensor([10], dtype=torch.long)), 'loop.onnx', verbose=True,
                    input_names = ['input_data', 'loop_range'],
                    output_names = ['result'],
                    dynamic_axes = { 'input_data': [0, 1] })
    model = onnx.load('loop.onnx')
    
    # override input shapes
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 3
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 5
    onnx.save(model, '../../onnxTestModels/loop_override.onnx')

    # using ONNX shape inference    
    inferModel = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferModel, '../../onnxTestModels/loop_override_inferred.onnx')
    
    # using symbolic shape inference
    out_mp = sym.SymbolicShapeInference.infer_shapes(in_mp=model, auto_merge=True)
    onnx.save(out_mp, '../../onnxTestModels/loop_override_symbolic.onnx')
