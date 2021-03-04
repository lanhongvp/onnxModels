import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import onnx
import onnxruntime
import onnxruntime.tools.symbolic_shape_infer as sym

class LoopMinistModel(nn.Module):
    def __init__(self):
        super(LoopMinistModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, img_tensor):
        item_count = img_tensor.size(0)
        out = torch.zeros((1, 16, img_tensor.size(2), img_tensor.size(3)))

        for i in range(item_count):
            x = img_tensor[i].unsqueeze(0)
            x = self.layer1(x)
            out = out + x

        return out 


if __name__ == '__main__':
    # Module instantiation:
    mnistLoop = LoopMinistModel()
    mnistLoop.eval()

    # Example inputs for torch.onnx.export:
    img_tensor = torch.zeros([10,1,28,28], dtype = torch.float32)
    img_tensor_numpy = np.random.random([10,1,28,28]).astype(np.float32)

    # Jit script for torch model
    mnistLoopTorchJit = torch.jit.script(mnistLoop)
    outputFromScripted = mnistLoopTorchJit(img_tensor)

    print("exporting to ONNX...\n")

    torch.onnx.export(model = mnistLoopTorchJit,
                    args= img_tensor,
                    f = "../../onnxTestModels/loopWithConv.onnx",
                    example_outputs=[outputFromScripted],
                    input_names = ["input_img"],
                    output_names = ["output"],
                    dynamic_axes = {"input_img": [2, 3]})

    sess = onnxruntime.InferenceSession('../../onnxTestModels/loopWithConv.onnx')   
    print('Load model')

    result = sess.run(None, {'input_img': img_tensor_numpy})
    model = onnx.load('../../onnxTestModels/loopWithConv.onnx')

    # check model
    onnx.checker.check_model(model)
    print("onnx checker done")

    # override input shapes
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 48
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 48
    onnx.save(model, '../../onnxTestModels/loopWithConv_override.onnx')

    # using ONNX shape inference    
    inferModel = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferModel, '../../onnxTestModels/loopWithConv_override_inferred.onnx')
    
    # using symbolic shape inference
    out_mp = sym.SymbolicShapeInference.infer_shapes(in_mp=model, auto_merge=True)
    onnx.save(out_mp, '../../onnxTestModels/loopWithConv_override_symbolic.onnx')

