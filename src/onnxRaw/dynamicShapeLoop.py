import onnx
import onnxruntime
import onnx.helper as helper
import numpy as np
import onnxruntime.tools.symbolic_shape_infer as sym

from onnx import TensorProto, shape_inference
from onnxruntime.backend import prepare

# a = 3
# b = 6
# keep_going = True
# max_trip_count = 10
# user_defined_vals = []
# for i in range(max_trip_count):
#     my_local = a + b
#     b = a - b
#     user_defined_vals.append(b + b)
# b_out = b
# keep_going_out = keep_going
# dynamic_input = 1
# dynamic_input_axis = 2

body = helper.make_graph(
        [
            helper.make_node('Add', ['a_in', 'b_in'], ['my_local'], 'inner_add'),
            helper.make_node('Sub', ['a_in', 'b_in'], ['a_sub_b_in'], 'inner_sub'),
            helper.make_node('Identity',['a_in'], ['a_out'], 'inner_identity'),
            helper.make_node('Add', ['a_sub_b_in', 'a_sub_b_in'], ['user_defined_vals'], 'inner_add2'),
            helper.make_node('Identity',['keep_going_inp'], ['keep_going_inp_2'], 'inner_identity2'),
        ],
        'loopbody',
        [
            helper.make_tensor_value_info('iteration_num', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('keep_going_inp', TensorProto.BOOL, [1]),
            helper.make_tensor_value_info('b_in', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('a_in', TensorProto.FLOAT, [])
        ],
        [
            helper.make_tensor_value_info('keep_going_inp_2', TensorProto.BOOL, [1]),
            helper.make_tensor_value_info('a_sub_b_in', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('a_out', TensorProto.FLOAT, []),
        ])

# Create the outer network
graph_proto = helper.make_graph(
[
    helper.make_node('Loop', ['max_trip_count','keep_going_in','b', 'a'],
                             ['b_loop', 'a_out'], body=body)
],
'outer',
[
    helper.make_tensor_value_info('a', TensorProto.FLOAT, ['']),
    helper.make_tensor_value_info('b', TensorProto.FLOAT, ['']),
    helper.make_tensor_value_info('keep_going_in', TensorProto.BOOL, [1]),
    helper.make_tensor_value_info('max_trip_count', TensorProto.INT64, [1]),
],
[
    helper.make_tensor_value_info('b_loop', TensorProto.FLOAT, []),
    helper.make_tensor_value_info('a_out', TensorProto.FLOAT, []),    
])

model = helper.make_model(graph_proto)
onnx.checker.check_model(model)
print("model checked")

onnx.save(model, '../../onnxTestModels/dynamicLoopModelv1.onnx')

# override input shapes
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.input[1].type.tensor_type.shape.dim[0].dim_value = 1

onnx.save(model, '../../onnxTestModels/dynamic_loop_override.onnx')

# using ONNX shape inference    
inferModel = onnx.shape_inference.infer_shapes(model)
onnx.save(inferModel, '../../onnxTestModels/dynamic_loop_override_inferred.onnx')

# using symbolic shape inference
out_mp = sym.SymbolicShapeInference.infer_shapes(in_mp=model, auto_merge=True)
onnx.save(out_mp, '../../onnxTestModels/dynamic_loop_override_symbolic.onnx')