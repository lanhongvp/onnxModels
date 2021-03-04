import onnx
import onnxruntime
import onnx.helper as helper
import numpy as np

from onnx import TensorProto, numpy_helper, shape_inference
from onnxruntime.backend import prepare

# Initializer setting
data_type = [20000, 200]
data_name = 'initialData'
data = np.random.rand(*data_type).astype(np.float32)
dataIntializer = numpy_helper.from_array(data, name=data_name)

idx_name = 'consantIndices'
idx = np.array((0), dtype=np.int64)
idxIntializer = numpy_helper.from_array(idx, name=idx_name)

concat_name0 = 'constantConcat0Input0'
concatInput0 = np.array([2], dtype=np.int64)
concatInput0Intializer = numpy_helper.from_array(concatInput0, name=concat_name0)

concat_name1 = 'constantConcat0Input1'
concatInput1 = np.array([128], dtype=np.int64)
concatInput1Intializer = numpy_helper.from_array(concatInput1, name=concat_name1)

w_type = [2, 512, 200]
w_name = 'W'
w_data = np.random.rand(*w_type).astype(np.float32)
wIntializer = numpy_helper.from_array(w_data, name=w_name)

r_type = [2, 512, 128]
r_name = 'R'
r_data = np.random.rand(*r_type).astype(np.float32)
rIntializer = numpy_helper.from_array(r_data, name=r_name)

b_type = [2, 1024]
b_name = 'B'
b_data = np.random.rand(*b_type).astype(np.float32)
bIntializer = numpy_helper.from_array(b_data, name=b_name)

# make inner body graph
body = helper.make_graph(
        [
            helper.make_node('Gather', inputs=['initialData', 'query'], outputs=['gatherOutput0']),
            helper.make_node('Shape', inputs=['gatherOutput0'], outputs=['gatherOutput0Shape']),
            helper.make_node('Gather', inputs=['gatherOutput0Shape', 'consantIndices'], outputs=['gatherOutput1'], axis=0),
            helper.make_node('Unsqueeze', inputs=['gatherOutput1'], outputs=['unsqueezeOutput0'], axes=[0]),
            helper.make_node('Concat', inputs=['constantConcat0Input0', 'unsqueezeOutput0', 'constantConcat0Input1'], outputs=['concatOutput0'], axis=0),
            helper.make_node('ConstantOfShape', inputs=['concatOutput0'], outputs=['consantShpaeOfConcatOutput0']),
            helper.make_node('Transpose', inputs=['gatherOutput0'], outputs=['transposeGatherOutput0'], perm=[1,0,2]),
            helper.make_node('LSTM', inputs=['transposeGatherOutput0', 'W', 'R', 'B', '', 'consantShpaeOfConcatOutput0', 'consantShpaeOfConcatOutput0'],
                                        outputs=['LSTM_Y', 'LSTM_Yh', 'LSTM_Yc'], hidden_size=128, direction='bidirectional'),
            helper.make_node('Identity', inputs=['keepGoingInput'], outputs=['keepGoingOutput'])
        ],
        'loopbody',
        [
            helper.make_tensor_value_info('iterationNum', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('keepGoingInput', TensorProto.BOOL, [1]),
        ],
        [
            helper.make_tensor_value_info('keepGoingOutput', TensorProto.BOOL, [1]),
            helper.make_tensor_value_info('LSTM_Y', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('LSTM_Yh', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('LSTM_Yc', TensorProto.FLOAT, [])
        ],
        [dataIntializer, idxIntializer, concatInput0Intializer, concatInput1Intializer, wIntializer, rIntializer, bIntializer])

# make loop node
graph_def = helper.make_graph(
        [
            helper.make_node('Loop', ['max_trip_count','keep_going_in'],
                                ['LSTM_Y_outer_output', 'LSTM_Yh_outer_output', 'LSTM_Yc_outer_output'], body=body)
        ],
        'outerGraph',
        [
            helper.make_tensor_value_info('keep_going_in', TensorProto.BOOL, [1]),
            helper.make_tensor_value_info('max_trip_count', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('query', TensorProto.INT64, ['batch_sz', 'max_sent_len'])
        ],
        [
            helper.make_tensor_value_info('LSTM_Y_outer_output', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('LSTM_Yh_outer_output', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('LSTM_Yc_outer_output', TensorProto.FLOAT, [])
        ])

# make loop model
loopModelLstm = helper.make_model(graph_def)
loopModelLstm.opset_import[0].version = 11

# check model
onnx.checker.check_model(loopModelLstm)
print("onnx model checked")

# save onnx model
onnx.save(loopModelLstm, '../../onnxTestModels/loopWithLstm.onnx')
sess = onnxruntime.InferenceSession('../../onnxTestModels/loopWithLstm.onnx')

# input feed value
keep_going = np.array([True], dtype=np.bool)
max_trip_count = np.array([2], dtype=np.int64)
batch_sz = 1
max_sent_len = 50
query_data = np.zeros((batch_sz, max_sent_len), dtype=np.int64)

# onnx runtime 
result = sess.run(None, {'max_trip_count': max_trip_count,
                         'keep_going_in': keep_going,
                        'query': query_data})

print("onnx runtime done")

