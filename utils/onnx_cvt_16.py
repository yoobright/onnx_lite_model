import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16

model_name = "mobilenetv2-1.0"
input_onnx_model = '{}.onnx'.format(model_name)
output_onnx_model = '{}_fp16.onnx'.format(model_name)

onnx_model = onnxmltools.utils.load_model(input_onnx_model)
onnx_model = convert_float_to_float16(onnx_model)
onnxmltools.utils.save_model(onnx_model, output_onnx_model)

