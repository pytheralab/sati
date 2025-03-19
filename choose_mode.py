import os

# get environment variable
mode = os.environ.get('MODEL_TYPE', 'model_w_mask')
backend = os.environ.get('BACKEND', 'onnxruntime')
if backend == 'tensorrt':
    platform = 'tensorrt_plan'
elif backend == 'onnxruntime':
    platform = 'onnxruntime_onnx'
else:
    raise ValueError(f"Backend {backend} not supported")

config_trt_path = os.path.join("/", "models", "sat_chunker", "config.pbtxt")

info_v1 = f"""name: "sat_chunker"
platform: "{platform}"
backend: "{backend}"
default_model_filename: "{mode}"
max_batch_size: 0
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_FP16
    dims: [ -1, -1 ]
  }}
]
output [
  {{
    name: "logits"
    data_type: TYPE_FP16
    dims: [ -1, -1, 1 ]
  }}
]"""
info_v2 = f"""name: "sat_chunker"
platform: "{platform}"
backend: "{backend}"
default_model_filename: "{mode}"
max_batch_size: 0"""

with open(config_trt_path, "w") as f:
    f.write(info_v1)
    
print(f"Write config to {config_trt_path}")