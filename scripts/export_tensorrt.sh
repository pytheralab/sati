trtexec \
  --onnx=/onnx/model.onnx \
  --builderOptimizationLevel=4 \
  --saveEngine=/onnx/model_fp32.engine \
  --minShapes=input_ids:1x1,attention_mask:1x1 \
  --optShapes=input_ids:1x256,attention_mask:1x256 \
  --maxShapes=input_ids:1x512,attention_mask:1x512