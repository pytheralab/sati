docker run --gpus all -it --rm \
    -v /home/presencecesw/project/Sat_chunker/models/sat-12l-sm:/onnx nvcr.io/nvidia/tensorrt:24.12-py3 bash