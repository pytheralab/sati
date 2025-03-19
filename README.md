# SaT
SaT (Segment Any Text) inference based on Triton Inference Server

## Installation

```bash
conda create -n sat_chunker python==3.10.11
conda activate sat_chunker
conda install nvidia/label/cuda-12.6.1::cuda-toolkit
pip install -r requirements.txt
```

## Convert model to onnx/tensorrt

```bash
# first convert to onnx
sh scripts/export_onnx.sh
# then convert to tensorrt
## init the triton server
sh scripts/run_triton_convert_tensorrt.sh
## convert to tensorrt by copy the script in scripts/export_tensorrt.sh to the triton container
```

## Model weights

The model weights are hosted on [huggingfcae](https://huggingface.co/presencesw/runpod_llm)
Please download the weights and put them in the `models` respectively.

## License
[AGPL v3.0](LICENSE).<br>
Copyright @ 2025 [Pythera](https://github.com/pytheralab/sati). All rights reserved.