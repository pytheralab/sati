docker run --gpus=all -it --rm \
    --name sat_chunker \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    sat_chunker:24.12 