docker run --gpus=all -it --rm \
    --name sat_client \
    -e TRITON_URL="localhost:8000" \
    -p7000:7000 \
    --network "host" \
    sat_client