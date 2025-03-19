export CHUNKER_NAME="sat_chunker"

uvicorn api.main_v1:app --host 0.0.0.0 --port 7999