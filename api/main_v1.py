from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from trism import TritonModel
from fastapi.middleware.cors import CORSMiddleware

import os
from typing import List
import numpy as np

from transformers import AutoTokenizer

# Parse environment variables
#
chunker_name    = os.getenv("CHUNKER_NAME")
tokenizer_name = os.getenv("TOKENIZER_NAME", "facebookAI/xlm-roberta-base")
model_version = int(os.getenv("MODEL_VERSION", 1))
batch_size    = int(os.getenv("BATCH_SIZE", 1))
#
url           = os.getenv("TRITON_URL", "localhost:8000")
protocol      = os.getenv("PROTOCOL", "HTTP")
verbose       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")
grpc = protocol.lower() == "grpc"

chunker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = TritonModel(
    model=chunker_name,
    version=model_version,
    url=url,
    grpc=grpc,
)

class ListStr(BaseModel):
    texts: List[str]

############
# FastAPI
############

app = FastAPI()

origins = [
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"Hello World": "Welcome to the API"}

@app.post("/chunker_tokenize/")
async def chunker_tokenize(text: str) -> List[dict]:
    # return_value = []
    # for text in texts:
    tokenized_text = chunker_tokenizer(
        text,
        return_offsets_mapping=True,
        verbose=False,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )
    return tokenized_text

# send n docs to the server
@app.post("/chunker/")
async def chunker(textRequest: str) -> JSONResponse:
    text = textRequest
    texts_token = await chunker_tokenize(text) # Tokenizer text type dict
    texts_responses = []
    # all_output =[]
    times = 0
    start = 0
    while start + 512 < len(texts_token['input_ids']):
        input_model = []
        for inp in model.inputs:
            input_model.append(np.array([texts_token[inp.name][start:start + 512]]))
        outputs = model.run(data=input_model)[model.outputs[0].name][0] # -> 1 x bz x length -> bz x length
        outputs =[value_outputs[0] for value_outputs in outputs]
        all_index = [index + 1 for index, value in enumerate(outputs) if value > 0]
        all_index = [512*times + index for index in all_index]
        if len(all_index) > 0:
            texts_responses.extend(all_index)
            start = texts_responses[-1]
        else:
            start += 512
        times += 1

    for inp in model.inputs:
        input_model.append(np.array([texts_token[inp.name][start:]]))
    outputs = model.run(data=input_model)[model.outputs[0].name][0]
    outputs =[value_outputs[0] for value_outputs in outputs]
    all_index = [index + 1 for index, value in enumerate(outputs) if value > 0]
    all_index = [start + index for index in all_index]
    texts_responses.extend(all_index)
    start = 0
    return_value = []
    for index in texts_responses:
        return_value.append(chunker_tokenizer.decode(texts_token['input_ids'][start:index]))
        start = index
    return_value = [value for value in return_value if value != ""]
    return JSONResponse(content=jsonable_encoder(return_value))