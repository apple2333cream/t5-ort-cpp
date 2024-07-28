import onnx 
import onnxruntime 
from onnxruntime import InferenceSession
from transformers import T5Tokenizer
import torch
import numpy as np
import os 
pretrained_model = '/home/wzp/t5-base' 
output_model_file="t5_base_beam_search.onnx"
tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
text = "translate English to French: I was a victim of a series of accidents."
# 使用tokenizer对文本进行编码
input_ids = tokenizer.encode(text, return_tensors='pt')  
# input_ids=torch.ones(1,18)
input_ids=input_ids.to(torch.int32)
inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int32),
        "max_length": np.array([100], dtype=np.int32),
        "min_length": np.array([1], dtype=np.int32),
        "repetition_penalty": np.array([1.0], dtype=np.float32),
        "num_beams": np.array([1], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
    }


ort_session = InferenceSession(output_model_file)
result = ort_session.run(None, inputs)
sequences = result[0]
print("sequences", sequences)

decoded_sequence = tokenizer.decode(sequences[0][0], skip_special_tokens=True)
print("decoded_sequence",decoded_sequence)