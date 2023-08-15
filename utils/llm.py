import tensorflow as tf
from transformers import GPT2Tokenizer, TFOPTForCausalLM
from enum import Enum
from parameters import *
import numpy as np


    # Add other models here as they become available

class OPTModel:
    def __init__(self, par : Params):
        self.tokenizer = GPT2Tokenizer.from_pretrained(par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR)
        self.model = TFOPTForCausalLM.from_pretrained(par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR)

    def get_hidden_states(self, text: str):
        inputs = self.tokenizer.encode(text, return_tensors="tf",truncation=True, max_length = self.model.config.max_position_embeddings-1)
        outputs = self.model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1].numpy()
        last_token_hidden_state = last_hidden_state[0, -1, :]
        average_hidden_stage = np.mean(last_hidden_state,1)
        return last_token_hidden_state, average_hidden_stage[0,:]

    def generate_text(self, text: str, max_length: int = 50):
        inputs = self.tokenizer.encode(text, return_tensors="tf")
        output = self.model.generate(inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(output[0])
        return generated_text


if __name__ == "__main__":
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_125m
    model = OPTModel(par)
    input_text = "Hello, World!"
    last_token_hidden_stage, average_token_hidden =model.get_hidden_states(input_text)
    print("Last hidden state: ", last_token_hidden_stage)
    print("Generated text: ", model.generate_text(input_text, max_length=50))
