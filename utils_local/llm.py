import tensorflow as tf
from transformers import GPT2Tokenizer, TFOPTForCausalLM
from enum import Enum
from parameters import *
import numpy as np
from utils_local.nlp_tokenize_and_bow import clean_from_txt_to_bow
import json

class EncodingModel:
    def __init__(self, par : Params):
        if par.enc.opt_model_type not in [OptModelType.BOW1]:
            self.tokenizer = GPT2Tokenizer.from_pretrained(par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR)
            self.model = TFOPTForCausalLM.from_pretrained(par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR)
            self.bow = False
        else:
            self.bow = True

    def get_hidden_states(self, text: str):
        inputs = self.tokenizer.encode(text, return_tensors="tf",truncation=True, max_length = self.model.config.max_position_embeddings-1)
        outputs = self.model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1].numpy()
        last_token_hidden_state = last_hidden_state[0, -1, :]
        average_hidden_stage = np.mean(last_hidden_state,1)
        return last_token_hidden_state, average_hidden_stage[0,:]

    def get_hidden_states_para(self, texts: list):
        if self.bow:
            # Create arrays to hold the processed hidden states for all texts
            first_encoding = []
            second_encoding = []

            for i, text in enumerate(texts):
                bow = clean_from_txt_to_bow(text)
                first_encoding.append(json.dumps(bow))
        else:
            inputs = self.tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=self.model.config.max_position_embeddings - 1)
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_states = hidden_states[-1].numpy()

            # Create arrays to hold the processed hidden states for all texts
            first_encoding = []
            second_encoding = []

            for i, text in enumerate(texts):
                last_token_hidden_state = last_hidden_states[i, -1, :]
                average_hidden_state = np.mean(last_hidden_states[i], axis=0)

                first_encoding.append(last_token_hidden_state)
                second_encoding.append(average_hidden_state)

        return first_encoding, second_encoding

    def generate_text(self, text: str, max_length: int = 50):
        inputs = self.tokenizer.encode(text, return_tensors="tf")
        output = self.model.generate(inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(output[0])
        return generated_text


if __name__ == "__main__":
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_125m
    model = EncodingModel(par)
    input_text = "Hello, World!"
    last_token_hidden_stage_v1, average_token_hidden_v2 =model.get_hidden_states(input_text)
    print("Last hidden state: ", last_token_hidden_stage_v1)
    print("Generated text: ", model.generate_text(input_text, max_length=50))

    input_texts = ["Hello, World!", "How are you?", "Nice to meet you"]
    last_token_hidden_states, average_hidden_states = model.get_hidden_states_para(input_texts)
    for i, text in enumerate(input_texts):
        print(f"Text: {text}")
        print("Last token hidden state:", last_token_hidden_states[i])
        print("Average hidden state:", average_hidden_states[i])
        print("Generated text:", model.generate_text(text, max_length=50))

