import gc

import tensorflow as tf
import torch

from transformers import AutoTokenizer, AutoModel
from enum import Enum
from parameters import *
import numpy as np
from utils_local.nlp_tokenize_and_bow import clean_from_txt_to_bow
import json



class EncodingModel:
    def __init__(self, par: Params):
        self.par = par
        self.framework = par.enc.framework
        if self.framework is None:
            par.enc.framework =Framework.TENSORFLOW
        self.bow = None
        self.model = None
        self.tokenizer = None
        self.init_models(True)

    def init_models(self, force = True):
        if force:
            if self.par.enc.opt_model_type not in [OptModelType.BOW1]:
                if self.framework == Framework.TENSORFLOW:
                    from transformers import GPT2Tokenizer, TFOPTForCausalLM, OPTForCausalLM
                    # self.tokenizer = AutoTokenizer.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR_TORCH)
                    # self.model = AutoModel.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR_TORCH)
                    self.tokenizer = GPT2Tokenizer.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR)
                    self.model = TFOPTForCausalLM.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR_TORCH)
                    self.model = AutoModel.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR_TORCH)
                    # self.tokenizer = GPT2Tokenizer.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR_TORCH)
                    # self.model = OPTForCausalLM.from_pretrained(self.par.enc.opt_model_type.value, cache_dir=Constant.HUGGING_DIR_TORCH)
                self.bow = False
            else:
                self.bow = True
            if self.par.enc.framework == Framework.PYTORCH:
                torch.cuda.empty_cache()
            gc.collect()


    def get_hidden_states(self, text: str):
        # if self.framework == Framework.TENSORFLOW:
        #     inputs = self.tokenizer.encode(text, return_tensors="tf", truncation=True, max_length=self.model.config.max_position_embeddings - 1)
        #     outputs = self.model(inputs, output_hidden_states=True)
        #     hidden_states = outputs.hidden_states
        #     last_hidden_state = hidden_states[-1].numpy()
        # else:
        #     with torch.no_grad():
        #         inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=self.model.config.max_position_embeddings - 1)
        #         outputs = self.model(inputs, output_hidden_states=True)
        #         hidden_states = outputs.hidden_states
        #         last_hidden_state = hidden_states[-1].detach().numpy()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get the embeddings of the last layer
        embeddings = outputs.last_hidden_state
        # You might want to pool these embeddings in some way, e.g., mean pooling
        pooled_embeddings = torch.mean(embeddings, dim=1).detach().numpy()[0,:]
        last_token_hidden_state = embeddings.detach().numpy()[0, -1, :]

        if self.par.enc.framework == Framework.PYTORCH:
            torch.cuda.empty_cache()
        gc.collect()

        return last_token_hidden_state, pooled_embeddings

    def get_hidden_states_para(self, texts: list):
        if self.bow:
            # Create arrays to hold the processed hidden states for all texts
            first_encoding = []
            second_encoding = []

            for i, text in enumerate(texts):
                bow = clean_from_txt_to_bow(text)
                first_encoding.append(json.dumps(bow))
        else:
            if self.framework == Framework.TENSORFLOW:
                inputs = self.tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=self.model.config.max_position_embeddings - 1)
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                last_hidden_states = hidden_states[-1].numpy()
            else:
                with torch.no_grad():
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.model.config.max_position_embeddings - 1)
                    outputs = self.model(inputs.input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    last_hidden_states = hidden_states[-1].detach().numpy()
                    del outputs, inputs
                    torch.cuda.empty_cache()
                    gc.collect()

            # Create arrays to hold the processed hidden states for all texts
            first_encoding = []
            second_encoding = []

            for i, text in enumerate(texts):
                last_token_hidden_state = last_hidden_states[i, -1, :]
                average_hidden_state = np.mean(last_hidden_states[i], axis=0)

                first_encoding.append(last_token_hidden_state)
                second_encoding.append(average_hidden_state)

                # Clear memory for PyTorch
                if self.framework == Framework.PYTORCH:
                    del last_token_hidden_state
                    del average_hidden_state
                    torch.cuda.empty_cache()
                    gc.collect()

            # Additional memory clearance after processing all texts
            if self.framework == Framework.PYTORCH:
                torch.cuda.empty_cache()

        return first_encoding, second_encoding
        # return [np.array(1) for x in range(len(first_encoding))],[np.array(1) for x in range(len(second_encoding))]
        # return ['a'],['b']

    def generate_text(self, text: str, max_length: int = 50):
        if self.framework == Framework.TENSORFLOW:
            inputs = self.tokenizer.encode(text, return_tensors="tf")
        else:
            inputs = self.tokenizer.encode(text, return_tensors="pt")

        output = self.model.generate(inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(output[0])
        return generated_text


if __name__ == "__main__":
    par = Params()
    par.enc.framework = Framework.PYTORCH
    par.enc.opt_model_type = OptModelType.OPT_125m
    model = EncodingModel(par)
    input_text = "Hello, World!"
    # last_token_hidden_stage_v1, average_token_hidden_v2 =model.get_hidden_states(input_text)
    # print("Last hidden state: ", last_token_hidden_stage_v1)
    # print("Generated text: ", model.generate_text(input_text, max_length=50))

    input_texts = ["Hello, World!", "How are you?", "Nice to meet you"]
    last_token_hidden_states, average_hidden_states = model.get_hidden_states_para(input_texts)
    for i, text in enumerate(input_texts):
        print(f"Text: {text}")
        print("Last token hidden state:", last_token_hidden_states[i])
        last_token_hidden_stage_v1, average_token_hidden_v2 = model.get_hidden_states(text)

        print('Max diff', np.max(np.abs(last_token_hidden_stage_v1-last_token_hidden_states[i])))
        print("Average hidden state:", average_hidden_states[i])



