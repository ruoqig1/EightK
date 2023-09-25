import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *

def df_size(df):
    memory_in_gb = df.memory_usage(deep=True).sum() / 1e9
    return memory_in_gb

if __name__ == "__main__":
    p = 'res/list_usage2/df_0.p'
    df = pd.read_pickle(p)
    df