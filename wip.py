import didipack as didi
from matplotlib import pyplot as plt
from data import Data
from parameters import *

if __name__ == '__main__':
    args = didi.parse()
    par = Params()
    save_dir = Constant.EMB_PAPER
    data = Data(par)
    df = data.load_logs_ev_study()

    f ='/Users/adidisheim/Dropbox/Melbourne/research/EightK/data/refinitiv/News/RTRS/Monthly/2006/JSON/'




