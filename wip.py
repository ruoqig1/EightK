from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    par = Params()
    data = Data(par)

    # do the matching
    rav = data.load_ravenpack_chunk()
    ev = data.load_icf_current()
    ev =ev.dropna(subset=['adate'])
    ev[['adate','permno','atime','']]

    rav = rav.dropna(subset=['rdate'])

    rav= rav[['rdate','rtime','permno','relevance','event_sentiment_score']].rename(columns={'rdate':'adate'})

