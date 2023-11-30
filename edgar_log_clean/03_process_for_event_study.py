import didipack as didi

from data import Data
from parameters import *

if __name__ == '__main__':
    args = didi.parse()
    par = Params()
    data = Data(par)
    log = data.load_logs_tot()
    t = log[['accession']].drop_duplicates()
    t['form_id'] = t['accession'].apply(lambda x: str(x).replace('-',''))
    log = log.merge(t).drop(columns='accession')
    ati = data.load_icf_ati_filter()

    ati = ati.rename(columns={'date':'form_date'})
    log = log.merge(ati[['form_id','form_date']])
    log['dist'] = (log['date']-log['form_date']).dt.days
    ind = log['dist'].between(0,40)
    print(ind.mean())
    t = log.groupby(['form_id','after'])['ip'].sum().reset_index().pivot(columns='after',index='form_id',values='ip')
    t['ratio'] = t[True]/t.sum(1)
    print(t.mean())

    max_dist = 40
    log.loc[log['dist']>max_dist,'dist'] = max_dist+1
    temp = log.groupby(['form_date','form_id','dist'])['ip'].sum().reset_index()
    temp.memory_usage(deep=True).sum() / (1024**3)
    temp.to_pickle(data.p_dir+'log_ev_study.p')





