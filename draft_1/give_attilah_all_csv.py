from parameters import *
from data import Data
import os

if __name__ == '__main__':
    par = Params()
    data = Data(par)
    os.makedirs(Constant.DRAFT_1_CSV_PATH,exist_ok=True)
    ati = data.load_icf_ati_filter(False,False)

    '''
    news0: binary equal 1 if there is a news on the day of the 8k
    news0_f: binary equal 1 if there is a news the day AFTER 8k is published
    rtime: short for ravenpack time. Is the time (hours, minute, seconds) at which the news0 was published.
    rtime_f: short for ravenpack time. Is the time (hours, minute, seconds) at which the news0_f was published.
    atime: short for accessionTime, time at which the 8k for mwas put up.
    '''


    csv_file_name = f"{Constant.DRAFT_1_CSV_PATH}new_coverage_def.csv"
    ati.to_csv(csv_file_name, index=False)
    print(f"Saved {csv_file_name}")

    # List of function names
    functions = ['load_ravenpack_all', 'load_some_relevance_icf', 'load_ml_forecast_draft_1', 'load_mkt_cap_yearly', 'load_snp_const']
    functions = ['load_logs_ev_study']
    # df = data.load_logs_ev_study()

    for func in functions:
        # Call each function on the data object
        df = getattr(data, func)()
        # Check if the dataframe is not empty or None
        if df is not None:
            # Save the dataframe to a CSV file
            csv_file_name = f"{Constant.DRAFT_1_CSV_PATH}{func}.csv"
            df.to_csv(csv_file_name, index=False)
            print(f"Saved {csv_file_name}")
        else:
            print(f"No data returned from {func}")

    for i in [-1,1,3,6,7]:
        df = data.load_abn_return(model=i)
        data = Data(par)
        per = data.load_some_relevance_icf()
        per['date'] = pd.to_datetime(per['adate'])
        per = per.dropna(subset='date')
        per = per.loc[per['date'].dt.year >= 2004, :]
        per['permno'] = per['permno'].astype(int)
        per = per[['date', 'permno', 'form_id']].dropna().drop_duplicates()
        df = df.merge(per,how='left')
        csv_file_name = f"{Constant.DRAFT_1_CSV_PATH}load_abn_return{i}.csv"
        df.to_csv(csv_file_name, index=False)
        print(f"Saved {csv_file_name}")