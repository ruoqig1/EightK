from parameters import *
from data import Data
import os

if __name__ == '__main__':
    par = Params()
    data = Data(par)
    os.makedirs(Constant.DRAFT_1_CSV_PATH,exist_ok=True)

    # List of function names
    functions = ['load_ravenpack_all', 'load_some_relevance_icf', 'load_ml_forecast_draft_1', 'load_mkt_cap_yearly', 'load_snp_const']
    functions = ['load_some_relevance_icf', 'load_ml_forecast_draft_1', 'load_mkt_cap_yearly', 'load_snp_const']

    # for func in functions:
    #     # Call each function on the data object
    #     df = getattr(data, func)()
    #     # Check if the dataframe is not empty or None
    #     if df is not None:
    #         # Save the dataframe to a CSV file
    #         csv_file_name = f"{Constant.DRAFT_1_CSV_PATH}{func}.csv"
    #         if type(df)==type(()):
    #             df = df[0]
    #         df.to_csv(csv_file_name, index=False)
    #         print(f"Saved {csv_file_name}")
    #     else:
    #         print(f"No data returned from {func}")

    for i in [-1,1,3,6]:
        df = data.load_abn_return(model=i)
        csv_file_name = f"{Constant.DRAFT_1_CSV_PATH}load_abn_return{i}.csv"
        df.to_csv(csv_file_name, index=False)
        print(f"Saved {csv_file_name}")