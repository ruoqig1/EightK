# TODO 
TODO:

## ON ML:
- FInish checking current experiment
- Re run vectorisation with the mean (and tf vectors)
- Re run the replciaiton

## Explorations
- on 8k check verbosity on each items and press release
- plot time series across time

## match with earnings
- build earning suprise and see if this relates to number of 8k and number of words in expectations. 



# Code structure:

## Data
* Raw data (csv and json) is stored on an external hard drive that attached to a VM. 
* I transformed it into pickles and push that on spratan with clean/news_01_refinitive_merge.py
* Same stuff with third party news in clean/news_03_third_party_merge
* clean/news_05_select_and_merge_tickers_relevant.py select and news with some returns and a single firm. 
* news_06_prep_vec_and_news_on_that_day finish the job and kinda get the same data as bryan. 

TF RECORDS:
* vec_to_tf_records.py is transforming it into this nice tf format
* train_tf.py is my full training routine.



to push to gadi
load_control_coverage
load_bryan_data
rel_max

