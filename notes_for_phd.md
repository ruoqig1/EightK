# starting point in my code is 
vec_news_ref.slurm 
    * this is slurm code to creates nodes on the HPC and launch it. 
    * partition is on gpu
    * note how I'm using virtual envirnoments
    * note how I'm parsing inpute in the code
in vec_news you can ignore 99.9% and only focus on the secitons with the args
Then you will see vectorise_in_batch


vectorise_in_batch
- this guy is doing a loop with a system of batch
- my idea was to fill the gpu by playing with this batch paramete but I think it's not working

the big class doing the heavy lifting is in llm.py and is just a wrapper arround LLM from hugging phase


### General code structure
1) Paramneters.
[perf_portfolio.py](perf_portfolio.py)


# not for angella
* check notes above to udnerstand how vecotrisation code works 
* check vec_to_tf to understand how we build the datset
* check train_tf to udnerstand how to train the elasticnet model. 
* perf_tf build the non working sharpe ratio :)