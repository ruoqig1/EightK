# TODO 
TODO:
Vectorisation:
Produce smaller dataset of news in loadable batches for vectorisation 
	- give them individual id
	- match them with price trend in.
	- save it in little batch
	- keep the one only talking about one firm :(
Launch Vectorisation of News

SSH:
	- setup ssh to new server
	- learn how to launch basic array job. 
	- set up a code pushing data from one area to the other. :(
		- one big one for all push
		- one specific where you define the directory to same directory on the other side. 
	- build a venv on the other side

TFIDF:
	- Launch tokenisation of 
		- news about one firm with ore more tickers
		- 8k press release
	- build a measure of text similarity (cosine similarity for example.)
		- run this measure with news that are about the firm up to 10 days before and after each 8k event per firm. 

# Training of models, 
First perfect the process with: 
	- training procedure easy to launch some variations
	- testing, simple code that can run one or many model side by side. 
try:
	- fix training horizons, 
	- different rolling window,
	- different hyper parameters
	- more complex model (gradient boosting trees for example?) 


# main result with new data:
	- check that it works with the new definition of in the news
