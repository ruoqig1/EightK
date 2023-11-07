# Dataset Column Descriptions

## `date`
This column likely contains the dates corresponding to each data entry, possibly in a standard date format such as YYYY-MM-DD.

## `permno`
The `permno` is a unique identifier assigned by the Center for Research in Security Prices (CRSP) to each security for the purpose of tracking its performance over time.

## `abret`
abnormal return computed on market abret_i,t = r_i,t -beta_i,t mkt_t

## `sigma_ra`
This column could represent the standard deviation of the abnormal returns.
Gpt is correct, it's the standard deviation of the abret i nthe 100 days rolling window used to estimate each beta_i,t

## `abs_abret`
"Abs_abret" could stand for the absolute value of abnormal returns, focusing on the magnitude of returns without regard to direction (positive or negative).

## `sigma_abs_ra`
Likely represents the standard deviation of the absolute abnormal returns, offering insight into the volatility of these magnitudes.
Correct again, it's std(abs(abret_i,t))

## `news`
Binary equal to 1 if there is any type of ravenpack news on that day about that firm

## `big`
1 if mcap_d = 1

## `mcap_d`
"MCap_d" possibly stands for market capitalization on the date provided, reflecting the total market value of the security's outstanding shares.
It's 10 if you are in the 10th decile during year 1 (by market cap)

## `in_snp`
This might be a binary indicator denoting whether the security is part of the S&P index at the time of the observation.
Correct :)

## `evttime`
"Evttime" could refer to the time of an event, such as a news release or financial report, likely time-stamped.
Indeed, -20 means 20 days vefore the date of the 8kj=

## `items`
itmes number of 8k

## `press`
1 if on that day we have a relevant press release defined by ravenpack
Relevant means ravenpack relevance above 0 for that day and firm.

## `article`
1 if on that day we have a relevant non-press release ravenpack news that day.
