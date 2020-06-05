# Building forex dataset for deep learning models - Manual

This manual will guide you through the process of creating forex dataset for deep learning models.

The dataset consist of two mayor parts:
* Forex tick data
* Economic news data

Four .ipynb files contains the code of the pipeline:
* [resampler.ipynb](https://github.com/sinusgamma/probabilistic_wavenet_fx/blob/master/resampler.ipynb) - Samples 5-minute bars with multiple features from tick data.
* [news_transformer.ipynb](https://github.com/sinusgamma/probabilistic_wavenet_fx/blob/master/news_transformer.ipynb) - Transforms the raw economic news data.
* [data_merger.ipynb](https://github.com/sinusgamma/probabilistic_wavenet_fx/blob/master/data_merger.ipynb) - Creates ona dataset from the forex and news data.
* [wavenet_fx_final.ipynb](https://github.com/sinusgamma/probabilistic_wavenet_fx/blob/master/wavenet_fx_final.ipynb) - Builds model specific features and inputs the data to the Tensorflow models with the Tensorflow Dataset API. The second part of this notebook is the model training and evaluation part.

Data timerange: 2016 - 2019

Data Source:

Forex data: https://www.dukascopy.com/swiss/english/marketwatch/charts/  
Economic news data: https://www.fxstreet.com/economic-calendar  

## resampler.ipynb

This notebook resamples tick data to 5 minute bars, and generates the bar features.
We use EUR/USD, GBP/USD and USD/JPY tick data. For my computing power the 4 year tick is a very large load, so I decided to experiment with bars. To retain information of the price movement in a bar range we generat multiple features. The USD/JPY pair is converted to JPY/USD, USD will be our base in every pair. Combining the ask and bid price could lead to usefull feature, but because of computing contrainse we use only the ask price.

The tick_resampler function generates our basic bar features:
* We generate the overused OHCL features.
* mean: average of ticks
* std: standard deviation of ticks
* bar_len: number of ticks in the period 
* bar_quantile_25 and bar_quantile_75: the prices at Q1 and Q3, less volatile than the OHCL features
* bar_spearman: Spearman's rank correlation of the price with time. Good feature for finding trending or volatile bars.
* bar_log_r: log-return between the open and close tick (not the same as the log return of subsequent bars!)
* ..._r features: Log returns have good properties, we convert some of the original features to a log-returnized version. The base of these features is the bar mean, which has the lowest variance, and the other features are log-returnized compared to the value of the mean.

The tick_resampler function can produce shifted bars, and this way we can generate augmented time series, but our dataset will consist of only the unshifted regular 5 minute bars.

![GitHub Logo](

