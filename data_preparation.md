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

The histrogram of the features, the bar_spearman is worth a glance.

![Feature Histograms](https://raw.githubusercontent.com/sinusgamma/probabilistic_wavenet_fx/master/bar_feature_hist.jpg)

## news_transformer.ipynb

We clear the economic news data in this notebook. The original database has 499 GBP, EUR, JPY or USD related event. We keep only the high impact events.  

Most of the features have numerical subfeatures like 'actual', 'consensus', 'previous'. Other events don't have numeric data only the time of the event. We group these other events without numeric subfeatures and handle them as 'speech' or 'Event'.

There are very rare events, we group them to the 'Event' as well. 

After that we have to fill the NaN values.

We create new economic news features: change and surprise.

From the EUR economic events we keep only events of the whole region, Germany and France.

After these transformations we have 76 economic event types.


## data_merger.ipynb

The forex data is sampled at every 5 minutes, but the news data isn't so frequent. We merge these datasets in this notebook.

First we process the news data again, and generate one-hot-encoded features from the datasets.
* event_exist: shows if there is any event at the timestep
* even_cur: shows if any of the currencies has any event at the timestep (far fewer columns than event_exist)
* actual_ohe: last actual data of the event
* previous_ohe: last previous data of the event
* change_ohe: last change of the event
* surprise_ohe: last change rate of the event
* after_counter_ohe: 1.0 when the event happens, and decreases for a predefined steps till it will reach 0.0. After that it is 0.0. This is a kind of memory, showing the network how far are the step from the last occurence of the event.
* time features

## wavenet_fx_final.ipynb

Final data preprocessing for the specific task and architecture of the deep learning models. Building Tensorflow Dataset API pipelins.  
This notebook containes the code of the models and evaulation as well.

New feature: mean_log_r - the log return of the subsequent bar means. Bar means are more stable than OHCL features, so we will forecast the change of the bar means (and the price in other models).

Divide the training (2016-2018) and validation (2019) set.

Based on the training period some features are normalized, standardized or scaled (when scaling we don't shift the data, only change its magnitude).

To keep the relative value of the log returnized features we only divide them by the standard deviation of the bar_log_r.

Building the label datasets for the currency prices and log returns of the bar means.

Calculating MAE of naive forecast.

With the Tensorflow Dataset API building variations of input datasets for the models of the notebbok.


