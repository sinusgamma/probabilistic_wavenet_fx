# Wavenet variations for financial time series prediction: the simple, the directional-Relu, and the probabilistic approach

Article on medium about the goals and the model: https://medium.com/@istvan.veber/wavenet-variations-for-financial-time-series-prediction-the-simple-the-directional-relu-and-the-4860d8a97af1

Data preprocessing and pipeline manual: https://github.com/sinusgamma/probabilistic_wavenet_fx/blob/master/data_preparation.md

My goal is to forecast forex bar features of the upcoming step. Forecasted pairs: EUR/USD, GBP/USD, and JPY/USD (not the more common USD/JPY). We will forecast simple price values, log-returns, and the directions of changes.

We will play a bit with the Relu activation as output. With activation functions as output, we can determine the codomain of our models. At the end, we will build the Wavenet model with distribution output. For that, we will use the Tensorflow probability library.

We will forecast features calculated from the tick means of the bar ranges, and not OHCL. Tick means are more representative measurements of the prices during the bar period, and not so noisy as the closing price. It is easier for a model to find patterns if we use means.

The data inputs have two main components:
 - features generated from tick data during a 5-minute range (source: Dukascopy)
 - features generated from economic news calendar (source: FXStreet)
 
Generating hundreds of features required compromises and arbitrary choices. The data preparation article is coming, preparing the data took a longer time than building the models and training them. In the article, Financial bars at the age of deep learning, you can read about some of the ideas.

To feed the data to our model we will use the Tensorflow dataset API.

For training, we will use the 2016–2018 period, and 2019 is the validation period. For simplicity, I didn’t use separate test data.

The models were trained on Google Cloud AI Platform.

The notebook with the code is available on Github: https://github.com/sinusgamma/probabilistic_wavenet_fx/blob/master/wavenet_fx_final.ipynb
