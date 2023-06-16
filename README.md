# Boston-House-Pricing-Forecast
[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/FywOo02/Boston-House-Pricing-Forecast) 
![Python](https://img.shields.io/badge/python-3.10-519dd9.svg?v=103)
![Name](https://badgen.net/badge/Author/FywOo02/orange?)
![Language](https://badgen.net/badge/Language/English/pink?)
![field](https://badgen.net/badge/Field/MachineLearning/green?)

## What is this Project?
- Boston House Price Forecast based on the scikit-learn library. It is very friendly for newbies to machine learning, easy to get started, and doesn't have a lot of dirty data to work with. The core goal of the project is to find potential relationships between feature values and target values in past experience (data), and use algorithms from different models to fit a function that best fits the house price trend and apply it to future predictions.

<div align=center>
<img src="https://github.com/FywOo02/Boston-House-Pricing-Forecast/blob/master/myplot.png">
</div>

## Where does the data from?
- URL Version: http://lib.stat.cmu.edu/datasets/boston
- CSV version: https://www.kaggle.com/datasets/altavish/boston-housing-dataset

## How did we optimize the module for fitting?
- Based on the classic linear regression model, we tried cross-sectional optimization to explore utilizing ridge regression and LASSO regression for the dataset prediction.
- Analyzing the correlation between various characteristics and house prices, and the accuracy of data. After conducting several comparison experiments, we finally found and tuned the Gradient Boosting Regressor model that can make the function fit up to 92%.

## How can I see the results?
1. Install the related libraries
> This project uses machine learning related libraries, go check them out if you don't have them locally installed
```
pip install sklearn
pip install numpy
pip install pandas
pip install matplotlib
```
2. Clone the original files in git bash
```
git clone https://github.com/FywOo02/Boston-House-Pricing-Forecast.git
```
3. Open the forecast_origin.py in your python ide, and try to catch the predicted and fitting results
```
forecast_origin.py
```


## Maintainer
[@FywOo02_Cho](https://github.com/FywOo02)

## Contributor
<a href="https://github.com/FywOo02">
  <img src="https://github.com/FywOo02.png?size=50">
</a>

## License
[MIT](https://github.com/FywOo02/Boston-House-Pricing-Forecast/blob/master/LICENSE) © Cho Zhu

