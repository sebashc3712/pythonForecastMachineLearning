# pythonForecastMachineLearning
This code finds the best algorithm from sklearn to forecast your numeric data

## Getting Started

To getting started with pythonFirecastMachineLearning you just clone this repository in your project folder and import forecastingML
in your file

```
git clone https://github.com/sebashc3712/pythonForecastMachineLearning
```
```
from ForecastingML import bestForecastModel
```
Then you have to call the function bestForecastingModel and save the model into a file .pkl
```
a,b=bestForecastModel(Dataset,Target)
joblib.dump(a, 'bestForecastModel.pkl') 
```

### Parameters of the function
```
bestForecastModel(Dataset, Target, MLP = False, MaxLayers=100, MaxDepth=10, neighbors=20)
```
- MLP (deafult False) when is True the neural network algorithm is activated. (This can take a while. Until 24 hrs depengin of the number of max layers and the numbers of rows of your dataset)
  - MaxLayers (default 100). It only works if MLP is True. This parameter is the highest number of hidden layers for the algorithm
- MaxDepth (default 10). That is the maximum deep for the random forest and decision tree algorithms
- neighbors (default 20). That is the maximun neighbors for the isomap algorithm

### Prerequisites

You need to install sklearn and pandas. The easiest way to install these packages is to install Anaconda in your PC

```
https://www.anaconda.com/download/
```
### License

MIT
