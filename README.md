# Noise CapuchinBird Recognized NeuralNetwork Model

A simple binary classification model for the sounds of the capuchin bird

Implementation using PyTorch

![img](https://i.imgur.com/uWP142C.jpg)

Model in 18 layers proved to be the best result
```Python
[1,    10] loss: 0.594 accuracy: 0.700
[1,    20] loss: 0.552 accuracy: 0.750
[1,    30] loss: 0.457 accuracy: 0.800
[1,    40] loss: 0.469 accuracy: 0.775
[1,    50] loss: 0.319 accuracy: 0.925
[1,    60] loss: 0.497 accuracy: 0.725
[1,    70] loss: 0.396 accuracy: 0.875
[1,    80] loss: 0.412 accuracy: 0.800
[1,    90] loss: 0.405 accuracy: 0.775
[1,   100] loss: 0.341 accuracy: 0.875
[1,   110] loss: 0.354 accuracy: 0.850
[1,   120] loss: 0.244 accuracy: 0.875
[1,   130] loss: 0.145 accuracy: 1.000
[1,   140] loss: 0.270 accuracy: 0.900

Score: 0.9877049180327869
```
# Parametric data hyperparameters
```
number features: 512
learn rate: 0.001
momentum: 0.3
epoche: 1
```
