#Technion Deep Learning Course HW1 - MNIST Classifier
###Nadav Shai Oved - 200689768
###Aviv Sugarman - 305652729


classification_mnist.lua - modified file from tutorial 4 which builds the network and saves it to a file in your cwd called 'MyModel.dat'

loadMyModel.lua - loads our saved network file and returns the error on the MNIST test set -
usage: 
```
th
loader = require 'loadMyModel.lua'
loader.loadMyModel()
```

Report.PDF - Short report describing our work + graphs
