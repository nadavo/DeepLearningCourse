#Technion Deep Learning Course HW2 - CNN
###Nadav Shai Oved - 200689768
###Aviv Sugarman - 305652729


**cifar10_classifier_*optimizer*_*data-augmentation*.lua** - modified file from Tutorial 5 which builds the network according to the optimizer and data augmentation and saves it to a file in your cwd called 'network.model'

We trained 4 different models (best accuracy over 300 epochs):
- Adam + hflip (81.6%)
- Adam + hflip + vflip (81.16%)
- Adam + hflip + vflip + randomcrop (81.75%)
- SGD + hflip (71.73%)


**loadMyModel.lua** - loads our trained network from given file and returns the average error on cifar10-test set.

receives 2 parameters:

-model *path to model file* - default is: ./network.model

-cifar *path to cifar10 datasets directory* - default is: ./cifar.torch

usage:
```
th loadMyModel.lua -model <path to model file> -cifar <path to cifar10 datasets directory>
```

**Report.PDF** - Short report describing our work + graphs
