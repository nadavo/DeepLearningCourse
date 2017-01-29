#Technion Deep Learning Course HW2 - CNN
###Nadav Shai Oved - 200689768
###Aviv Sugarman - 305652729


**Report.PDF** - Short report describing our work + graphs

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


Usage:
```
th loadMyModel.lua -model <path to model file> -cifar <path to cifar10 datasets directory>
```

Models are in:
```
on ml8.iem.technion.ac.il
/home/nadavo@st.technion.ac.il/hw2_models/<optimizer_data-augmentation>/network.model
```

Example:
```
th loadMyModel.lua -model /home/nadavo@st.technion.ac.il/hw2_models/adam_hflip/network.model  -cifar ./cifar.torch
```

Please make sure you have installed all dependencies:
```
luarocks install nn
luarocks install cunn
luarocks install cudnn
luarocks install optim
luarocks install image
luarocks install gnuplot
git clone https://github.com/soumith/cifar.torch.git
cd ./cifar.torch/
th Cifar10BinToTensor.lua
```
