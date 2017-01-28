#Technion Deep Learning Course HW3 - LSTM RNN
###Nadav Shai Oved - 200689768
###Aviv Sugarman - 305652729

wordRNN_edited.lua - modified file from Elad Hoffer's implementation which builds the network and saves it every 10 epochs

loadMyModel.lua - loads our saved network file and returns the test perplexity on the Penn Treebank test set and outputs 5 sentence completions

MyModel.dat - Our saved network model file

MyModelPerplexityGraph.png - Graph showing our network's Perplexity per epoch for test and training sets

MyModel_log.txt - Our network's training log

Please make sure you have installed all dependencies:
```
luarocks install nn
luarocks install optim
luarocks install recurrent
git clone https://github.com/eladhoffer/eladtools.git
cd ./eladtools/
luarocks make
```

Training: 
```
th wordRNN_edited.lua
```

Testing our model:
```
th loadMyModel.lua
```

Report.PDF - Short report describing our work + graph
