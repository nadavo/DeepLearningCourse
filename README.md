#Technion Deep Learning Course HW3 - LSTM RNN
###Nadav Shai Oved - 200689768
###Aviv Sugarman - 305652729


wordRNN_edited.lua - modified file from Elad Hoffer's implementation which builds the network and saves it to a file in your cwd called 'MyModel.dat'

loadMyModel.lua - loads our saved network file and returns the test perplexity on the Penn Treebank test set and outputs 5 sentence completions -
usage: 
```
th
loader = require 'loadMyModel.lua'
loader.loadMyModel()
```

Report.PDF - Short report describing our work + graphs
