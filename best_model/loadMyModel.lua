require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
-------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training recurrent networks on word-level text dataset - Penn Treebank')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Data Options')
cmd:option('-shuffle',            false,                       'shuffle training samples')

cmd:text('===>Model And Training Regime')
cmd:option('-model',              'LSTM',                      'Recurrent model [RNN, iRNN, LSTM, GRU]')
cmd:option('-seqLength',          8,                         'number of timesteps to unroll for')
cmd:option('-rnnSize',            212,                         'size of rnn hidden layer')
cmd:option('-numLayers',          2,                           'number of layers in the LSTM')
cmd:option('-dropout',            0.25,                           'dropout p value')
cmd:option('-LR',                 25e-4,                        'learning rate')
cmd:option('-LRDecay',            0,                           'learning rate decay (in # samples)')
cmd:option('-weightDecay',        0,                           'L2 penalty on the weights')
cmd:option('-momentum',           0,                           'momentum')
cmd:option('-batchSize',          50,                          'batch size')
cmd:option('-decayRate',          2,                           'exponential decay rate')
cmd:option('-initWeight',         0.08,                        'uniform weight initialization range')
cmd:option('-earlyStop',          5,                           'number of bad epochs to stop after')
cmd:option('-optimization',       'rmsprop',                   'optimization method')
cmd:option('-gradClip',           5,                           'clip gradients at this value')
cmd:option('-epoch',              30,                         'number of epochs to train')
cmd:option('-epochDecay',         5,                           'number of epochs to start decay learning rate')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                           'number of threads')
cmd:option('-type',               'cuda',                      'float or cuda')
cmd:option('-devid',              1,                           'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                           'num of gpu devices used')
cmd:option('-seed',               123,                         'torch manual random number generator seed')
cmd:option('-constBatchSize',     false,                       'do not allow varying batch sizes')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                          'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''),      'save directory')
cmd:option('-optState',           false,                       'Save optimization state every epoch')
cmd:option('-checkpoint',         0,                           'Save a weight check point every n samples. 0 for off')


opt = cmd:parse(arg or {})

require 'utils.textDataProvider'
--local loadTextFileWords = tester.loadTextFileWords

torch.setnumthreads(8)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
local trainWordVec, valWordVec, testWordVec, decoder, decoder_, vocab

trainWordVec, vocab, decoder = loadTextFileWords('./data/ptb.train.txt')
valWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.valid.txt', vocab)
assert(#decoder == #decoder_) --no new words
testWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.test.txt', vocab)
assert(#decoder == #decoder_) --no new words
data = {
  trainData = trainWordVec,
  validationData = valWordVec,
  testData = testWordVec,
  vocabSize = #decoder,
  decoder = decoder,
  vocab = vocab,
  decode = decodeFunc(vocab, 'word'),
  encode = encodeFunc(vocab, 'word')
}

local vocabSize = #decoder
----------------------------------------------------------------------

modelConfig = torch.load('MyModel.dat')
modelConfig.classifier:share(modelConfig.embedder, 'weight', 'gradWeight')

tester = require 'utils.trainRecurrent'
local evaluate = tester.evaluate
local sample = tester.sample

local testPerplexity = torch.Tensor(5)

for i=1,5 do
  local LossTest = evaluate(data.testData)
  testPerplexity[i] = torch.exp(LossTest)
  print('\nSampled Text:\n' .. sample('Buy low, sell high is the', opt.seqLength, true))

  print('\nTest Perplexity: ' .. testPerplexity[i])
end
