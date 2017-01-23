require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
require 'utils.testRecurrent'
-------------------------------------------------------

torch.setnumthreads(8)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
local testWordVec, decoder, decoder_, vocab
local batchSize = 50
local seqLength = 8

testWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.test.txt', vocab)
assert(#decoder == #decoder_) --no new words
data = {
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

local testPerplexity = torch.Tensor(5)

for i=1,5 do
  local LossTest = evaluate(data.testData,seqLength,batchSize)
  testPerplexity[i] = torch.exp(LossTest)
  print('\nSampled Text:\n' .. sample('Buy low, sell high is the', seqLength, true))

  print('\nTest Perplexity: ' .. testPerplexity[i])
end
