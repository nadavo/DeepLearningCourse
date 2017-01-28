require 'torch'
require 'nn'
require 'optim'
require 'eladtools'
require 'recurrent'
require 'utils.textDataProvider'
-------------------------------------------------------

cmd = torch.CmdLine()
cmd:option('-devid', 1, 'device ID (if using CUDA)')
cmd:option('-loadModel', 'MyModel.dat', 'Model filename')
cmd:option('-sentence', 'buy low sell high is the', 'Sentence to complete in sample function')
cmd:option('-seqLength', 10, 'Number of words to predict for sentence completion')

opt = cmd:parse(arg or {})

torch.setnumthreads(8)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')
local TensorType = 'torch.FloatTensor'

----------------------------------------------------------------------
local trainWordVec, testWordVec, decoder, decoder_, vocab

trainWordVec, vocab, decoder = loadTextFileWords('./data/ptb.train.txt')
testWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.test.txt', vocab)
assert(#decoder == #decoder_) --no new words
data = {
  trainData = trainWordVec,
  testData = testWordVec,
  vocabSize = #decoder,
  decoder = decoder,
  vocab = vocab,
  decode = decodeFunc(vocab, 'word'),
  encode = encodeFunc(vocab, 'word')
}

local vocabSize = #decoder
----------------------------------------------------------------------

modelConfig = torch.load(opt.loadModel)

local model =  nn.Sequential()
model:add(modelConfig.embedder)
model:add(modelConfig.recurrent)
model:add(nn.TemporalModule(modelConfig.classifier))
print (model)

modelConfig.classifier:share(modelConfig.embedder, 'weight', 'gradWeight')

print('Sampling 5 Sentences:')
local function sample(str, num, space, temperature)
    local num = num or 50
    local temperature = temperature or 1
    local function smp(preds)
        if temperature == 0 then
            local _, num = preds:max(2)
            return num
        else
            preds:div(temperature) -- scale by temperature
            local probs = preds:squeeze()
            probs:div(probs:sum()) -- renormalize so probs sum to one
            local num = torch.multinomial(probs:float(), 1):typeAs(preds)
            return num
        end
    end

    recurrent:evaluate()
    recurrent:single()

    local sampleModel = nn.Sequential():add(embedder):add(recurrent):add(classifier):add(nn.SoftMax():type(TensorType))

    local pred, predText, embedded
    if str then
        local encoded = data.encode(str)
        for i=1, encoded:nElement() do
            pred = sampleModel:forward(encoded:narrow(1,i,1))
        end
        wordNum = smp(pred)

        predText = str .. '... ' .. decoder[wordNum:squeeze()]
    else
        wordNum = torch.Tensor(1):random(vocabSize):type(TensorType)
        predText = ''
    end

    for i=1, num do
        pred = sampleModel:forward(wordNum)
        wordNum = smp(pred)
        if space then
            predText = predText .. ' ' .. decoder[wordNum:squeeze()]
        else
            predText = predText .. decoder[wordNum:squeeze()]
        end
    end
    return predText
end

for i=1,5 do
	print('\nSentence ' .. i .. ':\n' .. sample(opt.sentence, opt.seqLength, true))
end
