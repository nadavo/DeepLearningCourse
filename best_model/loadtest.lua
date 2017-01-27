require 'torch'
require 'nn'
require 'optim'
require 'eladtools'
require 'recurrent'
require 'utils.textDataProvider'

cmd = torch.CmdLine()
cmd:option('-model', 'MyModel.dat', 'Model filename')
cmd:option('-devid', 1, 'device ID (if using CUDA)')
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-string', 'buy low sell high is the', 'Sentence to complete in sample')
cmd:option('-num', 10, 'number of words to predict')

opt = cmd:parse(arg or {})

local TensorType = 'torch.FloatTensor'

print('Loading data')
local trainWordVec, decoder, vocab

trainWordVec, vocab, decoder = loadTextFileWords('./data/ptb.train.txt')
data = {
  trainingData = trainWordVec,
  vocabSize = #decoder,
  decoder = decoder,
  vocab = vocab,
  decode = decodeFunc(vocab, 'word'),
  encode = encodeFunc(vocab, 'word')
}
local vocabSize = #decoder


print('Loading model')

-- get model from modelConfig object
modelConfig = torch.load(opt.model)
local embedder = modelConfig.embedder
local classifier = modelConfig.classifier
local recurrent = modelConfig.recurrent

print('Re-creating model')
-- re-create model
local model =  nn.Sequential()
model:add(embedder)
model:add(recurrent)
model:add(nn.TemporalModule(classifier))
print (model)

print('Sampling:')
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
	print('\nSentence ' .. i .. ': ' .. sample(opt.string, opt.num, true))
end
