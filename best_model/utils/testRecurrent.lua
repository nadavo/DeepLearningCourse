require 'optim'
require 'nn'
require 'recurrent'

function loadTextFileWords(filename, vocab)

    local file = io.open(filename, 'r')
    local words = file:read("*all"):split(' ')
    local length = #words
    local wordsVec = torch.LongTensor(#words):zero()

    local vocab = vocab or {['\n'] = 1}
    local currentNum = 1
    --count num words (in case of existing vocab)
    for _ in pairs(vocab) do currentNum = currentNum + 1 end

    for i=1, length do
        local currWord = words[i]
        local encodedNum = vocab[currWord]
        if not encodedNum then
            vocab[currWord] = currentNum
            encodedNum = currentNum
            currentNum = currentNum + 1
        end
        wordsVec[i] = encodedNum
    end

    local decoder = {}
    for word, num in pairs(vocab) do
        decoder[num] = word
    end
    return wordsVec, vocab, decoder
end

function decodeFunc(decoder, mode)
  local space = ''
  if mode == 'word' then
    space = ' '
  end
  local func = function(vec)
    local output = ''
    for i=1, vec:size(1) do
        output = output .. space .. decoder[vec[i]]
    end
    return output
  end
  return func
end


 function encodeFunc(vocab, mode)
  local func
  if mode == 'word' then
    func = function(str)
      local words = str:split(' ')
      local length = #words
      local encoded = torch.LongTensor(#words):zero()

      for i=1, length do
          local currWord = words[i]
          local encodedNum = vocab[currWord]
          if not encodedNum then
              encodedNum = -1
          end
          encoded[i] = encodedNum
      end
      return encoded
    end
  elseif mode == 'char' then
    func = function(str)
    local length = #str
    local encoded = torch.ByteTensor(length):zero()

    for i=1, length do
      local encodedNum = vocab[str[i]]
      if not encodedNum then
          encodedNum = -1
        end
        encoded[i] = encodedNum
      end
      return encoded
    end
  end
    return func
end

local function ForwardSeq(model, dataVec, seqLength, batchSize)

    local data, labels = reshapeData(dataVec, seqLength, batchSize)
    local sizeData = data:size(1)
    local numSamples = 0
    local lossVal = 0
    local currLoss = 0
    local x = torch.Tensor(batchSize, seqLength):type(TensorType)
    local yt = torch.Tensor(batchSize, seqLength):type(TensorType)

    -- input is a sequence
    model:sequence()
    model:forget()

    for b=1, sizeData do
        if b==1 then --no dependancy between consecutive batches
            model:zeroState()
        end
        x:copy(data[b])
        yt:narrow(2,1,seqLength-1):copy(x:narrow(2,2,seqLength-1))
        yt:select(2,seqLength):copy(labels[b])

        y = model:forward(x)
        currLoss = seqCriterion:forward(y,yt)

        lossVal = currLoss /  seqLength + lossVal
        numSamples = numSamples + x:size(1)
        xlua.progress(numSamples, sizeData*batchSize)
    end

    collectgarbage()
    xlua.progress(numSamples, sizeData)
    return lossVal / sizeData
end


local function evaluate(model,dataVec,seqLength,batchSize)
    model:evaluate()
    return ForwardSeq(model, dataVec,seqLength,batchSize)
end

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

return 
{
	loadTextFileWords = loadTextFileWords,
	evaluate = evaluate,
	sample = sample
}
