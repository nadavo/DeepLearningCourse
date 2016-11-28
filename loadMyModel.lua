function loadMyModel()
    require 'nn'
    require 'cunn'
    local mnist = require 'mnist'
    local optim = require 'optim'
    local model = torch.load('MyModel')
    local testData = mnist.testdataset().data:float();
    testData:add(-mean):div(std);
    local testLabels = mnist.testdataset().label:add(1);
    local avgError = forwardNet(model,testData,testLabels)
    return avgError
end

local function forwardNet(model, data, labels)
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local batchSize = 64
    local criterion = nn.ClassNLLCriterion():cuda()
    local numBatches = 0
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        confusion:batchAdd(y,yt)
    end
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid
    return avgError
end