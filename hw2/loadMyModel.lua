require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-model','./network.model','Path to model file which will be loaded')
cmd:option('-cifar','./cifar.torch','Path to directory containing cifar10 train and test data sets')

-- parse input params
opt = cmd:parse(arg or {})

local trainset = torch.load(opt.cifar..'/cifar10-train.t7')
local testset = torch.load(opt.cifar..'/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

-- Load and normalize data:

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    --print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    --print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


do
local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

	function BatchFlip:__init()
		parent.__init(self)
		self.train = true
	end 

function BatchFlip:updateOutput(input)
	if self.train then
		local permutation = torch.randperm(input:size(1))
	for i=1,input:size(1) do
		if permutation[i] % 4 == 0 then
		-- hflip
			image.hflip(input[i]:float(), input[i]:float())
		end
		if permutation[i] % 4 == 1 then
		-- vflip
			image.vflip(input[i]:float(), input[i]:float())
		end
		if permutation[i] % 4 == 2 then
		-- random crop
			randomcrop(input[i], 10, 'reflection')
		end
	end
	end
	--self.output:set(input)
	self.output:set(input:cuda())
	return self.output
end
end

local batchSize = 64
local optimState = {}
model = torch.load(opt.model)
criterion = nn.ClassNLLCriterion():cuda()

function forwardNet(data,labels)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    model:evaluate() -- turn off drop-out
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
    end

    confusion:updateValids()
    local avgError = 1 - confusion.totalValid

    return avgError
end

---------------------------------------------------------------------

testError = forwardNet(testData, testLabels)
print(testError)
return testError
