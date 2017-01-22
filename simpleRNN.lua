require 'rnn'
-- nn.ParallelTable() : in its forward() method, applies the
-- i-th member module to the i-th input, and outputs a table of the set of outputs.
-- CAddTable() : Takes a table of Tensors and outputs summation of all Tensors.
local rm = nn.Sequential()
		   :add(nn.ParallelTable() 
				:add(nn.Linear(10, 7))
				:add(nn.Linear(7, 7)))
		   :add(nn.CAddTable())
		   :add(nn.Sigmoid())

local r = nn.Recurrence(rm, 7, 1) 

print(r)

--------------------------------------------------------------------------
-- Compute the output at a time step

local rr = nn.Sequential()
			:add(r)
			:add(nn.Linear(7, 10))
			:add(nn.LogSoftMax())

--------------------------------------------------------------------------
-- Make it a recurrent module

local rnn = nn.Recursor(rr, 5)

--------------------------------------------------------------------------
-- Apply each element of a sequence to the RNN step by step

local inputs = torch.Tensor(5,10)
local targets = torch.Tensor(5)
targets[1] = 8; targets[2] = 2; targets[3] = 3; targets[4] = 4; targets[5] = 5;

local outputs, err = {}, 0
local criterion = nn.ClassNLLCriterion()
for step=1,5 do
   outputs[step] = rnn:forward(inputs[step])
   err = err + criterion:forward(outputs[step], targets[step])
end

print(outputs)

--------------------------------------------------------------------------
-- Train the RNN step by step through the sequence

local gradOutputs, gradInputs = {}, {}
for step=5,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])  --accumulate grad parameters
end

print(gradOutputs)
print(gradInputs)  -- we need it or multi layer rnn.

rnn:updateParameters(0.1) -- learning rate

--------------------------------------------------------------------------
-- Reset the RNN

rnn:forget()
rnn:recycle()
rnn:zeroGradParameters()

--------------------------------------------------------------------------
-- Apply RNN to a sequence in one step

rnn = nn.Sequencer(rr)
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

outputs = rnn:forward(inputs)  --print(outputs)
err = criterion:forward(outputs, targets) --print(err)
gradOutputs = criterion:backward(outputs, targets) --print(gradOutputs)
gradInputs = rnn:backward(inputs, gradOutputs) --print(gradInputs)
rnn:updateParameters(0.1)

--------------------------------------------------------------------------
--Regularize RNN rr:add(nn.NormStabilizer([beata]))





