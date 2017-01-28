local samples = 60
local epochs = samples*5
local trainLoss = torch.Tensor(samples)
local testLoss = torch.Tensor(samples)

local i=1
local j=0

local file = io.open("testloss.txt")
if file then
    for line in file:lines() do
        local loss = unpack(line:split(" ")) --unpack turns a table like the one given (if you use the recommended version) into a bunch of separate variables
        testLoss[i] = loss
	--for j=0,4 do
		--testLoss[i+j] = loss
	--end
	i=i+1
	if i>samples then
		break
	end
    end
end
i=1
local file = io.open("trainloss.txt")
if file then
    for line in file:lines() do
        local loss = unpack(line:split(" ")) --unpack turns a table like the one given (if you use the recommended version) into a bunch of separate variables
        trainLoss[i] = loss
	--for j=0,4 do
		--trainLoss[i+j] = loss
	--end
	i=i+1
	if i>samples then
		break
	end
    end
end

--local avg1 = 0
--local num1 = 0
--local num2 = 0
--local avg2 = 0
--local num3 = 0
--local num4 = 0

--for i=1,(epochs-5) do
--	if (i-1)%5==0 then
--        	for j=0,4 do
			--num1 = trainLoss[i+j]
			--num2 = trainLoss[i+j+5]
			--avg1 = (num1+num2)/2	
			--trainLoss[i+j+1] = avg1
			--num3 = testLoss[i+j]
			--num4 = testLoss[i+j+5]
			--avg2 = (num3+num4)/2	
			--testLoss[i+j+1] = avg2
--		end
--	end
--	i=i+4
--end

print(trainLoss)
print(testLoss)

--local xtics_string = '('

--for i=1,60 do
	--xtics_string = xtics_string .. '"' .. i*5 .. '"' .. ' ' .. i .. ','
--end 

--xtics_string = xtics_string .. ')'

--print (xtics_string)

local function plotLoss(trainLoss, testLoss, title)
	require 'gnuplot'
	local range = torch.range(0, epochs,5)
	gnuplot.pngfigure('testVsTrainLoss.png')
	gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
	gnuplot.raw('set xtics ("0" 0, "50" 10, "100" 20, "150" 30, "200" 40, "250" 50, "300" 60)')
	gnuplot.axis{0,60,0,''}
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()
end

plotLoss (trainLoss,testLoss,'Classification Loss')
