math.randomseed(tick())

local rand = math.random
local max = math.max
local exp = math.exp

local function Activator(x)
	return max(0.1 * x, x)
end

local function Sigmoid(x)
	return 1 / (1 + exp(-x))
end

local module = {}

function module.Create(Size)
	local Network = {}
	for i=2,#Size do --for the number of specified layers
		Network[i] = {}
		for o=1,Size[i] do --for the number of specified nodes in the current layer
			Network[i][o] = {}
			for p=1,Size[i - 1] + 1 do --for the number of nodes in the previous layer [the extra one is the bias]
				repeat
					Network[i][o][p] = rand(-200000000, 200000000) / 100000000 * math.sqrt(2 / Size[i - 1])
				until Network[i][o][p] >= 0.5 or Network[i][o][p] <= -0.5
			end
		end
	end
	return Network
end

function module.Run(Network, Inputs, backp)
	local activations = {Inputs}
	local rawactivations = {Inputs} --for backpropagation
	if backp then
		for i = 2, #Network - 1 do --for the number of layers
			activations[i] = {}
			rawactivations[i] = {}
			for o = 1, #Network[i] do --for the number of nodes in the current layer
				activations[i][o] = 0

				for p = 1, #Network[i][o] - 1 do --for the number of weights for the current node
					activations[i][o] = activations[i][o] + Network[i][o][p] * activations[i - 1][p]
				end

				rawactivations[i][o] = activations[i][o]
				activations[i][o] = Activator(activations[i][o] + Network[i][o][#Network[i][o]])
			end
		end
		activations[#Network] = {}
		rawactivations[#Network] = {}
		for o=1,#Network[#Network] do
			activations[#Network][o] = 0
			for p=1,#Network[#Network][o] - 1 do
				activations[#Network][o] = activations[#Network][o] + Network[#Network][o][p] * activations[#Network - 1][p]
			end
			rawactivations[#Network][o] = activations[#Network][o]
			activations[#Network][o] = Sigmoid(activations[#Network][o] + Network[#Network][o][#Network[#Network][o]])
		end
		return {activations, rawactivations}
	else
		for i = 2, #Network - 1 do --for the number of layers
			activations[i] = {}
			for o = 1, #Network[i] do --for the number of nodes in the current layer
				activations[i][o] = 0
				for p = 1, #Network[i][o] - 1 do --for the number of weights for the current node
					activations[i][o] = activations[i][o] + Network[i][o][p] * activations[i - 1][p]
				end
				activations[i][o] = Activator(activations[i][o] + Network[i][o][#Network[i][o]])
			end
		end
		activations[#Network] = {}
		for o = 1,#Network[#Network] do
			activations[#Network][o] = 0
			for p = 1,#Network[#Network][o] - 1 do
				activations[#Network][o] = activations[#Network][o] + Network[#Network][o][p] * activations[#Network - 1][p]
			end
			activations[#Network][o] = Sigmoid(activations[#Network][o] + Network[#Network][o][#Network[#Network][o]])
		end
		return activations[#activations]
	end
end

function module.Mutate(Network, percentage, gradient)
	local NewNet = {}
	for i = 2,#Network do
		NewNet[i] = {}
		for o = 1, #Network[i] do
			NewNet[i][o] = {}
			for p = 1,#Network[i][o] do
				if rand(0,1000) < percentage * 10 then
					NewNet[i][o][p] = Network[i][o][p] + rand(-1000,1000) / (1000 * gradient)
				else
					NewNet[i][o][p] = Network[i][o][p]
				end
			end
		end
	end
	return NewNet
end

local function dsig(x)
	return Sigmoid(x) * (1 - Sigmoid(x))
end
local function dact(x)
	if x >= 0 then
		return 1
	end
	return 0.1
end

function module.BackPropagate(Network,Inputs,Targets)
	local gradients = {}
	local curgradients = {}
	local neuralnet=Network


	local lr = 0.1 --learning rate(rate that it learns(duh))


	local curerror = {}
	for i = 2, #Network do
		gradients[i] = {}
		curerror[i] = {}
		for o=1,#Network[i] do
			gradients[i][o] = {}
			curerror[i][o] = 0
			for p = 1, #Network[i][o] + 1 do
				gradients[i][o][p] = 0
			end
		end
	end
	curgradients = gradients

	local outlayer = #Network --the layer number of the output layer.
	for i = 1,#Inputs do
		for i = 2,#Network do
			for o = 1,#Network[i] do
				for p = 1,#Network[i][o] do
					curgradients[i][o][p] = 0 --resets the current gradients.
				end
			end
		end

		for o = 2,#Network do
			for p = 1,#Network[o] do
				curerror[o][p] = 0 --resets the current error of each of the neurons.
			end
		end

		local biasgrads = curerror
		local outputs = module.Run(Network, Inputs[i], true) --gives us the inputs(not the outputs) for each neuron

		for o = #Network,2,-1 do
			local layer = Network[o]

			if o ~= #Network then
				for p = 1,#layer do
					for u = 1,#Network[o + 1] do
						curerror[o][p] = curerror[o][p] + Network[o + 1][u][p] * curerror[o + 1][u]
					end
					curerror[o][p] = curerror[o][p] * dact(outputs[2][o][p])
				end
			else --if it is the last layer
				for p = 1,#layer do
					curerror[outlayer][p] = (outputs[1][outlayer][p] - Targets[i][p])* 2 * dsig(outputs[2][outlayer][p])
				end
			end
		end


		--add to the total gradient
		for o = 2,#Network do
			for p = 1,#Network[o] do
				for u = 1,#Network[o][1] - 1 do
					gradients[o][p][u] = gradients[o][p][u] + curerror[o][p] * outputs[1][o - 1][u]
				end
				gradients[o][p][#gradients[o][p]] = gradients[o][p][#gradients[o][p]] + biasgrads[o][p]
			end
		end
	end

	lr= -lr / #Targets

	for o = 2,#Network do
		for p = 1,#Network[o] do
			for u = 1,#Network[o][1] do
				neuralnet[o][p][u] = neuralnet[o][p][u] + gradients[o][p][u]*lr
			end
		end
	end

	return neuralnet
end
return module
