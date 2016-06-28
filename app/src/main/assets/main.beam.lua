-- Torch Android demo script
-- Script: main.lua
-- Copyright (C) 2013 Soumith Chintala
require 'torch'
require 'nn'
require 'nnx'
require 'dok'
require 'image'
require 'sys'

require 'nn'
require 'string'
--require 'hdf5'
require 'nngraph'
torch.setdefaulttensortype('torch.FloatTensor')

function nn.Module:reuseMem()
   self.reuse = true
   return self
end

function nn.Module:setReuse()
   if self.reuse then
      self.gradInput = self.output
   end
end

function make_lstm(data, opt, model, use_chars)
   assert(model == 'enc' or model == 'dec')
   local name = '_' .. model
   local dropout = opt.dropout or 0
   local n = opt.num_layers
   local rnn_size = opt.rnn_size
   local input_size
   if use_chars == 0 then
      input_size = opt.word_vec_size
   else
      input_size = opt.num_kernels
   end
   local offset = 0
  -- there will be 2*n+3 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
   if model == 'dec' then
      table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
      table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
      offset = offset + 2
   end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
     -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
       if use_chars == 0 then
	  local word_vecs
	  if model == 'enc' then
	     word_vecs = nn.LookupTable(data.source_size, input_size)
	  else
	     word_vecs = nn.LookupTable(data.target_size, input_size)
	  end
	  word_vecs.name = 'word_vecs' .. name
	  x = word_vecs(inputs[1]) -- batch_size x word_vec_size
       else
	  local char_vecs = nn.LookupTable(data.char_size, opt.char_vec_size)
	  char_vecs.name = 'word_vecs' .. name
	  local charcnn = make_cnn(opt.char_vec_size,  opt.kernel_width, opt.num_kernels)
	  charcnn.name = 'charcnn' .. name
	  x = charcnn(char_vecs(inputs[1]))
	  if opt.num_highway_layers > 0 then
	     local mlp = make_highway(input_size, opt.num_highway_layers)
	     mlp.name = 'mlp' .. name
	     x = mlp(x)
	  end
       end
       input_size_L = input_size
       if model == 'dec' then
	  x = nn.JoinTable(2)({x, inputs[1+offset]}) -- batch_size x (word_vec_size + rnn_size)
	  input_size_L = input_size + rnn_size
       end
    else
       x = outputs[(L-1)*2]
       if opt.res_net == 1 and L > 2 then
	  x = nn.CAddTable()({x, outputs[(L-2)*2]})
       end
       input_size_L = rnn_size
       if opt.hop_attn == L and model == 'dec' then
	  local hop_attn = make_decoder_attn(data, opt, 1)
	  hop_attn.name = 'hop_attn' .. L
	  x = hop_attn({x, inputs[offset]})
       end
       if dropout > 0 then
	  x = nn.Dropout(dropout, nil, false)(x)
       end
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if model == 'dec' then
     local top_h = outputs[#outputs]
     local decoder_attn = make_decoder_attn(data, opt)
     decoder_attn.name = 'decoder_attn'
     local attn_out = decoder_attn({top_h, inputs[offset]})
     if dropout > 0 then
	attn_out = nn.Dropout(dropout, nil, false)(attn_out)
     end
     table.insert(outputs, attn_out)
  end
  return nn.gModule(inputs, outputs)
end

function make_decoder_attn(data, opt, simple)
   -- 2D tensor target_t (batch_l x rnn_size) and
   -- 3D tensor for context (batch_l x source_l x rnn_size)

   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target_t = nn.LinearNoBias(opt.rnn_size, opt.rnn_size)(inputs[1])
   local context = inputs[2]
   simple = simple or 0
   -- get attention

   local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
   attn = nn.Sum(3)(attn)
   local softmax_attn = nn.SoftMax()
   softmax_attn.name = 'softmax_attn'
   attn = softmax_attn(attn)
   attn = nn.Replicate(1,2)(attn) -- batch_l x  1 x source_l

   -- apply attention to context
   local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   local context_output
   if simple == 0 then
      context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
      context_output = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2,
						 opt.rnn_size)(context_combined))
   else
      context_output = nn.CAddTable()({context_combined,inputs[1]})
   end
   return nn.gModule(inputs, {context_output})
end

function make_generator(data, opt)
   local model = nn.Sequential()
   model:add(nn.Linear(opt.rnn_size, data.target_size))
   model:add(nn.LogSoftMax())
   local w = torch.ones(data.target_size)
   w[1] = 0
   criterion = nn.ClassNLLCriterion(w)
   criterion.sizeAverage = false
   return model, criterion
end


-- cnn Unit
function make_cnn(input_size, kernel_width, num_kernels)
   local output
   local input = nn.Identity()()
   if opt.cudnn == 1 then
      local conv = cudnn.SpatialConvolution(1, num_kernels, input_size,
					    kernel_width, 1, 1, 0)
      local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
      output = nn.Sum(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
   else
      local conv = nn.TemporalConvolution(input_size, num_kernels, kernel_width)
      local conv_layer = conv(input)
      output = nn.Max(2)(nn.Tanh()(conv_layer))
   end
   return nn.gModule({input}, {output})
end

function make_highway(input_size, num_layers, output_size, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)

    local num_layers = num_layers or 1
    local input_size = input_size
    local output_size = output_size or input_size
    local bias = bias or -2
    local f = f or nn.ReLU()
    local start = nn.Identity()()
    local transform_gate, carry_gate, input, output
    for i = 1, num_layers do
       if i > 1 then
	  input_size = output_size
       else
	  input = start
       end
       output = f(nn.Linear(input_size, output_size)(input))
       transform_gate = nn.Sigmoid()(nn.AddConstant(bias, true)(
					nn.Linear(input_size, output_size)(input)))
       carry_gate = nn.AddConstant(1, true)(nn.MulConstant(-1)(transform_gate))
       local proj
       if input_size==output_size then
	  proj = nn.Identity()
       else
	  proj = nn.LinearNoBias(input_size, output_size)
       end
       input = nn.CAddTable()({
	                     nn.CMulTable()({transform_gate, output}),
                             nn.CMulTable()({carry_gate, proj(input)})})
    end
    return nn.gModule({start},{input})
end

--LinearNoBias from elements library
local LinearNoBias, Linear = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
   nn.Module.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   self:reset()
end

function LinearNoBias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end

   return self
end

function LinearNoBias:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      self.output:mv(self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end

--
-- Manages encoder/decoder data matrices.
--

local data = torch.class("data")

function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')

   self.source  = f:read('source'):all()
   self.target  = f:read('target'):all()
   self.target_output = f:read('target_output'):all()
   self.target_l = f:read('target_l'):all() --max target length each batch
   self.target_l_all = f:read('target_l_all'):all()
   self.target_l_all:add(-1)
   self.batch_l = f:read('batch_l'):all()
   self.source_l = f:read('batch_w'):all() --max source length each batch
   if opt.start_symbol == 0 then
      self.source_l:add(-2)
      self.source = self.source[{{},{2, self.source:size(2)-1}}]
   end
   self.batch_idx = f:read('batch_idx'):all()

   self.target_size = f:read('target_size'):all()[1]
   self.source_size = f:read('source_size'):all()[1]
   self.target_nonzeros = f:read('target_nonzeros'):all()

   if opt.use_chars_enc == 1 then
      self.source_char = f:read('source_char'):all()
      self.char_size = f:read('char_size'):all()[1]
      self.char_length = self.source_char:size(3)
   end

   if opt.use_chars_dec == 1 then
      self.target_char = f:read('target_char'):all()
      self.char_size = f:read('char_size'):all()[1]
      self.char_length = self.target_char:size(3)
   end

   self.length = self.batch_l:size(1)
   self.seq_length = self.target:size(2)
   self.batches = {}
   local max_source_l = self.source_l:max()
   local source_l_rev = torch.ones(max_source_l):long()
   for i = 1, max_source_l do
      source_l_rev[i] = max_source_l - i + 1
   end
   for i = 1, self.length do
      local source_i, target_i
      local target_output_i = self.target_output:sub(self.batch_idx[i],self.batch_idx[i]
							+self.batch_l[i]-1, 1, self.target_l[i])
      local target_l_i = self.target_l_all:sub(self.batch_idx[i],
					       self.batch_idx[i]+self.batch_l[i]-1)
      if opt.use_chars_enc == 1 then
	 source_i = self.source_char:sub(self.batch_idx[i],
					      self.batch_idx[i] + self.batch_l[i]-1, 1,
					      self.source_l[i]):transpose(1,2):contiguous()
      else
	 source_i =  self.source:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
				     1, self.source_l[i]):transpose(1,2)
      end
      if opt.reverse_src == 1 then
	 source_i = source_i:index(1, source_l_rev[{{max_source_l-self.source_l[i]+1,
						     max_source_l}}])
      end

      if opt.use_chars_dec == 1 then
	 target_i = self.target_char:sub(self.batch_idx[i],
					      self.batch_idx[i] + self.batch_l[i]-1, 1,
					      self.target_l[i]):transpose(1,2):contiguous()
      else
	 target_i = self.target:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
				    1, self.target_l[i]):transpose(1,2)
      end
      target_l_i = target_l_i:view(self.batch_l[i], 1)
      table.insert(self.batches,  {target_i,
				  target_output_i:transpose(1,2),
				  self.target_nonzeros[i],
				  source_i,
				  self.batch_l[i],
				  self.target_l[i],
				  self.source_l[i],
				  target_l_i})
   end
end

function data:size()
   return self.length
end

function data.__index(self, idx)
   if type(idx) == "string" then
      return data[idx]
   else
      local target_input = self.batches[idx][1]
      local target_output = self.batches[idx][2]
      local nonzeros = self.batches[idx][3]
      local source_input = self.batches[idx][4]
      local batch_l = self.batches[idx][5]
      local target_l = self.batches[idx][6]
      local source_l = self.batches[idx][7]
      local target_l_all = self.batches[idx][8]
      if opt.gpuid >= 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
	 cutorch.setDevice(opt.gpuid)
	 source_input = source_input
	 if opt.gpuid2 >= 0 then
	    cutorch.setDevice(opt.gpuid2)
	 end
	 target_input = target_input
	 target_output = target_output
	 target_l_all = target_l_all
      end
      return {target_input, target_output, nonzeros, source_input,
	      batch_l, target_l, source_l, target_l_all}
   end
end

-- return data



-- require 'models.lua'
-- require 'data.lua'
-- require 'util.lua'
-- require 'cudnn'
-- stringx = require('pl.stringx')

cmd = torch.CmdLine()

-- file location
cmd:option('-model', '/sdcard/en-de-2-300_final.ascii.t7', [[Path to model .t7 file]])
cmd:option('-src_file', '/sdcard/en-val-clean.txt',
	   [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', '/sdcard/pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', '/sdcard//en.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', '/sdcard//de.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character
                                                vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 5,[[Beam size]])
cmd:option('-batch', 10,[[Max decoding batch size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer
                               than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all
                         hypotheses that have been generated so far that ends with end-of-sentence
                         token and takes the highest scoring of all of them.]])
cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If srctarg_dict is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK
                             tokens. See README.md for the format this file should be in]])
cmd:option('-score_gold', 1, [[If = 1, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])

cmd:option('-gpuid',  -1,[[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1,[[Second GPU ID]])
cmd:option('-unk_penalty', 0)
cmd:option('-length_restriction', 0)
opt = cmd:parse({})


function copy(orig)
   local orig_type = type(orig)
   local copy
   if orig_type == 'table' then
      copy = {}
      for orig_key, orig_value in pairs(orig) do
         copy[orig_key] = orig_value
      end
   else
      copy = orig
   end
   return copy
end

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
   return {start}
end

function StateAll.advance(state, token)
   local new_state = copy(state)
   table.insert(new_state, token)
   return new_state
end

function StateAll.disallow(out)
   local bad = {1, 3} -- 1 is PAD, 3 is BOS
   for j = 1, #bad do
      out[bad[j]] = -1e9
   end
end

function StateAll.same(state1, state2)
   for i = 2, #state1 do
      if state1[i] ~= state2[i] then
         return false
      end
   end
   return true
end

function StateAll.next(state)
   return state[#state]
end

function StateAll.heuristic(state)
   return 0
end

function StateAll.print(state)
   for i = 1, #state do
      io.write(state[i] .. " ")
   end
   print()
end


-- Convert a flat index to a row-column tuple.
function flat_to_rc(v, flat_index)
   local row = math.floor((flat_index - 1) / v:size(2)) + 1
   return row, (flat_index - 1) % v:size(2) + 1
end

function generate_beam(model, initial, K, max_sent_l, source, gold, batch)
   --reset decoder initial states
   if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
   end
   local long_source = 0
   for i = 1, batch do
      long_source = math.max(long_source, source[i]:size(1))
   end

   local n
   local source_l = math.min(long_source, opt.max_sent_l)
   -- local source_l = math.min(source:size(1), opt.max_sent_l)
   local attn_argmax = {}   -- store attn weights
   if opt.length_restriction == 0 then
      n = max_sent_l
   else
      n = source_l + 5
   end

  -- Backpointer table.
   local prev_ks = torch.LongTensor(batch, n, K):fill(1)
   -- Current States.
   local next_ys = torch.LongTensor(batch, n, K):fill(1)
   -- Current Scores.
   local scores = torch.FloatTensor(batch, n, K)
   scores:zero()

   -- attn_argmax[1] = {}

   local states = {} -- store predicted word idx

   for b = 1, batch do
      states[b] = {}
      states[b][1] = {}
      for k = 1, 1 do
         table.insert(states[b][1], initial)
         -- table.insert(attn_argmax[][1], initial)
         next_ys[b][1][k] = State.next(initial)
      end
   end

      source_input = torch.LongTensor(long_source, batch):fill(1)
   for i = 1, batch do
      local size = source[i]:size(1)
      -- print(long_source-size+1)
      -- print(size)
      source_input[{{}, i}]:narrow(1, long_source-size+1, size):copy(source[i])
   end
   -- source_input = source:view(source_l, 1):expand(source_l, opt.batch):clone()
   -- print(source_input:size())
   local rnn_state_enc = {}

   local rnn_size = init_fwd_enc[1]:size(2)
   for i = 1, #init_fwd_enc do
      init_fwd_enc[i]:resize(batch, rnn_size)
      table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
   end
   local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
   context = context:resize(batch, source_l, rnn_size):view(1, batch, source_l, rnn_size)
   -- print(context:size())

   -- print("here")
   sys.tic()
   for t = 1, source_l do
      local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
      -- print("encoder", encoder_input)
      -- print(source_input[t])
      local out = model[1]:forward(encoder_input)
      -- print(out)
      for j = 1, batch do
         if t <= long_source - source[j]:size(1)  then
            for k = 1, #out do
               out[k][j]:zero()
            end
         end
      end

      rnn_state_enc = out
      -- print(rnn_state_enc)
      -- print(context[{{}, t}]:size(), t)
      context[{{}, {} ,t}]:copy(out[#out])

   end

   -- local source_input
   -- if model_opt.use_chars_enc == 1 then
   --    source_input = source:view(source_l, 1, source:size(2)):contiguous()
   -- else
   --    source_input = source:view(source_l, 1)
   -- end

   -- local rnn_state_enc = {}
   -- for i = 1, #init_fwd_enc do
   --    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
   -- end
   -- local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size

   -- for t = 1, source_l do
   --    local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
   --    local out = model[1]:forward(encoder_input)
   --    rnn_state_enc = out
   --    context[{{},t}]:copy(out[#out])
   -- end
   context = context:expand(K, batch, source_l, model_opt.rnn_size):contiguous():view(K*batch, source_l, model_opt.rnn_size)

   if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, K}, {1, source_l}}]
      context2:copy(context)
      context = context2
   end

   rnn_state_dec = {}
   for i = 1, #init_fwd_dec do
      init_fwd_dec[i]:resize(K* batch, rnn_size)
      table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
   end
   -- print(rnn_size_enc:size())
   if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
         -- print(rnn_state_enc[L*2-1]:size())
         -- print(rnn_state_dec[L*2-1]:size())

         rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1]:view(1, batch, model_opt.rnn_size): expand(K, batch, model_opt.rnn_size):contiguous():view(K*batch, model_opt.rnn_size))
         rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2]:view(1, batch, model_opt.rnn_size):expand(K, batch, model_opt.rnn_size):contiguous():view(K*batch, model_opt.rnn_size))
      end
   end
   out_float = torch.FloatTensor()

   local i = 1
   local done = {}
   local max_score = {}
   local max_hyp = {}
   local found_eos = {}

   local end_hyp = {}
   local end_score = {}
   local possible_hyp = {}
   for b = 1, batch do
      max_score[b] = -1e9
      max_hyp[b] = -1e9
      end_hyp[b] = {}
      end_score[b] = -1e9
      done[b] = false
      found_eos[b] = false
      possible_hyp[b] = false
   end
   local all_done = false
   while (not all_done) and (i < n) do
      i = i+1
      local decoder_input1 = torch.LongTensor(K, batch):zero()
      for b = 1, batch do
         states[b][i] = {}
         --attn_argmax[b][i] = {}


         if model_opt.use_chars_dec == 1 then
            decoder_input1[{{}, b}]:copy(word2charidx_targ:index(1, next_ys[b]:narrow(1,i-1,1):squeeze()))
         else
            decoder_input1[{{}, b}]:copy(next_ys[b]:narrow(1,i-1,1):squeeze())
            if opt.beam == 1 then
               decoder_input1[{{},b}]:copy(torch.LongTensor({decoder_input1}))
            end
         end
      end
      -- print(K, batch)
      decoder_input1 = decoder_input1:view(K * batch)

      local decoder_input = {decoder_input1, context, table.unpack(rnn_state_dec)}

      -- print("A")
      local out_decoder = model[2]:forward(decoder_input)

      -- print("B")
      local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      out = out:view(K, batch, out:size(2)):transpose(1, 2)

      rnn_state_dec = {} -- to be modified later
      table.insert(rnn_state_dec, out_decoder[#out_decoder])
      for j = 1, #out_decoder - 1 do
	 table.insert(rnn_state_dec, out_decoder[j])
      end
      -- print("C")
      out_float:resize(out:size()):copy(out)

      for b = 1, batch do
         for k = 1, K do
            -- State.disallow(out_float[b]:select(1, k))
            --	 if opt.unk_penalty == 1 and next_ys[-1][k] == UNK then
            --	    out_float[k][UNK] = -1e9
            --	 end
            out_float[b][k]:add(scores[b][i-1][k])
         end
         -- All the scores available.

         local flat_out = out_float[b]:view(-1)
         if i == 2 then
            flat_out = out_float[b][1] -- all outputs same for first batch
         end
         if model_opt.start_symbol == 1 then
            decoder_softmax.output[{{},1}]:zero()
            decoder_softmax.output[{{},source_l}]:zero()
         end
         for k = 1, K do
            local score, index = flat_out:max(1)
            local score = score[1]
            local prev_k, y_i = flat_to_rc(out_float[b], index[1])
            states[b][i][k] = State.advance(states[b][i-1][prev_k], y_i)
            max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
            prev_ks[b][i][k] = prev_k
            next_ys[b][i][k] = y_i
            scores[b][i][k] = score
            flat_out[index[1]] = -1e9
         end

         for j = 1, #rnn_state_dec do
            v = rnn_state_dec[j]:view(K, batch, rnn_size):contiguous()
            -- print(v:size(), prev_ks[b][i])
            -- print(v[{{}, b}]:index(1, prev_ks[b][i]):size())
            q = v[{{}, b}]
            q2 = q:contiguous():index(1, prev_ks[b][i]):clone()
            v[{{}, b}]:copy(q)
         end
         end_hyp[b] = states[b][i][1]
         end_score[b] = scores[b][i][1]
         if end_hyp[b][#end_hyp[b]] == END then
            done[b] = true
            found_eos[b] = true
         else
            for k = 1, K do
               possible_hyp[b] = states[b][i][k]
               if possible_hyp[b][#possible_hyp[b]] == END then
                  found_eos[b] = true
                  if scores[b][i][k] > max_score[b] then
                     max_hyp[b] = possible_hyp[b]
                     max_score[b] = scores[b][i][k]
                     -- max_attn_argmax = attn_argmax[b][i][k]
                  end
               end
            end
         end
      end
      all_done = true
      for b = 1, batch do
         if not done[b] then
            all_done = false
         end
      end
   end
   -- local gold_score = 0
   -- if opt.score_gold == 1 then
   --    rnn_state_dec = {}
   --    for i = 1, #init_fwd_dec do
   --   table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
   --    end
   --    if model_opt.init_dec == 1 then
   --   for L = 1, model_opt.num_layers do
   --      rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1][{{1}}])
   --      rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2][{{1}}])
   --   end
   --    end
   --    local target_l = gold:size(1)
   --    for t = 2, target_l do
   --   local decoder_input1
   --   if model_opt.use_chars_dec == 1 then
   --      decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
   --   else
   --      decoder_input1 = gold[{{t-1}}]
   --   end
   --   local decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
   --   local out_decoder = model[2]:forward(decoder_input)
   --   local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
   --   rnn_state_dec = {} -- to be modified later
   --   table.insert(rnn_state_dec, out_decoder[#out_decoder])
   --   for j = 1, #out_decoder - 1 do
   --      table.insert(rnn_state_dec, out_decoder[j])
   --   end
   --   gold_score = gold_score + out[1][gold[t]]

   --    end
   -- end
   for b = 1, batch do
      if opt.simple == 1 or end_score[b] > max_score[b] or not found_eos[b] then
         max_hyp[b] = end_hyp[b]
         max_score[b] = end_score[b]
         -- max_attn_argmax = end_attn_argmax
      end
   end
   return max_hyp
   --return max_hyp, max_score, max_attn_argmax, gold_score, states[i], scores[i], attn_argmax[i]
end

function idx2key(file)
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
	 table.insert(c, w)
      end
      t[tonumber(c[2])] = c[1]
   end
   return t
end

function flip_table(u)
   local t = {}
   for key, value in pairs(u) do
      t[value] = key
   end
   return t
end


function get_layer(layer)
   if layer.name ~= nil then
      if layer.name == 'decoder_attn' then
	 decoder_attn = layer
      elseif layer.name:sub(1,3) == 'hop' then
	 hop_attn = layer
      elseif layer.name:sub(1,7) == 'softmax' then
	 table.insert(softmax_layers, layer)
      elseif layer.name == 'word_vecs_enc' then
	 word_vecs_enc = layer
      elseif layer.name == 'word_vecs_dec' then
	 word_vecs_dec = layer
      end
   end
end

function sent2wordidx(sent, word2idx, start_symbol)
   local t = {}
   local u = {}
   if start_symbol == 1 then
      table.insert(t, START)
      table.insert(u, START_WORD)
   end

   for word in sent:gmatch'([^%s]+)' do
      local idx = word2idx[word] or UNK
      table.insert(t, idx)
      table.insert(u, word)
   end
   if start_symbol == 1 then
      table.insert(t, END)
      table.insert(u, END_WORD)
   end
   return torch.LongTensor(t), u
end

function sent2charidx(sent, char2idx, max_word_l, start_symbol)
   local words = {}
   if start_symbol == 1 then
      table.insert(START_WORD)
   end
   for word in sent:gmatch'([^%s]+)' do
      table.insert(words, word)
   end
   if start_symbol == 1 then
      table.insert(END_WORD)
   end
   local chars = torch.ones(#words, max_word_l)
   for i = 1, #words do
      chars[i] = word2charidx(words[i], char2idx, max_word_l, chars[i])
   end
   return chars, words
end

function word2charidx(word, char2idx, max_word_l, t)
   t[1] = START
   local i = 2
   for _, char in utf8.next, word do
      char = utf8.char(char)
      local char_idx = char2idx[char] or UNK
      t[i] = char_idx
      i = i+1
      if i >= max_word_l then
	 t[i] = END
	 break
      end
   end
   if i < max_word_l then
      t[i] = END
   end
   return t
end

function wordidx2sent(sent, idx2word, source_str, attn, skip_end)
   local t = {}
   local start_i, end_i
   skip_end = skip_start_end or true
   if skip_end then
      end_i = #sent-1
   else
      end_i = #sent
   end
   for i = 2, end_i do -- skip START and END
      if sent[i] == UNK then
	 if opt.replace_unk == 1 then
	    local s = source_str[attn[i]]
	    if phrase_table[s] ~= nil then
	       print(s .. ':' ..phrase_table[s])
	    end
	    local r = phrase_table[s] or s
	    table.insert(t, r)
	 else
	    table.insert(t, idx2word[sent[i]])
	 end
      else
	 table.insert(t, idx2word[sent[i]])
      end
   end
   return table.concat(t, ' ')
end

function clean_sent(sent)
   local s = stringx.replace(sent, UNK_WORD, '')
   s = stringx.replace(s, START_WORD, '')
   s = stringx.replace(s, END_WORD, '')
   s = stringx.replace(s, START_CHAR, '')
   s = stringx.replace(s, END_CHAR, '')
   return s
end

function strip(s)
   return s:gsub("^%s+",""):gsub("%s+$","")
end


function main()
   -- some globals
   PAD = 1; UNK = 2; START = 3; END = 4
   PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
   START_CHAR = '{'; END_CHAR = '}'
   MAX_SENT_L = opt.max_sent_l
   -- assert(path.exists(opt.src_file), 'src_file does not exist')
   -- assert(path.exists(opt.model), 'model does not exist')

   -- parse input params
   -- opt = cmd:parse(arg)
   if opt.gpuid >= 0 then
      require 'cutorch'
      require 'cunn'
   end
   print('loading ' .. opt.model .. '...')
   checkpoint = torch.load(opt.model, 'ascii')
   print('done!')

   if opt.replace_unk == 1 then
      phrase_table = {}
     --  if path.exists(opt.srctarg_dict) then
	 -- local f = io.open(opt.srctarg_dict,'r')
	 -- for line in f:lines() do
	 --    local c = line:split("|||")
	 --    phrase_table[strip(c[1])] = c[2]
	 -- end
     --  end
   end

   -- load model and word2idx/idx2word dictionaries
   model, model_opt = checkpoint[1], checkpoint[2]
   --   print(model_opt)
   if opt.unk_penalty ~= 0 then
      print(model[3].modules[1].bias[2])
      model[3].modules[1].bias[2] = model[3].modules[1].bias[2] + opt.unk_penalty
   end

   -- if model_opt.cudnn == 1 then
   --     require 'cudnn'
   --  end

   idx2word_src = idx2key(opt.src_dict)
   word2idx_src = flip_table(idx2word_src)
   idx2word_targ = idx2key(opt.targ_dict)
   word2idx_targ = flip_table(idx2word_targ)

   -- load character dictionaries if needed
   -- if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
   --    utf8 = require 'lua-utf8'
   --    char2idx = flip_table(idx2key(opt.char_dict))
   --    model[1]:apply(get_layer)
   -- end
   -- if model_opt.use_chars_dec == 1 then
   --    word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
   --    for i = 1, #idx2word_targ do
   --   word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
   --  				     model_opt.max_word_l, word2charidx_targ[i])
   --    end
   -- end

   -- -- load gold labels if it exists
   -- if path.exists(opt.targ_file) then
   --    print('loading GOLD labels at ' .. opt.targ_file)
   --    gold = {}
   --    local file = io.open(opt.targ_file, 'r')
   --    for line in file:lines() do
   --   table.insert(gold, line)
   --    end
   -- else
      opt.score_gold = 0
   -- end

   -- if opt.gpuid >= 0 then
   --    cutorch.setDevice(opt.gpuid)
   --    for i = 1, #model do
   --   if opt.gpuid2 >= 0 then
   --      if i == 1 then
   --         cutorch.setDevice(opt.gpuid)
   --      else
   --         cutorch.setDevice(opt.gpuid2)
   --      end
   --   end
   --   model[i]:double():cuda()
   --   model[i]:evaluate()
   --    end
   -- end

   softmax_layers = {}
   model[2]:apply(get_layer)
   decoder_attn:apply(get_layer)
   decoder_softmax = softmax_layers[1]
   attn_layer = torch.zeros(opt.beam, MAX_SENT_L)

   context_proto = torch.zeros(opt.batch, MAX_SENT_L, model_opt.rnn_size)
   -- local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
   local h_init_dec = torch.zeros(opt.batch, model_opt.rnn_size)

   local h_init_enc = torch.zeros(opt.batch, model_opt.rnn_size)
   if opt.gpuid >= 0 then
      h_init_enc = h_init_enc
      h_init_dec = h_init_dec
      cutorch.setDevice(opt.gpuid)
      if opt.gpuid2 >= 0 then
	 cutorch.setDevice(opt.gpuid)
	 context_proto = context_proto
	 cutorch.setDevice(opt.gpuid2)
	 context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.rnn_size)
      else
	 context_proto = context_proto
      end
      attn_layer = attn_layer
   end
   init_fwd_enc = {}
   init_fwd_dec = {h_init_dec:clone()} -- initial context
   for L = 1, model_opt.num_layers do
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
      table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
   end

   pred_score_total = 0
   gold_score_total = 0
   pred_words_total = 0
   gold_words_total = 0
   src_words_total = 0
   -- target_input = torch.LongTensor({1})
   target_input = torch.LongTensor(opt.batch):fill(1)
   local sent_id = 0
   pred_sents = {}
   local file = io.open(opt.src_file, "r")
   local out_file = io.open(opt.output_file,'w')
   timer = torch.Timer()
   local start_time = timer:time().real
   local sources = {}
   local targets = {}
   local source_len = {}
   local pred_sents = {}
   for line in file:lines() do
      sent_id = sent_id + 1
      --line = clean_sent(line)
      -- print('SENT ' .. sent_id .. ': ' ..line)
      local source, source_str
      if model_opt.use_chars_enc == 0 then
         source, source_str = sent2wordidx(line, word2idx_src, model_opt.start_symbol)

      else
         source, source_str = sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
      end
      if source:dim() > 0 and source:size(1) > 3 then

         src_words_total = src_words_total + source:size(1)

         if opt.score_gold == 1 then
            target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
            table.insert(targets, target)
         else
            table.insert(targets, {})
         end
         table.insert(sources, {source:size(1), sent_id, source, line})
         table.insert(pred_sents, "")
      else
         sent_id = sent_id - 1
      end
      if sent_id > 1000 then
         break
      end
   end
   table.sort(sources, function (a,b) return a[1] < b[1]; end)

   local i = 1
   while true and i <= #sources do
      -- for i = 1, #sources, opt.batch  do

      local to_decode = {}
      local to_decode_targ = {}
      local lines = {}
      local batch = 0
      local size = -1
      for j = i, i + opt.batch - 1 do
         local source_tab = sources[j]
         if j > #sources then
            break
         else
            -- print(source_tab[3]:size())
            -- print(size, source_tab[1])
            if size ~= -1 and size ~= source_tab[1] then
               break
            end

            table.insert(to_decode, source_tab[3])
            table.insert(lines, source_tab[4])
            table.insert(to_decode_targ, targets[source_tab[2]])
            size = source_tab[1]
            batch = batch + 1
         end
         -- local target = targets[i]
      end
      -- print(to_decode)
      local full_time = timer:time().real
      local start_time2 = timer:time().real
      state = State.initial(START)
      pred = generate_beam(model, state, opt.beam, MAX_SENT_L, to_decode, to_decode_targ, batch)
      -- print(pred)
      print("beam", timer:time().real - start_time2)

      for j = 1, batch do
         local source = to_decode[j]
         pred_sent = wordidx2sent(pred[j], idx2word_targ, source_str, attn, true)

         print('SENT ' .. i+j-1 .. ': ' ..lines[j])
         print('PRED ' .. i+j-1 .. ': ' .. pred_sent)
         if i+j-1 <= #sources then
            pred_sents[sources[i+j-1][2]] = pred_sent
         end
         -- print('PRED ' .. i+j-1 .. ': ', pred[j])
         if gold ~= nil then
            print('GOLD ' .. i .. ': ' .. gold[sent_id])
            if opt.score_gold == 1 then
               print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
               gold_score_total = gold_score_total + gold_score
               gold_words_total = gold_words_total + target:size(1) - 1
            end
         end
      end
      i = i + batch
      print('')
      time_taken2 = timer:time().real - full_time
      print(time_taken2)
      time_taken = timer:time().real - start_time
      print(time_taken)
      -- if sent_id > 100 then
      --    break
      -- end
   end
   for i = 1, #sources do
      out_file:write(pred_sents[i] .. '\n')
   end
   print(string.format("WORDS/SEC: %.2f, SENTS/SEC: %.2f", src_words_total/time_taken, sent_id/time_taken))
   out_file:close()
end
State = StateAll
main()


-- function generate_beam(model, initial, K, max_sent_l, source, gold, batch)
--    local long_source = 0
--    for i = 1, batch do
--       long_source = math.max(long_source, source[i]:size(1))
--    end
--    -- print(long_source)
--    --reset decoder initial states
--    local n
--    local source_l = math.min(long_source, opt.max_sent_l)
--    if opt.length_restriction == 0 then
--       n = max_sent_l
--    else
--       n = source_l + 5
--    end
--    source_input = torch.LongTensor(long_source, batch):fill(1)
--    for i = 1, batch do
--       local size = source[i]:size(1)
--       -- print(long_source-size+1)
--       -- print(size)
--       source_input[{{}, i}]:narrow(1, long_source-size+1, size):copy(source[i])
--    end
--    -- source_input = source:view(source_l, 1):expand(source_l, opt.batch):clone()
--    -- print(source_input:size())
--    local rnn_state_enc = {}

--    local rnn_size = init_fwd_enc[1]:size(2)
--    for i = 1, #init_fwd_enc do
--       init_fwd_enc[i]:resize(batch, rnn_size)
--       table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
--    end
--    local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
--    context:resize(batch, source_l, rnn_size)
--    -- print(context:size())

--    -- print("here")
--    sys.tic()
--    for t = 1, source_l do
--       local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
--       -- print("encoder", encoder_input)
--       -- print(source_input[t])
--       local out = model[1]:forward(encoder_input)
--       -- print(out)
--       for j = 1, batch do
--          if t <= long_source - source[j]:size(1)  then
--             for k = 1, #out do
--                out[k][j]:zero()
--             end
--          end
--       end

--       rnn_state_enc = out
--       -- print(rnn_state_enc)
--       -- print(context[{{}, t}]:size(), t)
--       context[{{},t}]:copy(out[#out])

--    end
--    -- print("Enc", sys.toc())


--    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
--       cutorch.setDevice(opt.gpuid2)
--       local context2 = context_proto2[{{1, K}, {1, source_l}}]
--       context2:copy(context)
--       context = context2
--    end

--    rnn_state_dec = {}
--    for i = 1, #init_fwd_dec do
--       init_fwd_dec[i]:resize(batch, rnn_size)
--       table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
--    end
--    -- print("Starting beam")
--    if model_opt.init_dec == 1 then
--       for L = 1, model_opt.num_layers do
-- 	 rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1])
-- 	 rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2])
--       end
--    end

--    print("Starting beam2")

--    local i = 1
--    local done = false
--    local found_eos = false
--    local pred_targ = {}
--    target_input = torch.LongTensor(batch):fill(1)
--    for i = 1, target_input:size(1) do
--       target_input[i] = START
--       table.insert(pred_targ, {START})
--    end

--    sys.tic()
--    local done = torch.LongTensor(batch):zero()
--    local all_done = false
--    while (not all_done) and (i < n) do
--       i = i+1
--       local decoder_input = {target_input, context, table.unpack(rnn_state_dec)}
--       -- print(decoder_input)

--       local out_decoder = model[2]:forward(decoder_input)
--       local out = model[3].modules[1]:forward(out_decoder[#out_decoder]) -- K x vocab_size
--       _, q = out:max(2)
--       for i = 1, target_input:size(1) do
--          if done[i] == 0 then
--             table.insert(pred_targ[i], q[i][1])
--          end
--          target_input[i] = q[i][1]
--          if q[i][1] == END then
--             done[i] = 1
--          end
--       end
--       if done:sum() == batch then
--          all_done = true
--       end
--       rnn_state_dec = {} -- to be modified later
--       table.insert(rnn_state_dec, out_decoder[#out_decoder])
--       for j = 1, #out_decoder - 1 do
--          table.insert(rnn_state_dec, out_decoder[j])
--       end
--    end
--    print("END", sys.toc())
--    return pred_targ
-- end

-- function idx2key(file)
--    local f = io.open(file,'r')
--    local t = {}
--    for line in f:lines() do
--       local c = {}
--       for w in line:gmatch'([^%s]+)' do
-- 	 table.insert(c, w)
--       end
--       t[tonumber(c[2])] = c[1]
--    end
--    return t
-- end

-- function flip_table(u)
--    local t = {}
--    for key, value in pairs(u) do
--       t[value] = key
--    end
--    return t
-- end


-- function get_layer(layer)
--    if layer.name ~= nil then
--       if layer.name == 'decoder_attn' then
-- 	 decoder_attn = layer
--       elseif layer.name:sub(1,3) == 'hop' then
-- 	 hop_attn = layer
--       elseif layer.name:sub(1,7) == 'softmax' then
-- 	 table.insert(softmax_layers, layer)
--       elseif layer.name == 'word_vecs_enc' then
-- 	 word_vecs_enc = layer
--       elseif layer.name == 'word_vecs_dec' then
-- 	 word_vecs_dec = layer
--       end
--    end
-- end

-- function sent2wordidx(sent, word2idx, start_symbol)
--    local t = {}
--    local u = {}
--    if start_symbol == 1 then
--       table.insert(t, START)
--       table.insert(u, START_WORD)
--    end

--    for word in sent:gmatch'([^%s]+)' do
--       local idx = word2idx[word] or UNK
--       table.insert(t, idx)
--       table.insert(u, word)
--    end
--    if start_symbol == 1 then
--       table.insert(t, END)
--       table.insert(u, END_WORD)
--    end
--    return torch.LongTensor(t), u
-- end

-- function sent2charidx(sent, char2idx, max_word_l, start_symbol)
--    local words = {}
--    if start_symbol == 1 then
--       table.insert(START_WORD)
--    end
--    for word in sent:gmatch'([^%s]+)' do
--       table.insert(words, word)
--    end
--    if start_symbol == 1 then
--       table.insert(END_WORD)
--    end
--    local chars = torch.ones(#words, max_word_l)
--    for i = 1, #words do
--       chars[i] = word2charidx(words[i], char2idx, max_word_l, chars[i])
--    end
--    return chars, words
-- end

-- function word2charidx(word, char2idx, max_word_l, t)
--    t[1] = START
--    local i = 2
--    for _, char in utf8.next, word do
--       char = utf8.char(char)
--       local char_idx = char2idx[char] or UNK
--       t[i] = char_idx
--       i = i+1
--       if i >= max_word_l then
-- 	 t[i] = END
-- 	 break
--       end
--    end
--    if i < max_word_l then
--       t[i] = END
--    end
--    return t
-- end

-- function wordidx2sent(sent, idx2word, source_str, attn, skip_end)
--    local t = {}
--    local start_i, end_i
--    skip_end = skip_start_end or true
--    if skip_end then
--       end_i = #sent-1
--    else
--       end_i = #sent
--    end
--    for i = 2, end_i do -- skip START and END
--       if sent[i] == UNK then
-- 	 if opt.replace_unk == 1 then
-- 	    local s = source_str[attn[i]]
-- 	    if phrase_table[s] ~= nil then
-- 	       print(s .. ':' ..phrase_table[s])
-- 	    end
-- 	    local r = phrase_table[s] or s
-- 	    table.insert(t, r)
-- 	 else
-- 	    table.insert(t, idx2word[sent[i]])
-- 	 end
--       else
-- 	 table.insert(t, idx2word[sent[i]])
--       end
--    end
--    return table.concat(t, ' ')
-- end

-- function clean_sent(sent)
--    local s = stringx.replace(sent, UNK_WORD, '')
--    s = stringx.replace(s, START_WORD, '')
--    s = stringx.replace(s, END_WORD, '')
--    s = stringx.replace(s, START_CHAR, '')
--    s = stringx.replace(s, END_CHAR, '')
--    return s
-- end

-- function strip(s)
--    return s:gsub("^%s+",""):gsub("%s+$","")
-- end

-- function main()
--    -- some globals
--    PAD = 1; UNK = 2; START = 3; END = 4
--    PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
--    START_CHAR = '{'; END_CHAR = '}'
--    MAX_SENT_L = opt.max_sent_l
--    -- assert(path.exists(opt.src_file), 'src_file does not exist')
--    -- assert(path.exists(opt.model), 'model does not exist')

--    -- parse input params
--    -- opt = cmd:parse(arg)
--    if opt.gpuid >= 0 then
--       require 'cutorch'
--       require 'cunn'
--    end
--    print('loading ' .. opt.model .. '...')
--    checkpoint = torch.load(opt.model, 'ascii')
--    print('done!')

--    if opt.replace_unk == 1 then
--       phrase_table = {}
--       -- if path.exists(opt.srctarg_dict) then
--       --    local f = io.open(opt.srctarg_dict,'r')
--       --    for line in f:lines() do
--       --       local c = line:split("|||")
--       --       phrase_table[strip(c[1])] = c[2]
--       --    end
--       -- end
--    end

--    -- load model and word2idx/idx2word dictionaries
--    model, model_opt = checkpoint[1], checkpoint[2]
--    --   print(model_opt)
--    if opt.unk_penalty ~= 0 then
--       print(model[3].modules[1].bias[2])
--       model[3].modules[1].bias[2] = model[3].modules[1].bias[2] + opt.unk_penalty
--    end

--    -- if model_opt.cudnn == 1 then
--    --     require 'cudnn'
--    --  end

--    idx2word_src = idx2key(opt.src_dict)
--    word2idx_src = flip_table(idx2word_src)
--    idx2word_targ = idx2key(opt.targ_dict)
--    word2idx_targ = flip_table(idx2word_targ)

--    -- load character dictionaries if needed
--    -- if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
--    --    utf8 = require 'lua-utf8'
--    --    char2idx = flip_table(idx2key(opt.char_dict))
--    --    model[1]:apply(get_layer)
--    -- end
--    -- if model_opt.use_chars_dec == 1 then
--    --    word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
--    --    for i = 1, #idx2word_targ do
--    --   word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
--    --  				     model_opt.max_word_l, word2charidx_targ[i])
--    --    end
--    -- end

--    -- load gold labels if it exists
--    -- if path.exists(opt.targ_file) then
--    --    print('loading GOLD labels at ' .. opt.targ_file)
--    --    gold = {}
--    --    local file = io.open(opt.targ_file, 'r')
--    --    for line in file:lines() do
--    --   table.insert(gold, line)
--    --    end
--    -- else
--       opt.score_gold = 0
--    -- end

--    -- if opt.gpuid >= 0 then
--    --    cutorch.setDevice(opt.gpuid)
--    --    for i = 1, #model do
--    --   if opt.gpuid2 >= 0 then
--    --      if i == 1 then
--    --         cutorch.setDevice(opt.gpuid)
--    --      else
--    --         cutorch.setDevice(opt.gpuid2)
--    --      end
--    --   end
--    --   model[i]:double()
--    --   model[i]:evaluate()
--    --    end
--    -- end

--    softmax_layers = {}
--    model[2]:apply(get_layer)
--    decoder_attn:apply(get_layer)
--    decoder_softmax = softmax_layers[1]
--    attn_layer = torch.zeros(opt.beam, MAX_SENT_L)

--    context_proto = torch.zeros(opt.batch, MAX_SENT_L, model_opt.rnn_size)
--    -- local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
--    local h_init_dec = torch.zeros(opt.batch, model_opt.rnn_size)

--    local h_init_enc = torch.zeros(opt.batch, model_opt.rnn_size)
--    if opt.gpuid >= 0 then
--       h_init_enc = h_init_enc
--       h_init_dec = h_init_dec
--       cutorch.setDevice(opt.gpuid)
--       if opt.gpuid2 >= 0 then
-- 	 cutorch.setDevice(opt.gpuid)
-- 	 context_proto = context_proto
-- 	 cutorch.setDevice(opt.gpuid2)
-- 	 context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.rnn_size)
--       else
-- 	 context_proto = context_proto
--       end
--       attn_layer = attn_layer
--    end
--    init_fwd_enc = {}
--    init_fwd_dec = {h_init_dec:clone()} -- initial context
--    for L = 1, model_opt.num_layers do
--       table.insert(init_fwd_enc, h_init_enc:clone())
--       table.insert(init_fwd_enc, h_init_enc:clone())
--       table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
--       table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
--    end

--    pred_score_total = 0
--    gold_score_total = 0
--    pred_words_total = 0
--    gold_words_total = 0
--    src_words_total = 0
--    -- target_input = torch.LongTensor({1})
--    target_input = torch.LongTensor(opt.batch):fill(1)
--    local sent_id = 0
--    pred_sents = {}
--    local file = io.open(opt.src_file, "r")
--    local out_file = io.open(opt.output_file,'w')
--    timer = torch.Timer()
--    local start_time = timer:time().real
--    local sources = {}
--    local targets = {}
--    local source_len = {}
--    local pred_sents = {}
--    for line in file:lines() do
--       sent_id = sent_id + 1
--       --line = clean_sent(line)
--       -- print('SENT ' .. sent_id .. ': ' ..line)
--       local source, source_str
--       if model_opt.use_chars_enc == 0 then
--          source, source_str = sent2wordidx(line, word2idx_src, model_opt.start_symbol)

--       else
--          source, source_str = sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
--       end
--       if source:dim() > 0 then

--          src_words_total = src_words_total + source:size(1)

--          if opt.score_gold == 1 then
--             target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
--             table.insert(targets, target)
--          else
--             table.insert(targets, {})
--          end
--          table.insert(sources, {source:size(1), sent_id, source, line})
--          table.insert(pred_sents, "")
--       else
--          sent_id = sent_id - 1
--       end
--       if sent_id > 200 then
--          break
--       end
--    end
--    table.sort(sources, function (a,b) return a[1] < b[1]; end)

--    local i = 1
--    while true and i <= #sources do
--       -- for i = 1, #sources, opt.batch  do

--       local to_decode = {}
--       local to_decode_targ = {}
--       local lines = {}
--       local batch = 0
--       local size = -1
--       for j = i, i + opt.batch - 1 do
--          local source_tab = sources[j]
--          if j > #sources then
--             break
--          else
--             -- print(source_tab[3]:size())
--             print(size, source_tab[1])
--             if size ~= -1 and size ~= source_tab[1] then
--                break
--             end

--             table.insert(to_decode, source_tab[3])
--             table.insert(lines, source_tab[4])
--             table.insert(to_decode_targ, targets[source_tab[2]])
--             size = source_tab[1]
--             batch = batch + 1
--          end
--          -- local target = targets[i]
--       end
--       print(to_decode)
--       local full_time = timer:time().real
--       local start_time2 = timer:time().real
--       pred = generate_beam(model, state, opt.beam, MAX_SENT_L, to_decode, to_decode_targ, batch)
--       print("beam", timer:time().real - start_time2)

--       for j = 1, batch do
--          local source = to_decode[j]
--          pred_sent = wordidx2sent(pred[j], idx2word_targ, source_str, attn, true)

--          print('SENT ' .. i+j-1 .. ': ' ..lines[j])
--          print('PRED ' .. i+j-1 .. ': ' .. pred_sent)
--          if i+j-1 <= #sources then
--             pred_sents[sources[i+j-1][2]] = pred_sent
--          end
--          -- print('PRED ' .. i+j-1 .. ': ', pred[j])
--          if gold ~= nil then
--             print('GOLD ' .. i .. ': ' .. gold[sent_id])
--             if opt.score_gold == 1 then
--                print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
--                gold_score_total = gold_score_total + gold_score
--                gold_words_total = gold_words_total + target:size(1) - 1
--             end
--          end
--       end
--       i = i + batch
--       print('')
--       time_taken2 = timer:time().real - full_time
--       print(time_taken2)
--       time_taken = timer:time().real - start_time
--       print(time_taken)
--       -- if sent_id > 100 then
--       --    break
--       -- end
--    end
--    for i = 1, #sources do
--       out_file:write(pred_sents[i] .. '\n')
--    end
--    print(string.format("WORDS/SEC: %.2f, SENTS/SEC: %.2f", src_words_total/time_taken, sent_id/time_taken))
--    out_file:close()
-- end
-- main()
