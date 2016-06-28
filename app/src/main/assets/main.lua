-- Torch Android demo script
-- Script: main.lua
-- Copyright (C) 2013 Soumith Chintala
require 'torch'
require 'nn'
require 'nnx'
require 'dok'
require 'image'
require 'sys'
require 'torchandroid'
require 'nn'
require 'string'
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

cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'en-de-2-500_final.float.t7', [[Path to model .t7 file]])
cmd:option('-src_file', '/sdcard/en-val-clean.txt',
	   [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', '/sdcard/pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'en.dict.t7', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'de.dict.t7', [[Path to target vocabulary (*.targ.dict file)]])


-- beam search options
cmd:option('-beam', 1,[[Beam size]])
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
cmd:option('-unk_penalty', 0)
cmd:option('-length_restriction', 0)
opt = cmd:parse({})


function generate_beam(model, initial, K, max_sent_l, source, gold, batch)
   local long_source = 0
   for i = 1, batch do
      long_source = math.max(long_source, source[i]:size(1))
   end
   -- print(long_source)
   --reset decoder initial states
   local n
   local source_l = math.min(long_source, opt.max_sent_l)
   if opt.length_restriction == 0 then
      n = max_sent_l
   else
      n = source_l + 5
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
   context:resize(batch, source_l, rnn_size)
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
      context[{{},t}]:copy(out[#out])

   end
   -- print("Enc", sys.toc())


   if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, K}, {1, source_l}}]
      context2:copy(context)
      context = context2
   end

   rnn_state_dec = {}
   for i = 1, #init_fwd_dec do
      init_fwd_dec[i]:resize(batch, rnn_size)
      table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
   end
   -- print("Starting beam")
   if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
	 rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1])
	 rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2])
      end
   end

   print("Starting beam2")

   local i = 1
   local done = false
   local found_eos = false
   local pred_targ = {}
   target_input = torch.LongTensor(batch):fill(1)
   for i = 1, target_input:size(1) do
      target_input[i] = START
      table.insert(pred_targ, {START})
   end

   sys.tic()
   local done = torch.LongTensor(batch):zero()
   local all_done = false
   while (not all_done) and (i < n) do
      i = i+1
      local decoder_input = {target_input, context, table.unpack(rnn_state_dec)}
      -- print(decoder_input)

      local out_decoder = model[2]:forward(decoder_input)
      local out = model[3].modules[1]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      _, q = out:max(2)
      for i = 1, target_input:size(1) do
         if done[i] == 0 then
            table.insert(pred_targ[i], q[i][1])
         end
         target_input[i] = q[i][1]
         if q[i][1] == END then
            done[i] = 1
         end
      end
      if done:sum() == batch then
         all_done = true
      end
      rnn_state_dec = {} -- to be modified later
      table.insert(rnn_state_dec, out_decoder[#out_decoder])
      for j = 1, #out_decoder - 1 do
         table.insert(rnn_state_dec, out_decoder[j])
      end
   end
   print("END", sys.toc())
   return pred_targ
end

function idx2key(file)
   t = torch.load(file, "apkbinary64")
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

function strip(s)
   return s:gsub("^%s+",""):gsub("%s+$","")
end

PAD = 1; UNK = 2; START = 3; END = 4
PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
START_CHAR = '{'; END_CHAR = '}'
MAX_SENT_L = opt.max_sent_l
function main(st)
   print(st)
   -- some globals

   -- assert(path.exists(opt.src_file), 'src_file does not exist')
   -- assert(path.exists(opt.model), 'model does not exist')

   -- parse input params
   -- opt = cmd:parse(arg)
   --if opt.gpuid >= 0 then
   --   require 'cutorch'
   --   require 'cunn'
   --end


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

   -- load gold labels if it exists
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
   --   model[i]:double()
   --   model[i]:evaluate()
   --    end
   -- end



   pred_score_total = 0
   gold_score_total = 0
   pred_words_total = 0
   gold_words_total = 0
   src_words_total = 0
   -- target_input = torch.LongTensor({1})
   target_input = torch.LongTensor(opt.batch):fill(1)
   local sent_id = 0
   pred_sents = {}
   --local file = io.open(opt.src_file, "r")
   lines = {st}
   --local out_file = io.open(opt.output_file,'w')
   timer = torch.Timer()
   local start_time = timer:time().real
   local sources = {}
   local targets = {}
   local source_len = {}
   local pred_sents = {}
   for q = 1, #lines do
      line = lines[q]
      sent_id = sent_id + 1
      --line = clean_sent(line)
      -- print('SENT ' .. sent_id .. ': ' ..line)
      local source, source_str
      --if model_opt.use_chars_enc == 0 then
      source, source_str = sent2wordidx(line, word2idx_src, model_opt.start_symbol)

      --else
         --source, source_str = sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
      --end
      if source:dim() > 0 then

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
            print(size, source_tab[1])
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
      --print(to_decode)
      local full_time = timer:time().real
      local start_time2 = timer:time().real
      pred = generate_beam(model, state, opt.beam, MAX_SENT_L, to_decode, to_decode_targ, batch)
      --print("beam", timer:time().real - start_time2)

      for j = 1, batch do
         local source = to_decode[j]
         pred_sent = wordidx2sent(pred[j], idx2word_targ, source_str, attn, true)

         --print('SENT ' .. i+j-1 .. ': ' ..lines[j])
         --print('PRED ' .. i+j-1 .. ': ' .. pred_sent)
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
      --print('')
      time_taken2 = timer:time().real - full_time
      --print(time_taken2)
      time_taken = timer:time().real - start_time
      --print(time_taken)
      -- if sent_id > 100 then
      --    break
      -- end
   end
   --for i = 1, #sources do
   print(pred_sents[1])
   --end
   --print(string.format("WORDS/SEC: %.2f, SENTS/SEC: %.2f", src_words_total/time_taken, sent_id/time_taken))
   --out_file:close()
   return pred_sents[1]
end

function load_model()
print('loading ' .. opt.model .. '...')
checkpoint = torch.load(opt.model, 'apkbinary64')
print('done!')

if opt.replace_unk == 1 then
phrase_table = {}
-- if path.exists(opt.srctarg_dict) then
--    local f = io.open(opt.srctarg_dict,'r')
--    for line in f:lines() do
--       local c = line:split("|||")
--       phrase_table[strip(c[1])] = c[2]
--    end
-- end
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
print("middle")
idx2word_src = idx2key(opt.src_dict)
word2idx_src = flip_table(idx2word_src)
idx2word_targ = idx2key(opt.targ_dict)
word2idx_targ = flip_table(idx2word_targ)


softmax_layers = {}
model[2]:apply(get_layer)
decoder_attn:apply(get_layer)
decoder_softmax = softmax_layers[1]
attn_layer = torch.zeros(opt.beam, MAX_SENT_L)
print("attend")

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

print("end")

init_fwd_enc = {}
init_fwd_dec = {h_init_dec:clone()} -- initial context


print("here")

for L = 1, model_opt.num_layers do
table.insert(init_fwd_enc, h_init_enc:clone())
table.insert(init_fwd_enc, h_init_enc:clone())
table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
end

print("done")
end

