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

d = idx2key("de.dict")
torch.save("de.dict.t7", d)
