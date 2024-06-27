#!/usr/bin/env lua
local the={bins=17, fmt="%6.3f", cohen=0.35, seed=1234567891,
           train="../data/misc/auto93.csv"}
local big=1E30

local DATA,SYM,NUM,COLS,XY,ROW = {},{},{},{},{},{}
local abs,floor,max,min = math.abs,math.floor,math.max, math.min
local coerce,coerces,csv,fmt,new,o,okey,okeys,olist,push,sort
-----------------------------------------------------------------------------------------
function NUM.new(name,pos)
  return new(NUM,{name=name, pos=pos, n=0, mu=0, m2=0, sd=0, lo=big, hi= -big,
                  goal= name:find"-$" and 0 or 1}) end

function NUM:add(x,     d)
  if x ~= "?" then
    self.n  = self.n + 1
    d       = x - self.mu
    self.mu = self.mu + d/self.n
    self.m2 = self.m2 + d*(x - self.mu)
    self.sd = self.n<2 and 0 or (self.m2/(self.n - 1))^.5 
    self.lo = min(x, self.lo)
    self.hi = max(x, self.hi)
    return self end end 

function NUM:norm(x) return x=="?" and x or (x - self.lo)/(self.hi - self.lo) end

function NUM:small(x) return x < the.cohen * self.sd end

function NUM.same(i,j,    pooled)
  pooled = (((i.n-1)*i.sd^2 + (j.n-1)*j.sd^2)/ (i.n+j.n-2))^0.5
  return abs(i.mu - j.mu) / pooled >= (the.cohen or .35) end
-----------------------------------------------------------------------------------------
function SYM.new(name,pos)
  return new(SYM,{name=name, pos=pos, n=0, has={}, most=0, mode=nil}) end

function SYM:add(x,     d)
  if x ~= "?" then
    self.n  = self.n + 1
    self.has[x] = 1 + (self.has[x] or 0)
    if self.has[x] > self.most then self.most,self.mode = self.has[x], x end end end
-----------------------------------------------------------------------------------------
local _id=0
local function id() _id=_id+1; return _id end

function ROW.new(t) return new(ROW,{cells=t,y=0,id=id()}) end

function DATA.new(file,    self) 
  self = new(DATA, {rows={}, cols=nil})
  for row in csv(file) do  self:add(ROW.new(row)) end
  for n,row in pairs(self.rows) do  row.y =  1 - self:chebyshev(row) end
  return self end

function DATA:add(row)
  if self.cols then push(self.rows, self.cols:add(row)) else 
     self.cols = COLS.new(row) end end 

function DATA:chebyshev(row,     d) 
  d=0; for _,col in pairs(self.cols.y) do 
         d = max(d,abs(col:norm(row.cells[col.pos]) - col.goal)) end
  return d end

function DATA:bins(rows,      bins,val) 
  bins = {}
  for _,col in pairs(self.cols.x) do
    val = function(a) return a.cells[col.pos]=="?" and -big or a.cells[col.pos] end
    col:bins(bins, sort(rows, function(a,b) return val(a) < val(b) end)) end
  return bins end 

for row in rows
  one:add(row)
  two:add(row)
  onetwo:add(row)
  if enough(one)
    push(tmp,two)
    two.same(one)
    all[#all]=onetwo
    two=new(thing)
    one=two 

a,b,ab
andd to b and ab
if a:same b, switch a for ab, make new b
of a different make a new 
-- add the stats test here using pooled cohen
function NUM:bins(rows,bins,     big,dull,b,out,start) 
  tmp = {}
  b1 = XY(col.name, col.pos)
  b2 = XY(col.name, col.pos)
  b12 = XY(col.name,cols.pos)
  twin = XY(col.name, col.pos)
  for k,row in pairs(rows) do
    if row[cols.pos] ~= "?" then 
      want = want or (#rows - k - 1)/the.bins
      if b.y.n >= want and #rows - k > want and not col:small(b.hi - b.lo) then
        if b2:same(b1) then
           tmp[#tmp] = b12
           b1 = b2
           b12 =
        if b1 same as b12
          
        
        b = push(bins, push(tmp, XY(col.name, col.pos))) end
      c:add(row) 
      b:add(row) end end 
  tmp[1].lo = - big
  tmp[#tmp].hi = big
  for k = 2,#t do tmp[k].lo = tmp[k-1].hi end end
-------------------------------------------------------------------------------------
function COLS.new(row,    self,skip,col)
  self = new(COLS,{all={},x={}, y={}, klass=nil})
  skip={}
  for k,v in pairs(row.cells) do
    col = push(v:find"X$" and skip or v:find"[!+-]$" and self.y or self.x,
               push(self.all, 
                    (v:find"^[A-Z]" and NUM or SYM).new(v,k))) 
    if v:find"!$" then self.klass=col end end
  return self end 

function COLS:add(row)
  for _,cols in pairs{self.x, self.y} do
    for _,col in pairs(cols) do  col:add(row.cells[col.pos]) end end 
  return row end
-----------------------------------------------------------------------------------------
function XY.new(name,pos)
  return new(XY,{lo=big, hi= -big,  rules={}, y=NUM(name,pos)}) end

function XY:add(row,     x) 
  x = row[self.y.pos]
  if x ~= "?" then
    if x < self.lo then self.lo = x end
    if x > self.hi then self.hi = x end
    self.rules[row.id] = row.id
    self.y:add(row.y) end end
-----------------------------------------------------------------------------------------
fmt = string.format

function pub(k)  return not tostring(k):find"^_" end
function okeys(t)  local u={}; for k,v in pairs(t) do if pub(k) then push(u, fmt(":%s %s", k,o(v))) end end; return sort(u) end
function olist(t)  local u={}; for k,v in pairs(t) do push(u, fmt("%s", o(v))) end; return u end

function o(x)
  if type(x)=="number" then return x == floor(x) and tostring(x) or fmt(the.fmt,x) end
  if type(x)~="table"  then return tostring(x) end 
  return "{" .. table.concat(#x==0 and okeys(x) or olist(x),", ")  .. "}" end

function new (klass,object) 
  klass.__index=klass; klass.__tostring=o; setmetatable(object, klass); return object end

function coerce(s,    also)
  if s ~= nil then
    also = function(s) return s=="true" or s ~="false" and s end 
    return math.tointeger(s) or tonumber(s) or also(s:match"^%s*(.-)%s*$") end end

function coerces(s,    t)
  t={}; for s1 in s:gsub("%s+", ""):gmatch("([^,]+)") do t[1+#t]=coerce(s1) end
  return t end

function csv(src)
  src = src=="-" and io.stdin or io.input(src)
  return function(      s)
    s = io.read()
    if s then return coerces(s) else io.close(src) end end end

function push(t,x) t[1+#t]=x; return x end 
function sort(t,fun) table.sort(t,fun); return t end

function copy(t,     u)
  if type(t) ~= "table" then return t end 
  u={}; for k,v in pairs(t) do u[copy(k)] = copy(v) end 
  return setmetable(u, getmetatable(t)) end
-----------------------------------------------------------------------------------------
local eg={}

eg["-h"] = function(_) print"USAGE: lua sandbox.lua -[hkln] [ARG]" end
eg["-s"] = function(s) print(s) end
eg["-t"] = function(file,     d) 
  d= DATA.new(file or the.train) 
  want=1
  for i,row in pairs(sort(d.rows,function(a,b) return a.y > b.y end)) do
    if i == want then want=2*want; print(i, o{y=row.y,row=row.cells}) end end end

if   pcall(debug.getlocal, 4, 1) 
then return {DATA=DATA,NUM=NUM,SYM=SYM,XY=XY}
else math.randomseed(the.seed or 1234567891)
     for k,v in pairs(arg) do if eg[v] then eg[v](coerce(arg[k+1])) end end end
