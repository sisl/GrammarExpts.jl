# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã 2015, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.  The Reinforcement Learning Encounter Simulator (RLES)
# platform is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You
# may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
# law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# _____________________________________________________________________________
# Reinforcement Learning Encounter Simulator (RLES) includes the following
# third party software. The SISLES.jl package is licensed under the MIT Expat
# License: Copyright (c) 2014: Youngjun Kim.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
# "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# *****************************************************************************

#TODO: Convert to new Problem format and move to /modules/AntProblem

module GrammarDef

export create_grammar, to_function
export World, Ant, copy

using GrammaticalEvolution
import GrammaticalEvolution.isless
import Base: +, -, copy

function create_grammar()
  @grammar grammar begin
    start = block
    command = turnleft | turnright | forward | if_statement | loop
    turnleft = Expr(:call, :turn_left, :ant)
    turnright = Expr(:call, :turn_right, :ant)
    forward = Expr(:call, :forward, :world, :ant)
    if_statement = Expr(:if, condition, block, block)
    condition = food_ahead
    food_ahead = Expr(:call, :food_ahead, :world, :ant)
    loop = Expr(:for, Expr(:(=), :i, Expr(:(:), 1, digit)), block)
    block[make_block] = (command)^(1:5)
    digit = 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
  end
  return grammar
end

function to_function(code)
  @eval f(world::World, ant::Ant) = $code
  return f
end

const FOOD = convert(Int64, '1')
const GAP = convert(Int64, '0')
const EMPTY = 46 #convert(Int64, '.')

type World
  width::Int64
  height::Int64
  value::Array{Int64, 2}
  start::Array{Int64, 1}

  function World(width::Int64, height::Int64)
    value = zeros(width, height)
    return new(width, height, value)
  end

  function World(map::Array{Int64, 2}, start::Array)
    return new(size(map, 1), size(map, 2), map, start)
  end
end

copy(world::World) = World(copy(world.value), world.start)

immutable Direction
  value::Int64
end

+{T <: Number}(d::Direction, v::T) = Direction(d.value + v)
-{T <: Number}(d::Direction, v::T) = Direction(d.value - v)
isless(d1::Direction, d2::Direction) = isless(d1.value, d2.value)
isless(d1::Direction, v::Int64) = isless(d1.value, value)

const north = Direction(1)
const east = Direction(2)
const south = Direction(3)
const west = Direction(4)

type Ant
  x::Int64
  y::Int64
  direction::Direction
  eaten::Int64

  function Ant(x::Int64, y::Int64)
    return new(x, y, north, 0)
  end
end

function calc_forward(world::World, ant::Ant)
  nx = ant.x
  ny = ant.y

  if ant.direction === north
    ny -= 1
    if ny < 1 ny = world.height end
  elseif ant.direction === east
    nx += 1
    if nx > world.width nx = 1 end
  elseif ant.direction === south
    ny += 1
    if ny > world.height ny = 1 end
  else
    nx -= 1
    if nx < 1 nx = world.width end
  end

  return (nx, ny)
end

function food_ahead(world::World, ant::Ant)
  (x, y) = calc_forward(world, ant)
  return world.value[x, y] == FOOD
end

function forward(world::World, ant::Ant)
  (ant.x, ant.y) = calc_forward(world, ant)
  if world.value[ant.x, ant.y] == FOOD
    world.value[ant.x, ant.y] = EMPTY
    ant.eaten += 1
  elseif world.value[ant.x, ant.y] == EMPTY
    ant.eaten -= 1
  end
end

function turn_right(ant::Ant)
  #ant.direction = ant.direction + 1 > south ? north : ant.direction+1
  ndir = ant.direction + 1
  ant.direction = ndir.value > south.value ? north : ndir;
end

function turn_left(ant::Ant)
  #ant.direction = ant.direction - 1 < 1 ? south : ant.direction-1
  ndir = ant.direction - 1
  ant.direction = ndir.value < 1 ? south : ndir
end

make_block(lst::Array) = Expr(:block, lst...)
make_loop(values::Array) = Expr(:for, :(=), :i, Expr(:(:), 1, values[1]), values[2])
make_if(values::Array) = Expr(:if, values[1], values[2], values[3])

macro make_if(condition, true_block, false_block)
  Expr(:if, condition, true_block, false_block)
end

macro make_call(fn, args...)
  Expr(:call, fn, args...)
end

macro make_for(start, stop, block)
  Expr(:for, Expr(:(=), :i, Expr(:(:), start, stop)), block)
end


end #module
