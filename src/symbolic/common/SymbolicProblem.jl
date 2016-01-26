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

module SymbolicProblem

export Symbolic, create_grammar, get_fitness, to_function

using ExprSearch
import ExprSearch: ExprProblem, create_grammar, get_fitness

const XRANGE = 0.0:0.5:10.0
const YRANGE = 0.0:0.5:10.0
const W_LEN = 0.1

type Symbolic{T<:AbstractFloat} <: ExprProblem
  xrange::FloatRange{T}
  yrange::FloatRange{T}
  w_len::Float64
end

function Symbolic{T<:AbstractFloat}(gt_file::AbstractString, xrange::FloatRange{T}=XRANGE, yrange::FloatRange{T}=YRANGE, w_len::Float64=W_LEN)
  @eval include($gt_file) #define gt in module scope
  return Symbolic(xrange, yrange, w_len)
end

function ExprSearch.create_grammar(problem::Symbolic)
  @grammar grammar begin
    start = ex
    ex = sum | product | (ex) | value
    sum = Expr(:call, :+, ex, ex)
    product = Expr(:call, :*, ex, ex)
    value = :x | :y | digit
    digit = 0:9
  end
  return grammar
end

function to_function(problem::Symbolic, expr)
  @eval f(x, y) = $expr
  return f
end

function ExprSearch.get_fitness(problem::Symbolic, expr)
  #mean-square error over a range
  sum_se = 0.0
  f = to_function(problem, expr)
  for x in problem.xrange, y in problem.yrange
    sum_se += abs2(f(x, y) - gt(x, y))
  end
  n = length(problem.xrange) * length(problem.yrange)
  fitness = sum_se / n + problem.w_len * length(string(expr))

  return fitness
end

end #module
