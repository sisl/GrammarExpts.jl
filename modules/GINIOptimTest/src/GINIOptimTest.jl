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

"""
Test various optimistic bounds for GINI impurity, esp on split groups.
"""
module GINIOptimTest

export gini_optim_test, gini_optim_loop

using Distributions #mode
using Reexport
@reexport using DataFrames
using RLESUtils: MathUtils

function gini_optim_loop(ntrials=100)
  fs = [:evenly, :shortest, :longest, :lowest, :highest,
                       :weighted_lowest, :weighted_highest, :max_p, :nmax_p, :bound1, :bound1multi]
  results = DataFrame([ASCIIString, ASCIIString, Int64, Int64, Float64, Int64, Float64,
                       Float64, Float64, Float64, Float64, Float64, Float64, Float64,
                       Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
                      [:v1, :v2, :N, :n1, :g1, :n2, :g2, :max_p1, :max_p2, :nmax_p1, :nmax_p2,
                       fs...], 0)
  for i = 1:ntrials
    n1 = rand(1:10)
    v1 = rand(1:3, n1)
    n2 = rand(1:10)
    v2 = rand(1:3, n2)
    N = rand(1:20)
    trial = loopinner(v1, v2, N)
    push!(results, trial)
  end

  for sym in fs
    results[symbol(sym, :_ismin)] = map(eachrow(results[:, fs])) do r
      a = convert(Array, r)
      Int64(all(r[sym] .<= a))
    end
  end
  results
end

function loopinner(v1, v2, N)
  n1 = length(v1)
  n2 = length(v2)
  mp1 = maximum(counts(v1) / n1)
  mp2 = maximum(counts(v2) / n2)
  [string(v1), string(v2), N,
   n1, gini_impurity(v1),
   n2, gini_impurity(v2),
   mp1, mp2, n1 * mp1, n2 * mp2,
   gini_impurity(fillevenly(v1, v2, N)...),
   gini_impurity(fill2short(v1, v2, N)...),
   gini_impurity(fill2long(v1, v2, N)...),
   gini_impurity(fill2lowest(v1, v2, N)...),
   gini_impurity(fill2highest(v1, v2, N)...),
   gini_impurity(fill2wlowest(v1, v2, N)...),
   gini_impurity(fill2whighest(v1, v2, N)...),
   gini_impurity(fill2maxp(v1, v2, N)...),
   gini_impurity(fill2nmaxp(v1, v2, N)...),
   gini_impurity(fill2bound1(v1, v2, N)...),
   gini_impurity(fill2bound1multi(v1, v2, N)...)]
end

function gini_optim_test(v1=[1,1,2], v2=[1,1,1,2], N=5)
  @show v1
  @show v2
  @show N
  @show gini_impurity(v1, v2)
  @show gini_impurity(fillevenly(v1, v2, N)...)
  @show gini_impurity(fill2short(v1, v2, N)...)
  @show gini_impurity(fill2long(v1, v2, N)...)
  @show gini_impurity(fill2lowest(v1, v2, N)...)
  @show gini_impurity(fill2highest(v1, v2, N)...)
end

function fillevenly(v1, v2, N)
  a = vcat(v1, fill(mode(v1), ceil(Int, N / 2)))
  b = vcat(v2, fill(mode(v2), floor(Int, N / 2)))
  return a, b
end

function fill2short(v1, v2, N)
  if length(v1) < length(v2)
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2long(v1, v2, N)
  if length(v1) > length(v2)
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2lowest(v1, v2, N)
  if gini_impurity(v1) < gini_impurity(v2)
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2highest(v1, v2, N)
  if gini_impurity(v1) > gini_impurity(v2)
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2wlowest(v1, v2, N)
  if length(v1) * gini_impurity(v1) < length(v2) * gini_impurity(v2)
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2whighest(v1, v2, N)
  if length(v1) * gini_impurity(v1) > length(v2) * gini_impurity(v2)
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2maxp(v1, v2, N)
  if maximum(counts(v1)) > maximum(counts(v2))
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2nmaxp(v1, v2, N)
  if length(v1) * maximum(counts(v1)) > length(v2) * maximum(counts(v2))
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2bound1(v1, v2, N)
  c1 = counts(v1)
  c2 = counts(v2)
  n1 = sum(c1)
  n2 = sum(c2)
  n1t = maximum(c1)
  n2t = maximum(c2)

  n1f = n1 - n1t
  n2f = n2 - n2t

  b1 = (n1f^2) / (n1 * (n1 + N))
  b2 = (n2f^2) / (n2 * (n2 + N))

  if b1 < b2
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

function fill2bound1multi(v1, v2, N)
  c1 = counts(v1)
  c2 = counts(v2)
  n1 = sum(c1)
  n2 = sum(c2)
  n1t = maximum(c1)
  n2t = maximum(c2)

  i1 = indmax(c1)
  i2 = indmax(c2)
  c1[i1] = 0
  c2[i2] = 0

  b1 = (sum(c1)^2 + sumabs2(c1)) / (n1 * (n1 + N))
  b2 = (sum(c2)^2 + sumabs2(c2)) / (n2 * (n2 + N))

  if b1 < b2
    v = vcat(v1, fill(mode(v1), N))
    return v, v2
  else
    v = vcat(v2, fill(mode(v2), N))
    return v1, v
  end
end

end #module
