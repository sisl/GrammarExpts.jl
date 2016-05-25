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
Test tight bound on GINI impurity index with remaining
"""
module GINIOptimTest2

export testmain

using Distributions #mode
using Reexport
@reexport using DataFrames
using RLESUtils, MathUtils, BitCounters

function testmain(ntrials=50; d=3)
  results = DataFrame([Int64,Int64,Int64,Int64,Int64,Int64,Int64,Int64,Int64,
                       Int64,Int64,Int64,Int64,Int64,Int64,Float64,
                       Int64,Int64,Int64,Int64,Int64,Int64,Float64,
                       Int64,Int64,Int64,Int64,Int64,Int64,Float64],
                      [:Ca, :Cb, :Cc, :n1a, :n1b, :n1c, :n2a, :n2b,:n2c,
                       :samp_u1a, :samp_u1b, :samp_u1c, :samp_u2a, :samp_u2b, :samp_u2c, :samp_score,
                       :allc_u1a, :allc_u1b, :allc_u1c, :allc_u2a, :allc_u2b, :allc_u2c, :allc_score,
                       :heur_u1a, :heur_u1b, :heur_u1c, :heur_u2a, :heur_u2b, :heur_u2c, :heur_score], 0)
  for i = 1:ntrials
    C = rand(1:10, d)
    n1 = rand(1:10, d)
    n2 = rand(1:10, d)

    samp_u = samplingbest(C, n1, n2)
    samp_score = evalscore(n1, n2, samp_u...)

    allc_u = allcombos(C, n1, n2)
    allc_score = evalscore(n1, n2, allc_u...)

    heur_u = heuristic1(C, n1, n2)
    heur_score = evalscore(n1, n2, heur_u...)

    trial = [C..., n1..., n2...,
             samp_u..., samp_score,
             allc_u..., allc_score,
             heur_u..., heur_score]
    push!(results, trial)
  end

  results
end

function evalscore(n1, n2, u1, u2)
  gini_from_counts(n1 + u1, n2 + u2)
end

function samplingbest(C, n1, n2; nsamples=10000)
  u1 = u1_ = similar(C)
  u2 = u2_ = similar(C)
  score = 1.0

  #draw samples
  for i = 1:nsamples
    for j = 1:length(C)
      u1_[j] = rand(0:C[j])
      u2_[j] = C[j] - u1_[j]
    end
    score_ = evalscore(n1, n2, u1_, u2_)

    if score_ < score
      score = score_
      u1 = u1_
      u2 = u2_
    end
  end

  u1, u2
end

function allcombos(C, n1, n2)
  ncases = 2^(length(C))
  case_u1 = Array(Vector, ncases)
  case_u2 = Array(Vector, ncases)
  case_score = Array(Float64, ncases)
  for (i, v) in enumerate(BitCounter(length(C)))
    case_u1[i] = v .* C
    case_u2[i] = !v .* C
    case_score[i] = evalscore(n1, n2, case_u1[i], case_u2[i])
  end

  i = indmin(case_score)
  return case_u1[i], case_u2[i]
end

#apply "remaining" bound greedily based on descending order on C[i]
function heuristic1(C, n1, n2)
  u1 = zeros(Int64, length(C))
  u2 = zeros(Int64, length(C))
  n1_ = deepcopy(n1)
  n2_ = deepcopy(n2)
  C_ = deepcopy(C)

  for i in sortperm(C_)
    j = evalcriteria(n1_, n2_, i, C_[i])
    if j == 1
      n1_[i] += C_[i]
      u1[i] += C_[i]
    else #j==2
      n2_[i] += C_[i]
      u2[i] += C_[i]
    end
    C_[i] = 0
  end
  u1, u2
end

function evalcriteria(c1, c2, i, N)
  n1 = sum(c1)
  n2 = sum(c2)

  c1_ = deepcopy(c1)
  c1_[i] = 0 #don't count the i of interest
  c2_ = deepcopy(c2)
  c2_[i] = 0 #don't count the i of interest

  b1 = (sum(c1_)^2 + sumabs2(c1_)) / (n1 * (n1 + N))
  b2 = (sum(c2_)^2 + sumabs2(c2_)) / (n2 * (n2 + N))

  if b1 < b2
    return 1
  else
    return 2
  end
end

end #module
