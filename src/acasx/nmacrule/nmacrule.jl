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

include("../common/ACASXProblem.jl")
using .ACASXProblem

using ExprSearch
using DerivationTrees

const CONF_DATA = :libcas098_small
const NMAC_SEQUENCE = Int64[2, 2, 7, 11, 7, 7, 5, 3, 10]
#results in :(F((D[:,76] .< 100) & (D[:,77] .< 500))) which is 38 chars

include("config.jl")

if CONF_DATA == :dasc
  include("../common/data_dasc.jl")
elseif CONF_DATA == :libcas098_small
  include("../common/data_libcas098_small.jl")
end

function setup(; runtype::Symbol=:nmacs_vs_nonnmacs,
               clusterdataname::AbstractString="",
               data::DFSet=DATASET,
               data_meta::DataFrame=DATASET_META,
               maxsteps::Int64=MAXSTEPS)

  problem = ACASXClustering(runtype, data, clusterdataname, data_meta)

  grammar = create_grammar(problem)
  tree_params = DerivTreeParams(grammar, maxsteps)
  tree = DerivationTree(tree_params)

  return problem, tree
end

function playsequence(problem::ExprProblem, tree::DerivationTree,
                      sequence::Vector{Int64}=NMAC_SEQUENCE)
  play!(tree, sequence)
  expr = get_expr(tree)
  fitness = get_fitness(problem, expr)

  return tree, fitness, expr
end

function nmacrule()
  problem, tree = setup()
  return tree, fitness, expr = playsequence(problem, tree)
end
