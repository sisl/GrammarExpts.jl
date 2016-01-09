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

module ACASX_MCTS

export acasx_mcts

using Reexport

include("../grammar/grammar_typed/GrammarDef.jl") #grammar
include("config.jl")

#pick a dataset
include("data_dasc.jl")
#include("data_libcas098_small.jl")

include("labeleddata.jl")
include("infogain.jl")
include("fitness.jl")
include("reward.jl")
include("logs.jl")

using .GrammarDef
using Datasets
using TikzQTrees

using ExprSearch

#nmacs vs nonnmacs
function acasx_mcts(; seed=1,
                    runtype::AbstractString="nmacs_vs_nonnmacs",
                    clusterdataname::AbstractString="",
                    data::DFSet=DATASET,
                    data_meta::DataFrame=DATASET_META,
                    n_iters::Int64=N_ITERS,
                    searchdepth::Int64=SEARCHDEPTH,
                    exploration_const::Float64=EXPLORATIONCONST,
                    safetylimit::Int64=SAFETYLIMIT)
  srand(seed)

  Dl = if runtype == "nmacs_vs_nonnmacs"
    nmacs_vs_nonnmacs(data, data_meta)
  elseif runtype == "nmac_clusters"
    clustering = dataset(manuals, clusterdataname)
    nmac_clusters(clustering, data)
  elseif runtype == "nonnmacs_extra_cluster"
    clustering = dataset(manuals, clusterdataname)
    nonnmacs_extra_cluster(clustering, data, data_meta)
  else
    error("runtype not recognized ($runtype)")
  end

  define_reward(Dl)

  grammar = create_grammar()
  tree_params = DerivTreeParams(grammar, MAXSTEPS)
  mdp_params = DerivTreeMDPParams(grammar)

  observer = Observer()
  add_observer(observer, "action", x -> println("step=$(x[1]), action=$(x[2])"))
  add_observer(observer, "result", x -> println("total_reward=$(x[1]), expr=$(x[2])"))
  logs = define_logs(observer)

  mcts_observer = Observer()

  mcts_params = MCTSESParams(tree_params, mdp_params, n_iters, searchdepth,
                             exploration_const, mcts_observer, safetylimit,
                             observer)

  result = exprsearch(mcts_params)

  save_log("$(runtype)_$(clusterdataname)_log.txt", logs)

  return result
end

end #module
