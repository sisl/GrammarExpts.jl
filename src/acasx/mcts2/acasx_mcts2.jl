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

module ACASX_MCTS2

export acasx_mcts2

using ExprSearch.MCTS2
using Datasets
using Reexport
using JSON

import GrammarExpts.CONFIG

#defaults
if !haskey(CONFIG, :config)
  CONFIG[:config] = :test
end
if !haskey(CONFIG, :data)
  CONFIG[:data] = :dasc
end
if !haskey(CONFIG, :vis)
  CONFIG[:vis] = true
end
if !haskey(CONFIG, :mctstreevis)
  CONFIG[:mctstreevis] = false
end

println("Configuring: config=$(CONFIG[:config]), data=$(CONFIG[:data]), vis=$(CONFIG[:vis]), treevis=$(CONFIG[:mctstreevis])")

include("../common/ACASXProblem.jl")
using .ACASXProblem

if CONFIG[:config] == :test
  include("test_config.jl") #for testing
elseif CONFIG[:config] == :normal
  include("config.jl")
elseif CONFIG[:config] == :highest
  include("highest_config.jl")
else
  error("config not valid ($config)")
end

if CONFIG[:data] == :dasc
  include("../common/data_dasc.jl")
elseif CONFIG[:data] == :libcas098_small
  include("../common/data_libcas098_small.jl")
else
  error("data not valid ($data)")
end

include("../../logs/mcts2_logs.jl")
include("../common/format.jl")

if CONFIG[:mctstreevis]
  include("treeview.jl") #mcts tree
end
if CONFIG[:vis]
  include("../common/derivtreevis.jl") #derivation tree
end

function acasx_mcts2(outdir::AbstractString="./"; seed=1,
                     runtype::Symbol=:nmacs_vs_nonnmacs,
                     clusterdataname::AbstractString="",
                     logfileroot::AbstractString="acasx_mcts2_log",
                     data::DFSet=DATASET,
                     data_meta::DataFrame=DATASET_META,
                     n_iters::Int64=N_ITERS,
                     searchdepth::Int64=SEARCHDEPTH,
                     exploration_const::Float64=EXPLORATIONCONST,
                     q0::Float64=MAX_NEG_REWARD,
                     vis::Bool=CONFIG[:vis],
                     mctstreevis::Bool=CONFIG[:mctstreevis],
                     maxsteps::Int64=MAXSTEPS,
                     max_neg_reward::Float64=MAX_NEG_REWARD,
                     step_reward::Float64=STEP_REWARD)
  srand(seed)

  problem = ACASXClustering(runtype, data, clusterdataname, data_meta)

  observer = Observer()
  logs = default_logs(observer)
  default_console!(observer)

  mcts2_observer = Observer()

  if mctstreevis
    view, viewstep = viewstep_f(TREEVIS_INTERVAL)
    add_observer(observer, "mcts_tree", viewstep)
  end

  mcts2_params = MCTS2ESParams(maxsteps, max_neg_reward, step_reward, n_iters, searchdepth,
                             exploration_const, q0, mcts2_observer,
                             observer)

  result = exprsearch(mcts2_params, problem)

  @notify_observer(observer, "parameters", ["seed", seed])
  @notify_observer(observer, "parameters", ["runtype", runtype])
  @notify_observer(observer, "parameters", ["clusterdataname", clusterdataname])
  @notify_observer(observer, "parameters", ["config", CONFIG[:config]])
  @notify_observer(observer, "parameters", ["data", CONFIG[:data]])
  @notify_observer(observer, "parameters", ["vis", vis])
  @notify_observer(observer, "parameters", ["mctstreevis", mctstreevis])
  @notify_observer(observer, "expression", [string(result.expr),
                                             pretty_string(result.tree, FMT_PRETTY),
                                             pretty_string(result.tree, FMT_NATURAL, true)])

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  #save("$(logfileroot)_derivtree.jld", "tree", result.tree)

  if vis
    derivtreevis(result.tree, joinpath(outdir, "$(logfileroot)_derivtreevis"))
  end

  #save mcts tree
  if mctstreevis
    open(joinpath(outdir, "mctstreevis.json"), "w") do f
      JSON.print(f, view.steps)
    end
  end

  return result
end

end #module

