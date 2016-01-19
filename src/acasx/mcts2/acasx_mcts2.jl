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

if !haskey(CONFIG, :treevis)
  CONFIG[:treevis] = false
end

println("Configuring: config=$(CONFIG[:config]), data=$(CONFIG[:data]), treevis=$(CONFIG[:treevis])")

include("../grammar/grammar_typed/GrammarDef.jl") #grammar

if CONFIG[:config] == :test
  include("test_config.jl") #for testing
elseif CONFIG[:config] == :normal
  include("config.jl")
elseif CONFIG[:config] == :higher
  include("higher_config.jl")
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

include("../common/labeleddata.jl")
include("reward.jl")
include("logs.jl")

if CONFIG[:treevis]
  include("treeview.jl")
end

using .GrammarDef

function acasx_mcts2(outdir::AbstractString="./"; seed=1,
                    runtype::AbstractString="nmacs_vs_nonnmacs",
                    clusterdataname::AbstractString="",
                    logfileroot::AbstractString="acasx_mcts2_log",
                    data::DFSet=DATASET,
                    data_meta::DataFrame=DATASET_META,
                    n_iters::Int64=N_ITERS,
                    searchdepth::Int64=SEARCHDEPTH,
                    exploration_const::Float64=EXPLORATIONCONST,
                    q0::Float64=MAX_NEG_REWARD,
                    treevis::Bool=CONFIG[:treevis])
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
  add_observer(observer, "verbose1", x -> println(x[1]))
  add_observer(observer, "iteration", x -> begin
                 i = x[1]
                 rem(i, 100) == 0 && println("iteration $i")
               end)
  add_observer(observer, "result", x -> println("total_reward=$(x[1]), expr=$(x[2]), best_at_eval=$(x[3]), total_evals=$(x[4])"))
  add_observer(observer, "current_best", x -> begin
                 i, reward, state = x
                 rem(i, 100) == 0 && println("step $i: best_reward=$(reward), best_state=$(state.past_actions)")
               end)

  logs = define_logs(observer)
  @notify_observer(observer, "parameters", ["seed", seed])
  @notify_observer(observer, "parameters", ["runtype", runtype])
  @notify_observer(observer, "parameters", ["clusterdataname", clusterdataname])
  @notify_observer(observer, "parameters", ["config", CONFIG[:config]])
  @notify_observer(observer, "parameters", ["data", CONFIG[:data]])
  @notify_observer(observer, "parameters", ["treevis", CONFIG[:treevis]])

  mcts2_observer = Observer()

  if treevis
    startstate = DerivTreeState() #assumes empty constructor is initial state...
    view, viewstep = viewstep_f(startstate, TREEVIS_INTERVAL, Counter(1))
    add_observer(mcts2_observer, "tree", viewstep)
  end

  mcts2_params = MCTS2ESParams(tree_params, mdp_params, n_iters, searchdepth,
                             exploration_const, q0, mcts2_observer,
                             observer)

  result = exprsearch(mcts2_params)

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  #save treevis
  if treevis
    open("treevis.json", "w") do f
      JSON.print(f, view.steps)
    end
  end

  return result
end

end #module
