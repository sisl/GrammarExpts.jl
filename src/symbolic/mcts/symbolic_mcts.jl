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

module SYMBOLIC_MCTS

export symbolic_mcts

using ExprSearch.MCTS
using Reexport

import GrammarExpts.CONFIG

#defaults
if !haskey(CONFIG, :config)
  CONFIG[:config] = :test
end
if !haskey(CONFIG, :gt)
  CONFIG[:gt] = :easy
end
if !haskey(CONFIG, :treevis)
  CONFIG[:treevis] = false
end

println("Configuring: config=$(CONFIG[:config]), gt=$(CONFIG[:gt]), treevis=$(CONFIG[:treevis])")

#2d polynomials with integral coeffs
include("../grammar/grammar_poly2d/GrammarDef.jl") #grammar

if CONFIG[:config] == :test
  include("test_config.jl") #for testing
elseif CONFIG[:config] == :normal
  include("config.jl")
else
  error("config not valid ($(CONFIG[:config]))")
end

if CONFIG[:gt] == :easy
  include("../common/gt_easy.jl")
elseif CONFIG[:gt] == :higherorder
  include("../common/gt_higherorder.jl")
else
  error("gt not valid ($(CONFIG[:gt]))")
end
include("reward.jl")
include("logs.jl")

#FIXME
if CONFIG[:treevis]
  include("treeview.jl")
end

using .GrammarDef

function symbolic_mcts(outdir::AbstractString="./"; seed=1,
                    logfileroot::AbstractString="symbolic_mcts_log",
                    n_iters::Int64=N_ITERS,
                    searchdepth::Int64=SEARCHDEPTH,
                    exploration_const::Float64=EXPLORATIONCONST,
                    safetylimit::Int64=SAFETYLIMIT,
                    q0::Float64=MAX_NEG_REWARD,
                    treevis::Bool=CONFIG[:treevis])
  srand(seed)

  grammar = create_grammar()
  tree_params = DerivTreeParams(grammar, MAXSTEPS)
  mdp_params = DerivTreeMDPParams(grammar)

  observer = Observer()
  add_observer(observer, "verbose1", x -> println(x[1]))
  add_observer(observer, "action", x -> println("step=$(x[1]), action=$(x[2])"))
  add_observer(observer, "result", x -> println("total_reward=$(x[1]), expr=$(x[2]), best_at_eval=$(x[3]), total_evals=$(x[4])"))
  add_observer(observer, "current_best", x -> println("step $(x[1]): best_reward=$(x[2]), best_state=$(x[3].past_actions)"))
  logs = define_logs(observer)

  mcts_observer = Observer()

  if treevis
    startstate = DerivTreeState() #assumes empty constructor is initial state...
    view, viewstep = viewstep_f(startstate, 1)
    add_observer(mcts_observer, "tree", viewstep)
  end

  mcts_params = MCTSESParams(tree_params, mdp_params, n_iters, searchdepth,
                             exploration_const, q0, mcts_observer, safetylimit,
                             observer)

  result = exprsearch(mcts_params)

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  return result
end

end #module
