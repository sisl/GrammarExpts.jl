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
MCTS for the Symbolic regression problem.
Example usage: config=configure(Symbolic_MCTS,"normal","nvn_dasc"); symbolic_mcts(;config...)
"""
module Symbolic_MCTS

export configure, symbolic_mcts

using ExprSearch.MCTS
using Reexport
using JSON, GZip
using RLESUtils, FileUtils, Configure

using GrammarExpts
using SymbolicProblem, MCTS_Logs
using DerivTreeVis, MCTSTreeView
import Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")

configure(::Type{Val{:Symbolic_MCTS}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

function symbolic_mcts(;outdir::AbstractString="./Symbolic_MCTS",
                        seed=1,
                        logfileroot::AbstractString="symbolic_mcts_log",

                        gt_file::AbstractString="gt_easy.jl",
                        maxsteps::Int64=25,

                        n_iters::Int64=200,
                        searchdepth::Int64=20,
                        explorationconst::Float64=2000.0,
                        q0::Float64=-1000.0,
                        max_neg_reward::Float64=-1000.0,
                        step_reward::Float64=0.0,

                        loginterval::Int64=100,
                        vis::Bool=true,
                        mctstreevis::Bool=false,
                        treevis_interval::Int64=50,
                        observer::Observer=Observer())
  mkpath(outdir)

  problem = Symbolic(gt_file)

  logs = default_logs(observer, loginterval)
  default_console!(observer)

  if mctstreevis
    view, viewstep = viewstep_f(treevis_interval)
    add_observer(observer, "mcts_tree", viewstep)
  end

  mcts_observer = Observer()
  #add_observer(mcts_observer, "terminal_reward", x -> println("r=", x[1], " state=", x[2].past_actions))

  mcts_params = MCTSESParams(maxsteps, max_neg_reward, step_reward, n_iters, searchdepth,
                             explorationconst, q0, seed, mcts_observer,
                             observer)

  result = exprsearch(mcts_params, problem)

  @notify_observer(observer, "parameters", ["seed", seed])

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  #save mcts tree
  if mctstreevis
    GZip.open(joinpath(outdir, "mctstreevis.json.gz"), "w") do f
      JSON.print(f, view.steps)
    end
  end

  textfile(joinpath(outdir, "summary.txt"), "mcts", seed=seed, n_iters=n_iters,
           reward=result.reward, expr=string(result.expr))

  if vis
    derivtreevis(result.tree, joinpath(outdir, "$(logfileroot)_derivtreevis"))
  end

  return result
end

end #module
