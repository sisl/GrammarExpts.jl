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
Use a decision tree to recursively split encounters in the ACASX Problem. MCTS2 algorithm.
Example usage: config=configure(ACASX_MCTS2_Tree,"normal","nvn_dasc"); acasx_mcts2_tree(;config...)
"""
module ACASX_MCTS2_Tree

export configure, acasx_mcts2_tree

using DecisionTrees
using ExprSearch.MCTS2
using Datasets
using RLESUtils.Obj2Dict
using Reexport

using GrammarExpts
using ACASXProblem, MCTS2_Tree_Logs
using DerivTreeVis, MCTSTreeView, DecisionTreeVis
using RLESUtils.Configure
import RLESUtils.Configure.configure

include("dtree_callbacks.jl")

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")

configure(::Type{Val{:ACASX_MCTS2_Tree}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

function train_dtree{T}(mcts2_params::MCTS2ESParams, problem::ACASXClustering, Dl::DFSetLabeled{T},
                        maxdepth::Int64, loginterval::Int64)

  logs = default_logs()
  num_data = length(Dl)
  T1 = Bool #predict_type
  T2 = Int64 #label_type

  p = DTParams(num_data, maxdepth, T1, T2)

  dtree = build_tree(p,
                     Dl, problem, mcts2_params, logs, loginterval) #userargs...

  return dtree, logs
end

function acasx_mcts2_tree(;outdir::AbstractString="./",
                          seed=1,
                          logfileroot::AbstractString="acasx_mcts2_tree_log",

                          runtype::Symbol=:nmacs_vs_nonnmacs,
                          data::AbstractString="dasc",
                          data_meta::AbstractString="dasc_meta",
                          manuals::AbstractString="dasc_manual",
                          clusterdataname::AbstractString="josh1",

                          n_iters::Int64=200,
                          searchdepth::Int64=20,
                          explorationconst::Float64=2000.0,
                          q0::Float64=-1000.0,
                          maxsteps::Int64=20,
                          max_neg_reward::Float64=-1000.0,
                          step_reward::Float64=0.0,
                          maxdepth::Int64=1,

                          loginterval::Int64=100,
                          vis::Bool=true,
                          limit_members::Int64=10)
  mkpath(outdir)

  problem = ACASXClustering(runtype, data, data_meta, manuals, clusterdataname)

  mcts2_params = MCTS2ESParams(maxsteps, max_neg_reward, step_reward, n_iters, searchdepth,
                             explorationconst, q0, seed, Observer(), Observer())

  Dl = problem.Dl
  dtree, logs = train_dtree(mcts2_params, problem, Dl, maxdepth, loginterval)

  #add to log
  push!(logs, "parameters", ["seed", seed, 0])
  push!(logs, "parameters", ["runtype", runtype, 0])
  push!(logs, "parameters", ["clusterdataname", clusterdataname, 0])

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  #visualize
  if vis
    decisiontreevis(dtree, Dl, joinpath(outdir, "$(logfileroot)_vis"), limit_members,
                    FMT_PRETTY, FMT_NATURAL)
    #logvis(logs, joinpath(outdir, "$(logfileroot)_logs"))
  end

  return dtree, logs
end

end #module

