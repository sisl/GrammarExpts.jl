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

module ACASX_SA_Tree

export configure, acasx_sa_tree

using DecisionTrees
using ExprSearch.SA
using Datasets
using RLESUtils.Obj2Dict
using Reexport

using GrammarExpts
using Configure, ACASXProblem, SA_Tree_Logs
using DerivTreeVis, DecisionTreeVis

include("dtree_callbacks.jl")

const CONFIGDIR = joinpath(dirname(@__FILE__), "config")

configure(configs::AbstractString...) = _configure(CONFIGDIR, configs...)

function train_dtree{T}(psa_params::PSAESParams, problem::ACASXClustering, Dl::DFSetLabeled{T},
                        loginterval::Int64, maxdepth::Int64)

  logs = default_logs()
  num_data = length(Dl)
  T1 = Bool #predict_type
  T2 = Int64 #label_type

  p = DTParams(num_data, maxdepth, T1, T2)

  dtree = build_tree(p,
                     Dl, problem, psa_params, logs, loginterval) #userargs...

  return dtree, logs
end

function acasx_sa_tree(outdir::AbstractString="./"; seed=1,
                       logfileroot::AbstractString="acasx_sa_tree_log",

                       runtype::Symbol=:nmacs_vs_nonnmacs,
                       data::AbstractString="dasc",
                       data_meta::AbstractString="dasc_meta",
                       manuals::AbstractString="dasc_manual",
                       clusterdataname::AbstractString="josh1",

                       maxsteps::Int64=20,
                       T1::Float64=12.184,
                       alpha::Float64=0.99976,
                       n_epochs::Int64=50,
                       n_starts::Int64=1,
                       n_batches::Int64=1,
                       maxdepth::Int64=1,

                       loginterval::Int64=100,
                       vis::Bool=true,
                       limit_members::Int64=20)

  problem = ACASXClustering(runtype, data, data_meta, manuals, clusterdataname)

  sa_params = SAESParams(maxsteps, T1, alpha, n_epochs, n_starts, Observer())
  psa_params = PSAESParams(n_batches, sa_params)

  Dl = problem.Dl
  dtree, logs = train_dtree(psa_params, problem, Dl, loginterval, maxdepth)

  #add to log
  push!(logs, "parameters", ["seed", seed, 0])
  push!(logs, "parameters", ["runtype", runtype, 0])
  push!(logs, "parameters", ["clusterdataname", clusterdataname, 0])

  outfile = joinpath(outdir, "$(logfileroot).json")
  Obj2Dict.save_obj(outfile, dtree)
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

