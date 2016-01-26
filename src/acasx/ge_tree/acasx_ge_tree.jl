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

module ACASX_GE_Tree

export acasx_ge_tree, configure

using DecisionTrees
using ExprSearch.GE
using Datasets
using RLESUtils: ArrayUtils, Obj2Dict
using Reexport

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

println("Configuring: config=$(CONFIG[:config]), data=$(CONFIG[:data]), vis=$(CONFIG[:vis])")

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

if CONFIG[:vis]
  include("vishelpers.jl")
  include("../common/decisiontreevis.jl")
  include("logvis.jl")
else
  error("data not valid ($(CONFIG[:vis])")
end

include("../../logs/ge_tree_logs.jl")
include("../common/format.jl")

include("dtree_callbacks.jl")

function train_dtree{T}(ge_params::GEESParams, problem::ACASXClustering, Dl::DFSetLabeled{T})

  logs = default_logs()
  num_data = length(Dl)
  T1 = Bool #predict_type
  T2 = Int64 #label_type

  p = DTParams(num_data, MAXDEPTH, T1, T2)

  dtree = build_tree(p,
                     Dl, problem, ge_params, logs) #userargs

  return dtree, logs
end

#nmacs vs nonnmacs
function acasx_ge_tree(outdir::AbstractString="./"; seed=1,
                       runtype::Symbol=:nmacs_vs_nonnmacs,
                       clusterdataname::AbstractString="",
                       logfileroot::AbstractString="acasx_ge_log",
                       data::DFSet=DATASET,
                       data_meta::DataFrame=DATASET_META,
                       genome_size::Int64=GENOME_SIZE,
                       pop_size::Int64=POP_SIZE,
                       maxwraps::Int64=MAXWRAPS,
                       top_percent::Float64=TOP_PERCENT,
                       prob_mutation::Float64=PROB_MUTATION,
                       mutation_rate::Float64=MUTATION_RATE,
                       default_code::Expr=DEFAULTCODE,
                       maxiterations::Int64=MAXITERATIONS,
                       vis::Bool=CONFIG[:vis],
                       w_ent::Float64=W_ENT,
                       w_len::Float64=W_LEN)
  srand(seed)

  problem = ACASXClustering(runtype, data, clusterdataname, data_meta, w_ent, w_len)

  ge_params = GEESParams(genome_size, pop_size, maxwraps,
                         top_percent, prob_mutation, mutation_rate, default_code,
                         maxiterations, Observer(), Observer())

  Dl = problem.Dl
  dtree, logs = train_dtree(ge_params, problem, Dl)

  #add to log
  push!(logs, "parameters", ["seed", seed, 0])
  push!(logs, "parameters", ["runtype", runtype, 0])
  push!(logs, "parameters", ["clusterdataname", clusterdataname, 0])
  push!(logs, "parameters", ["config", CONFIG[:config], 0])
  push!(logs, "parameters", ["data", CONFIG[:data], 0])
  push!(logs, "parameters", ["treevis", CONFIG[:vis], 0])

  outfile = joinpath(outdir, "$(logfileroot).json")
  Obj2Dict.save_obj(outfile, dtree)
  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  #visualize
  if vis
    decisiontreevis(dtree, Dl, joinpath(outdir, "$(logfileroot)_vis"))
    logvis(logs, joinpath(outdir, "$(logfileroot)_logs"))
  end

  return dtree, logs
end

end #module
