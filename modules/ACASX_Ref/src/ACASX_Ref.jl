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
Reference input for the ACASX problem.
Example usage: config=configure(ACASX_Ref, "nmac_rule", "nvn_libcas0100llcem"); acasx_ref(;config...)
"""
module ACASX_Ref

export configure, acasx_ref, setup, playsequence

import Compat.ASCIIString
using ExprSearch.Ref
using Reexport

using GrammarExpts
using ACASXProblem, DerivationTrees, DerivTreeVis, Ref_Logs
using RLESUtils, Configure, DataFrameSets
import Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")

configure(::Type{Val{:ACASX_Ref}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

function acasx_ref(; outdir::AbstractString="./ACASX_Ref",
                   seed=1, #not used, only for compatibility with other modules
                   logfileroot::AbstractString="acasx_ref_log",

                   runtype::Symbol=:nmacs_vs_nonnmacs,
                   data::AbstractString="dasc",
                   manuals::AbstractString="dasc_manual",
                   clusterdataname::AbstractString="josh1",
                   maxsteps::Int64=20,

                   actions::Vector{Int64}=Int64[2, 2, 7, 10, 7, 7, 5, 3, 10], #action seq to be replayed
                   n_samples::Int64=200, #for emulated logs

                   loginterval::Int64=100,
                   vis::Bool=true)

  mkpath(outdir)

  problem = ACASXClustering(runtype, data, manuals, clusterdataname)

  observer = Observer()
  logs = default_logs(observer, loginterval)
  default_console!(observer, loginterval)

  ref_params = RefESParams(maxsteps, actions, n_samples, observer)
  result = exprsearch(ref_params, problem)

  add_members_to_log!(logs, problem, result.expr)
  add_predicted_to_log!(logs, problem, result.expr)
  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  if vis
    derivtreevis(get_derivtree(result), joinpath(outdir, "$(logfileroot)_derivtreevis"))
  end

  return result
end

function add_members_to_log!{T}(logs::TaggedDFLogger, problem::ACASXClustering{T}, expr)
  add_folder!(logs, "members", [ASCIIString, ASCIIString], ["members_true", "members_false"])
  members_true, members_false = get_members(problem, expr)
  push!(logs, "members", [join(members_true, ","), join(members_false, ",")])
end

function add_predicted_to_log!{T}(logs::TaggedDFLogger, problem::ACASXClustering{T}, expr)
  Dl = problem.Dl
  labels = Dl.labels
  encs = getmeta(Dl)[:encounter_id]

  add_folder!(logs, "predicted", [ASCIIString, T, Bool], ["encounter", "label", "predict"])
  predicts = get_predicts(problem, expr)

  for i = 1:length(problem.Dl)
    push!(logs, "predicted", ["$(encs[i])", labels[i], predicts[i]])
  end
end

end #module
