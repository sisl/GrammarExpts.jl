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
Grammatical Evolution for the ACASX problem.
Example usage: config=configure(ACASX_GE,"normal","nvn_dasc"); acasx_ge(;config...)
"""
module ACASX_GE

export configure, acasx_ge

using ExprSearch.GE
using Datasets
using RLESUtils.ArrayUtils
using Reexport

using GrammarExpts
using ACASXProblem, GE_Logs, DerivTreeVis
using RLESUtils.Configure
import RLESUtils.Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")

configure(::Type{Val{:ACASX_GE}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

#nmacs vs nonnmacs
function acasx_ge(;outdir::AbstractString="./",
                  seed=1,
                  logfileroot::AbstractString="acasx_ge_log",

                  runtype::Symbol=:nmacs_vs_nonnmacs,
                  data::AbstractString="dasc",
                  data_meta::AbstractString="dasc_meta",
                  manuals::AbstractString="dasc_manual",
                  clusterdataname::AbstractString="josh1",

                  genome_size::Int64=20,
                  pop_size::Int64=50,
                  maxwraps::Int64=0,
                  top_percent::Float64=0.5,
                  prob_mutation::Float64=0.2,
                  mutation_rate::Float64=0.2,
                  defaultcode::Union{Symbol,Expr}=:(eval(false)),
                  maxiterations::Int64=3,

                  limit_members::Int64=30,
                  hist_nbins::Int64=40,
                  hist_edges::Range{Float64}=linspace(0.0, 200.0, hist_nbins + 1),
                  hist_mids::Vector{Float64}=collect(Base.midpoints(hist_edges)),
                  loginterval::Int64=100,
                  vis::Bool=true)
  srand(seed)
  mkpath(outdir)

  problem = ACASXClustering(runtype, data, data_meta, manuals, clusterdataname)

  observer = Observer()
  logs = default_logs(observer, hist_edges, hist_mids)
  default_console!(observer)

  ge_observer = Observer()

  ge_params = GEESParams(genome_size, pop_size, maxwraps,
                         top_percent, prob_mutation, mutation_rate, defaultcode,
                         maxiterations, ge_observer, observer)

  result = exprsearch(ge_params, problem)

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  if vis
    derivtreevis(result.tree, joinpath(outdir, "$(logfileroot)_derivtreevis"))
  end

  return result
end

end #module
