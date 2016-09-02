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
Monte Carlo for the ACASX problem.
Example usage: config=configure(ACASX_MC,"normal","nvn_dasc"); acasx_mc(;config...)
"""
module ACASX_MC

export configure, acasx_mc, acasx_mc1

using ExprSearch.MC
using Reexport

using GrammarExpts
using ACASXProblem, DerivTreeVis, MC_Logs
using RLESUtils, Configure
import Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

configure(::Type{Val{:ACASX_MC}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

"""
Example call:
config=configure(ACASX_MC, "nvn_dasc", "test")
acasx_mc1(; config...)
or
config=configure(ACASX_MC, "nvn_dasc", "normal")
acasx_mc(; config...)
"""
function acasx_mc(; outdir::AbstractString=joinpath(RESULTDIR, "ACASX_MC"),
                  seed=1,
                  logfileroot::AbstractString="acasx_mc_log",

                  runtype::Symbol=:nmacs_vs_nonnmacs,
                  data::AbstractString="dasc",
                  manuals::AbstractString="dasc_manual",
                  clusterdataname::AbstractString="josh1",

                  maxsteps::Int64=20,
                  n_samples::Int64=50,
                  n_threads::Int64=1,
                  earlystop::Bool=true,
                  earlystop_div::Int64=10,

                  loginterval::Int64=100,
                  vis::Bool=true)

  srand(seed)
  mkpath(outdir)

  problem = ACASXClustering(runtype, data, manuals, clusterdataname)

  observer = Observer()
  par_observer = Observer()
  logs = default_logs(par_observer)
  default_console!(observer, loginterval)

  mc_params = MCESParams(maxsteps, n_samples, earlystop, earlystop_div, observer)
  pmc_params = PMCESParams(n_threads, mc_params, par_observer)

  result = exprsearch(pmc_params, problem)

  add_members_to_log!(logs, problem, result.expr)
  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  if vis
    derivtreevis(result.tree, joinpath(outdir, "$(logfileroot)_derivtreevis"))
  end

  return result
end

#TODO: combine these two versions
"single-thread version of acasx_mc"
function acasx_mc1(; outdir::AbstractString=joinpath(RESULTDIR, "ACASX_MC1"),
                   seed=1,
                   logfileroot::AbstractString="acasx_mc_log",

                   runtype::Symbol=:nmacs_vs_nonnmacs,
                   data::AbstractString="dasc",
                   manuals::AbstractString="dasc_manual",
                   clusterdataname::AbstractString="josh1",

                   maxsteps::Int64=20,
                   n_samples::Int64=50,
                   earlystop::Bool=true,
                   earlystop_div::Int64=10,

                   loginterval::Int64=100,
                   vis::Bool=true)

  srand(seed)
  mkpath(outdir)

  problem = ACASXClustering(runtype, data, manuals, clusterdataname)

  observer = Observer()

  logs = default_logs1(observer, loginterval)
  default_console!(observer, loginterval)

  mc_params = MCESParams(maxsteps, n_samples, earlystop, earlystop_div, observer)
  result = exprsearch(mc_params, problem)

  add_members_to_log!(logs, problem, result.expr)
  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  if vis
    derivtreevis(result.tree, joinpath(outdir, "$(logfileroot)_derivtreevis"))
  end

  return result
end

function add_members_to_log!{T}(logs::TaggedDFLogger, problem::ACASXClustering{T}, expr)
  add_folder!(logs, "members", [ASCIIString, ASCIIString], ["members_true", "members_false"])
  members_true, members_false = get_members(problem, expr)
  push!(logs, "members", [join(members_true, ","), join(members_false, ",")])
end

end #module
