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

import Compat.ASCIIString
using ExprSearch.GE
using Datasets
using RLESUtils, ArrayUtils, Configure, LogSystems, Loggers
using Reexport

using GrammarExpts
using ACASXProblem, DerivTreeVis
import Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

configure(::Type{Val{:ACASX_GE}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

#nmacs vs nonnmacs
"""
Example call:
config=configure(ACASX_GE, "nvn_dasc", "normal")
acasx_ge(; config...)
"""
function acasx_ge(;outdir::AbstractString=joinpath(RESULTDIR, "./ACASX_GE"),
                  seed=1,
                  logfileroot::AbstractString="acasx_ge_log",

                  runtype::Symbol=:nmacs_vs_nonnmacs,
                  data::AbstractString="dascfilt",
                  manuals::AbstractString="dasc_manual",
                  clusterdataname::AbstractString="josh1",

                  genome_size::Int64=20,
                  pop_size::Int64=50,
                  maxwraps::Int64=0,
                  top_keep::Float64=0.25,
                  top_seed::Float64=0.5,
                  rand_frac::Float64=0.25,
                  prob_mutation::Float64=0.2,
                  mutation_rate::Float64=0.2,
                  defaultcode::Union{Symbol,Expr}=:(eval(false)),
                  maxiterations::Int64=3,

                  limit_members::Int64=30,
                  hist_nbins::Int64=40,
                  hist_edges::Range{Float64}=linspace(0.0, 200.0, hist_nbins + 1),
                  hist_mids::Vector{Float64}=collect(Base.midpoints(hist_edges)),
                  vis::Bool=true)
    srand(seed)
    mkpath(outdir)

    problem = ACASXClustering(runtype, data, manuals, clusterdataname)

    logsys = GE.logsystem()
    empty_listeners!(logsys)
    send_to!(STDOUT, logsys, ["verbose1", "current_best_print", "result"])
    logs = TaggedDFLogger()
    send_to!(logs, logsys, ["code", "computeinfo", "current_best", "elapsed_cpu_s", "fitness",
        "fitness5", "parameters", "result"])

    ge_params = GEESParams(genome_size, pop_size, maxwraps,
                         top_keep, top_seed, rand_frac, prob_mutation, mutation_rate, defaultcode,
                         maxiterations, logsys)

    result = exprsearch(ge_params, problem)

    #manually push! extra info to log
    push!(logs, "parameters", ["seed", seed])
    push!(logs, "parameters", ["runtype", runtype])
    push!(logs, "parameters", ["data", data])
    add_folder!(logs, "expression", [ASCIIString, ASCIIString, ASCIIString],
        ["raw", "pretty", "natural"]) 
    push!(logs, "expression", [string(result.expr), pretty_string(result.tree, FMT_PRETTY),
        pretty_string(result.tree, FMT_NATURAL, true)])

    #save log
    outfile = joinpath(outdir, "$(logfileroot).txt")
    save_log(outfile, logs)

    if vis
        derivtreevis(get_derivtree(result), joinpath(outdir, "$(logfileroot)_derivtreevis"))
    end

    result
end

end #module
