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
Monte Carlo tree search for the ACASX problem.
Example usage: config=configure(ACASX_MCTS,"normal","nvn_dasc"); acasx_mcts(;config...)
"""
module ACASX_MCTS

export configure, acasx_mcts

import Compat.ASCIIString
using ExprSearch.MCTS
using Datasets
using Reexport
using JSON, GZip
using RLESUtils, FileUtils, Configure, Observers, Loggers, LogSystems
import RLESTypes.SymbolTable

using GrammarExpts
using ACASXProblem
using DerivTreeVis, MCTSTreeView
import Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

configure(::Type{Val{:ACASX_MCTS}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

"""
Example call:
config=configure(ACASX_MCTS, "nvn_dasc", "normal")
acasx_mcts(; config...)
"""
function acasx_mcts(;outdir::AbstractString=joinpath(RESULTDIR, "ACASX_MCTS"),
                    seed=1,
                    logfileroot::AbstractString="acasx_mcts_log",

                    runtype::Symbol=:nmacs_vs_nonnmacs,
                    data::AbstractString="dasc",
                    manuals::AbstractString="",
                    clusterdataname::AbstractString="",

                    n_iters::Int64=200,
                    searchdepth::Int64=20,
                    explorationconst::Float64=2000.0,
                    q0::Float64=-1000.0,
                    maxsteps::Int64=20,
                    max_neg_reward::Float64=-1000.0,
                    step_reward::Float64=0.0,
                    maxmod::Bool=false, #use the max update mod

                    loginterval::Int64=100,
                    vis::Bool=true,
                    mctstreevis::Bool=false,
                    treevis_interval::Int64=50)
    mkpath(outdir)

    problem = ACASXClustering(runtype, data, manuals, clusterdataname)

    logsys = MCTS.logsystem()
    empty_listeners!(logsys)
    send_to!(STDOUT, logsys, ["verbose1", "result"])
    send_to!(STDOUT, logsys, "current_best"; interval=loginterval)
    logs = TaggedDFLogger()
    send_to!(logs, logsys, ["computeinfo", "parameters", "result"])
    send_to!(logs, logsys,  "current_best"; interval=loginterval)
    send_to!(logs, logsys,  "elapsed_cpu_s"; interval=loginterval)

    #if mctstreevis
        #view, viewstep = viewstep_f(treevis_interval)
        #add_observer(observer, "mcts_tree", viewstep)
    #end

    mcts_params = MCTSESParams(maxsteps, max_neg_reward, step_reward, n_iters, searchdepth,
                             explorationconst, maxmod, q0, seed, logsys;
                             userargs=SymbolTable(:ids=>collect(1:length(problem.Dl))))

    result = exprsearch(mcts_params, problem)

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

    #save mcts tree
    if mctstreevis
        GZip.open(joinpath(outdir, "mctstreevis.json.gz"), "w") do f
            JSON.print(f, view.steps)
        end
    end

    textfile(joinpath(outdir, "summary.txt"), "mcts", seed=seed, n_iters=n_iters,
           fitness=result.fitness, expr=string(result.expr))

    result
end

end #module

