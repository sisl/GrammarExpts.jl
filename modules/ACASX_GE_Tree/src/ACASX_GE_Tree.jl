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
Create a decision tree to recursively split encounters in the ACASX Problem. GE algorithm.
Example usage: config=configure(ACASX_GE_Tree,"normal","nvn_dasc"); acasx_ge_tree(;config...)
"""
module ACASX_GE_Tree

export configure, acasx_ge_tree

import Compat.ASCIIString

using ExprSearch.GE
using Datasets
using RLESUtils, Configure, Loggers, LogSystems
using JLD

using GrammarExpts, GBDTs, ACASXProblem, DecisionTreeVis
import Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")
const T1 = Bool #predict_type
const T2 = Int64 #label_type

configure(::Type{Val{:ACASX_GE_Tree}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

"""
Example call:
config=configure(ACASX_GE_Tree, "nvn_dasc", "normal")
acasx_ge_tree(; config...)
"""
function acasx_ge_tree(;outdir::AbstractString=joinpath(RESULTDIR, "./ACASX_GE_Tree"),
                        seed=1,
                        logfileroot::AbstractString="acasx_ge_tree_log",

                        #dataset
                        runtype::Symbol=:nmacs_vs_nonnmacs,
                        data::AbstractString="dasc",
                        manuals::AbstractString="dasc_manual",
                        clusterdataname::AbstractString="josh1",

                        #decision tree
                        maxdepth::Int64=1,

                        #GE
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
        
                        #DT vis
                        vis::Bool=true,
                        plotpdf::Bool=true,
                        limit_members::Int64=10,

                        #save tree
                        b_jldsave::Bool=true
                        )

    srand(seed)
    mkpath(outdir)

    problem = ACASXClustering(runtype, data, manuals, clusterdataname)

    ge_logsys = GE.logsystem()
    send_to!(STDOUT, ge_logsys, ["verbose1", "current_best_print"])
    ge_params = GEESParams(genome_size, pop_size, maxwraps, top_keep, top_seed, 
        rand_frac, prob_mutation, mutation_rate, defaultcode, maxiterations, ge_logsys)
    gbdt_logsys = GBDTs.logsystem()
    send_to!(STDOUT, gbdt_logsys, ["verbose1", "split_result_print"])
    logs = TaggedDFLogger()
    send_to!(logs, gbdt_logsys, ["computeinfo", "parameters", "elapsed_cpu_s", 
        "members", "classifier_metrics", "interpretability_metrics", "split_result"])

    gbdt_params = GBDTParams(problem, length(problem.Dl), ge_params, maxdepth, 
        T1, T2, gbdt_logsys)
  
    result = induce_tree(gbdt_params)

    ##################################
    #add local items to log
    push!(logs, "parameters", ["seed", seed])
    push!(logs, "parameters", ["runtype", runtype])
    push!(logs, "parameters", ["data", data])

    outfile = joinpath(outdir, "$(logfileroot).txt")
    save_log(outfile, logs)
    ##################################

    #visualize
    if vis
        decisiontreevis(result.decision_tree, problem.Dl, 
            joinpath(outdir, "$(logfileroot)_vis"), limit_members, FMT_PRETTY, 
            FMT_NATURAL; plotpdf=plotpdf)
    end
    
    if b_jldsave
        jldfile = joinpath(outdir, "save.jld")
        save(jldfile, "dtree", result.decision_tree)
    end
    result
end

end #module
