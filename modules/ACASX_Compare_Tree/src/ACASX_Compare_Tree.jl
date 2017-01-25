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
ACASX Study comparing performance of GP, MC, MCTS, GE Trees.
Single-threaded versions are used for more stable comparison.
combine_and_plot()
"""
module ACASX_Compare_Tree

export combine_and_plot, combine_mc_logs, combine_mcts_logs, combine_ge_logs, combine_gp_logs,
    master_log, master_plot, run_main

import Compat: ASCIIString, UTF8String
using GrammarExpts, GBDTs
using ExprSearch: GP, MC, MCTS, GE
using ACASX_GP_Tree, ACASX_GE_Tree, ACASX_MC_Tree, ACASX_MCTS_Tree
using LogJoiner
using RLESUtils, Loggers, MathUtils, LatexUtils, Sweeper
using DataFrames
using PGFPlots, TikzPictures

const CONFIG = "nvn_libcas098smallfilt_10K"
#const CONFIG = "nvn_dascfilt"
const STUDYNAME = "ACASX_Compare_Tree"
const MC_NAME = "ACASX_MC_Tree"
const MCTS_NAME = "ACASX_MCTS_Tree"
const GE_NAME = "ACASX_GE_Tree"
const GP_NAME = "ACASX_GP_Tree"

const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

const MASTERLOG_FILE = joinpath(RESULTDIR, STUDYNAME, "masterlog.csv.gz")
const PLOTLOG_FILE =  joinpath(RESULTDIR, STUDYNAME, "plotlog.csv.gz")
const PLOTFILEROOT = joinpath(RESULTDIR, STUDYNAME, "plots")

resultpath(dir::ASCIIString="") = joinpath(RESULTDIR, dir)
studypath(dir::ASCIIString="") = joinpath(RESULTDIR, STUDYNAME, dir)

function combine_mc_logs(dir::AbstractString=studypath(MC_NAME))
    logjoin(dir, "acasx_mc_tree_log.txt", ["classifier_metrics", 
        "interpretability_metrics", "computeinfo"], joinpath(dir, "subdirjoined"))
end
function combine_mcts_logs(dir::AbstractString=studypath(MCTS_NAME))
    logjoin(dir, "acasx_mcts_tree_log.txt", ["classifier_metrics", 
        "interpretability_metrics", "computeinfo"], joinpath(dir, "subdirjoined"))
end
function combine_ge_logs(dir::AbstractString=studypath(GE_NAME))
    logjoin(dir, "acasx_ge_tree_log.txt", ["classifier_metrics", 
        "interpretability_metrics", "computeinfo"], joinpath(dir, "subdirjoined"))
end
function combine_gp_logs(dir::AbstractString=studypath(GP_NAME))
    logjoin(dir, "acasx_gp_tree_log.txt", ["classifier_metrics", 
        "interpretability_metrics", "computeinfo"], joinpath(dir, "subdirjoined"))
end

function master_log(namedirs::Vector{Pair{Symbol,ASCIIString}}=[:MC=>studypath(MC_NAME), 
    :MCTS=>studypath(MCTS_NAME), :GE=>studypath(GE_NAME), :GP=>studypath(GP_NAME)])

    masterlog = nothing 
    for (name, dir) in namedirs
        logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))

        #handle cpu time
        Dcpu = logs["computeinfo"]
        Dcpu = Dcpu[Dcpu[:parameter].=="cpu_time", [:value,:name]]
        rename!(Dcpu, :value, :cpu_time)
        Dcpu[:cpu_time] = map(x->parse(Float64,x), Dcpu[:cpu_time])

        D = join(logs["interpretability_metrics"], logs["classifier_metrics"]; on=[:name])
        D = join(D, Dcpu; on=[:name])
        D[:algorithm] = fill(name, nrow(D))
        if masterlog == nothing
            masterlog = D
        else
            append!(masterlog, D)
        end
    end
    writetable(MASTERLOG_FILE, masterlog)
    masterlog
end

master_plot(; kwargs...) = master_plot(readtable(MASTERLOG_FILE); kwargs...)

"""
Subsamples the collected data at 'subsample' rate to plot at a lower rate than collected
"""
function master_plot(masterlog::DataFrame; subsample::Int64=25000)
    D = masterlog 

    delete!(D, :name)
    #aggregate over seed
    D = aggregate(D, [:algorithm], [mean, std, length, SEM])

    #workaround for naming in julia 0.4
    for s0 in names(D)
        str = string(s0)
        str = replace(str, "MathUtils.", "")
        s1 = Symbol(str)
        if s0 != s1
            rename!(D, s0, s1)
        end
    end

    writetable(PLOTLOG_FILE, D)
end

function combine_and_plot()
    combine_mc_logs()
    combine_mcts_logs()
    combine_ge_logs()
    combine_gp_logs()
    ml = master_log()
    master_plot(ml)
end

end #module

