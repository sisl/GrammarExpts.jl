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
ACASX Study comparing performance of GP, MC, MCTS, GE.
Single-threaded versions are used for more stable comparison.
Main entry: study_main()
"""
module ACASX_Compare

export run_main, plot_main
export run_mc, run_mcts, run_ge, run_gp
export combine_sweep_logs, combine_mc_logs, combine_mcts_logs, 
    combine_ge_logs, combine_gp_logs, combine_logs
export master_log, master_plot
export combine_and_plot

import Compat: ASCIIString, UTF8String
using GrammarExpts
using ExprSearch: GP, MC, MCTS, GE
using ACASX_GP, ACASX_GE, ACASX_MC, ACASX_MCTS
using LogJoiner

using RLESUtils, Loggers, MathUtils, Configure, LatexUtils, Sweeper
using DataFrames
using PGFPlots, TikzPictures
import Configure.configure

const CONFIG = "nvn_libcas098smallfilt_10K"
#const CONFIG = "nvn_dasc"
const STUDYNAME = "ACASX_Compare"
const MC_NAME = "ACASX_MC"
const MCTS_NAME = "ACASX_MCTS"
const GE_NAME = "ACASX_GE"
const GP_NAME = "ACASX_GP"

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

const MASTERLOG_FILE = joinpath(RESULTDIR, STUDYNAME, "masterlog.csv.gz")
const PLOTLOG_FILE =  joinpath(RESULTDIR, STUDYNAME, "plotlog.csv.gz")
const PLOTFILEROOT = joinpath(RESULTDIR, STUDYNAME, "plots")

configure(::Type{Val{:ACASX_Compare}}, configs::AbstractString...) = 
    configure_path(CONFIGDIR, configs...)

resultpath(dir::ASCIIString="") = joinpath(RESULTDIR, dir)
studypath(dir::ASCIIString="") = joinpath(RESULTDIR, STUDYNAME, dir)

function run_mc(; seed=1:5, n_samples=500000)
    baseconfig = configure(ACASX_MC, "normal", CONFIG)
    baseconfig[:outdir] = "./"
    baseconfig[:n_samples] = n_samples
    sweep_cfg = configure(ACASX_Compare, "mc")
    sweep_cfg[:outdir] = studypath(MC_NAME)
    sweep_cfg[:seed] = seed
    result = sweeper(acasx_mc1, MCESResult, baseconfig; sweep_cfg...)
    result
end
function run_mcts(; seed=1:5, n_iters=500000)
    baseconfig = configure(ACASX_MCTS, "normal", CONFIG)
    baseconfig[:outdir] = "./"
    baseconfig[:maxmod] = false
    baseconfig[:n_iters] = n_iters
    sweep_cfg = configure(ACASX_Compare, "mcts")
    sweep_cfg[:outdir] = studypath(MCTS_NAME)
    sweep_cfg[:seed] = seed
    result = sweeper(acasx_mcts, MCTSESResult, baseconfig; sweep_cfg...)
    result
end
function run_ge(; seed=1:5, n_iters=100)
    baseconfig = configure(ACASX_GE, "normal", CONFIG)
    baseconfig[:outdir] = "./"
    baseconfig[:maxiterations] = n_iters
    sweep_cfg = configure(ACASX_Compare, "ge")
    sweep_cfg[:outdir] = studypath(GE_NAME)
    sweep_cfg[:seed] = seed
    result = sweeper(acasx_ge, GEESResult, baseconfig; sweep_cfg...)
    result
end
function run_gp(; seed=1:5, n_iters=100)
    baseconfig = configure(ACASX_GP, "normal", CONFIG)
    baseconfig[:outdir] = "./"
    baseconfig[:iterations] = n_iters
    sweep_cfg = configure(ACASX_Compare, "gp")
    sweep_cfg[:outdir] = studypath(GP_NAME)
    sweep_cfg[:seed] = seed
    result = sweeper(acasx_gp, GPESResult, baseconfig; sweep_cfg...)
    result
end

function combine_mc_logs()
    dir = studypath(MC_NAME)
    logjoin(dir, "acasx_mc_log.txt", ["current_best", "elapsed_cpu_s"], 
        joinpath(dir, "subdirjoined"))
end
function combine_mcts_logs()
    dir = studypath(MCTS_NAME)
    logjoin(dir, "acasx_mcts_log.txt", ["current_best", "elapsed_cpu_s"], 
        joinpath(dir, "subdirjoined"))
end
function combine_ge_logs()
    dir = studypath(GE_NAME)
    logjoin(dir, "acasx_ge_log.txt", ["current_best", "elapsed_cpu_s"], 
        joinpath(dir, "subdirjoined"))
end
function combine_gp_logs()
    dir = studypath(GP_NAME)
    logjoin(dir, "acasx_gp_log.txt", ["current_best", "elapsed_cpu_s"], 
        joinpath(dir, "subdirjoined"))
end
function combine_sweep_logs()
    dir = studypath()
    logjoin(dir, "sweeper_log.txt", ["result"], joinpath(dir, "sweepjoined"))
end

#TODO: clean this up...
function master_log(; b_mc=true, b_mcts=true, b_ge=true, b_gp=true)
    masterlog = DataFrame([Int64, Float64, Float64, UTF8String, ASCIIString, UTF8String], 
        [:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name], 0)

    #MC
    if b_mc
        dir = studypath(MC_NAME)
        logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
        D[:algorithm] = fill("MC", nrow(D))
        rename!(D, :iter, :nevals)
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    #MCTS
    if b_mcts
        dir = studypath(MCTS_NAME)
        logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
        D[:algorithm] = fill("MCTS", nrow(D))
        rename!(D, :iter, :nevals)
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    #GE
    if b_ge
        dir = studypath(GE_NAME)
        logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:nevals, :name])
        D[:algorithm] = fill("GE", nrow(D))
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    #GP
    if b_gp
        dir = studypath(GP_NAME)
        logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:nevals, :name])
        D[:algorithm] = fill("GP", nrow(D))
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
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

    #aggregate over seed
    D = aggregate(D[[:nevals, :elapsed_cpu_s, :fitness, :algorithm]], [:nevals, :algorithm], 
        [mean, std, length, SEM])
    D = D[rem(D[:nevals], subsample) .== 0, :] #subsample data

    #workaround for naming in julia 0.4
    rename!(D, Symbol("fitness_MathUtils.SEM"), :fitness_SEM) 
    rename!(D, Symbol("elapsed_cpu_s_MathUtils.SEM"), :elapsed_cpu_s_SEM) 

    writetable(PLOTLOG_FILE, D)

    td = TikzDocument()
    algo_names = unique(D[:algorithm])
    n_algos = length(algo_names)

    #nevals_vs_fitness
    plotarray = Plots.Plot[]
    for i = 1:n_algos 
        D1 = D[D[:algorithm].==algo_names[i], [:nevals, :fitness_mean, :fitness_SEM]]
        push!(plotarray, Plots.Linear(D1[:nevals], D1[:fitness_mean], 
            errorBars=ErrorBars(; y=D1[:fitness_SEM]),   
            legendentry=escape_latex(algo_names[i])))
    end
    tp = PGFPlots.plot(Axis(plotarray, xlabel="Number of Evaluations", ylabel="Fitness",
        title="Fitness vs. Number of Evaluations", legendPos="north east"))
    push!(td, tp) 
    
    #nevals_vs_elapsed_cpu
    empty!(plotarray)
    for i = 1:n_algos
        D1 = D[D[:algorithm].==algo_names[i], [:nevals, :elapsed_cpu_s_mean, 
            :elapsed_cpu_s_SEM]]
        push!(plotarray, Plots.Linear(D1[:nevals], D1[:elapsed_cpu_s_mean], 
            errorBars=ErrorBars(;y=D1[:elapsed_cpu_s_SEM]), 
            legendentry=escape_latex(algo_names[i])))
    end
    tp = PGFPlots.plot(Axis(plotarray, xlabel="Number of Evaluations", ylabel="Elapsed CPU Time (s)",
        title="Elapsed CPU Time vs. Number of Evaluations", legendPos="north east"))
    push!(td, tp)

    save(PDF(PLOTFILEROOT * ".pdf"), td)
    save(TEX(PLOTFILEROOT * ".tex"), td)
end

#Configured for single-thread at the moment...
#Start separate sessions manually to parallelize...
function run_main(; 
    b_mc::Bool=false,
    b_mcts::Bool=false,
    b_ge::Bool=false,
    b_gp::Bool=false
    )

    #do runs
    if b_mc
        mc = run_mc()
        combine_mc_logs()
    end

    if b_mcts
        mcts = run_mcts()
        combine_mcts_logs()
    end

    if b_ge
        ge = run_ge()
        combine_ge_logs()
    end

    if b_gp
        gp = run_gp()
        combine_gp_logs()
    end
end

function plot_main(;
    b_mc::Bool=true,
    b_mcts::Bool=true,
    b_gp::Bool=true,
    b_ge::Bool=true)

    #meta info logs
    combine_sweep_logs()

    #create master log
    masterlog = master_log(; b_mc=b_mc, b_mcts=b_mcts, b_gp=b_gp, b_ge=b_ge)

    #plot
    master_plot(masterlog)
end

function combine_and_plot()
    combine_ge_logs()
    combine_gp_logs()
    combine_mc_logs()
    combine_mcts_logs()
    ml = master_log()
    master_plot(ml)
end

end #module

    #= #elapsed_cpu_vs_fitness =#
    #= empty!(plotarray) =#
    #= for i = 1:n_algos =#
    #=     D1 = D[D[:algorithm].==algo_names[i], [:elapsed_cpu_s_mean, :fitness_mean,  =#
    #=         :fitness_SEM]] =#
    #=     push!(plotarray, Plots.ErrorBars(D1[:elapsed_cpu_s_mean], D1[:fitness_mean],  =#
    #=         D1[:fitness_SEM], legendentry=escape_latex(algo_names[i]))) =#
    #= end =#
    #= tp = PGFPlots.plot(Axis(plotarray, xlabel="CPU Time (s)", ylabel="Fitness", =#
    #=     title="Fitness vs. Number of CPU Time", legendPos="north east")) =#
    #= push!(td, tp)  =#
    #=  =#
