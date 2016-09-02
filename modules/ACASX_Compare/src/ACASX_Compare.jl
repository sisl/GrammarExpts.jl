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
ACASX Study comparing performance of SA, MC with full evaluations, and MC with early stop.
Single-threaded versions are used for more stable comparison.
Main entry: study_main()
"""
module ACASX_Compare

export study_main
export run_ref, run_mc_full, run_mcts
export combine_sweep_logs, combine_ref_logs, combine_mc_full_logs, combine_mcts_logs
export master_log, master_plot

using GrammarExpts
using Sweeper
using ExprSearch: Ref, SA, MC, MCTS
using ACASX_Ref, ACASX_SA, ACASX_MC, ACASX_MCTS
using LogJoiner

using RLESUtils, Loggers, MathUtils, Configure
using DataFrames
using Gadfly
import Configure.configure

const CONFIG = "nvn_dasc"
const STUDYNAME = "ACASX_Compare"
const REF_NAME = "ACASX_Ref"
const SA_NAME = "ACASX_SA"
const MCFULL_NAME = "ACASX_MC_full"
const MCTS_NAME = "ACASX_MCTS"

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

const MASTERLOG_FILE = joinpath(RESULTDIR, STUDYNAME, "masterlog.csv.gz")
const PLOTLOG_FILE =  joinpath(RESULTDIR, STUDYNAME, "plotlog.csv.gz")

configure(::Type{Val{:ACASX_Compare}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

resultpath(dir::ASCIIString="") = joinpath(RESULTDIR, dir)
studypath(dir::ASCIIString="") = joinpath(RESULTDIR, STUDYNAME, dir)

#TODO: to fix: configuration is distributed, too many magic numbers
function run_ref(; seed=1:1, n_samples::Int64=10000)
  baseconfig = configure(ACASX_Ref, "nmac_rule", CONFIG) #start with base config
  baseconfig[:outdir] = "./"
  baseconfig[:n_samples] = n_samples
  sweep_cfg = configure(ACASX_Compare, "ref") #study config that goes over base config
  sweep_cfg[:outdir] = studypath(REFNAME)
  sweep_cfg[:seed] = seed
  ref_results = sweeper(acasx_ref, RefESResult, baseconfig; sweep_cfg...)
  ref_results
end
function run_mc_full(; seed=1:5, n_samples=10000)
  baseconfig = configure(ACASX_MC, "singlethread", CONFIG)
  baseconfig[:outdir] = "./"
  baseconfig[:earlystop] = false #FIXME: locally overriding (in all of these!) is not a good idea!
  baseconfig[:n_samples] = n_samples
  sweep_cfg = configure(ACASX_Compare, "mc")
  sweep_cfg[:outdir] = studypath(MCFULL_NAME)
  sweep_cfg[:seed] = seed
  mc_results = sweeper(acasx_mc1, MCESResult, baseconfig; sweep_cfg...)
  mc_results
end
function run_mcts(; seed=1:5, n_iters=10000)
  baseconfig = configure(ACASX_MCTS, "normal", CONFIG)
  baseconfig[:outdir] = "./"
  baseconfig[:maxmod] = false
  baseconfig[:n_iters] = n_iters
  sweep_cfg = configure(ACASX_Compare, "mcts")
  sweep_cfg[:outdir] = studypath(MCTS_NAME)
  sweep_cfg[:seed] = seed
  mcts_results = sweeper(acasx_mcts, MCTSESResult, baseconfig; sweep_cfg...)
  mcts_results
end

function combine_ref_logs()
    dir = studypath(REF_NAME)
    logjoin(dir, "acasx_ref_log.txt", ["current_best", "elapsed_cpu_s"], 
        joinpath(dir, "subdirjoined"))
end
function combine_mc_full_logs()
    dir = studypath(MCFULL_NAME)
    logjoin(dir, "acasx_mc_log.txt", ["current_best", "elapsed_cpu_s"], 
        joinpath(dir, "subdirjoined"))
end
function combine_mcts_logs()
    dir = studypath(MCTS_NAME)
    logjoin(dir, "acasx_mcts_log.txt", ["current_best", "elapsed_cpu_s"], 
        joinpath(dir, "subdirjoined"))
end
function combine_sweep_logs()
    dir = studypath()
    logjoin(dir, "sweeper_log.txt", ["result"], joinpath(dir, "sweepjoined"))
end

#TODO: make these better...
function master_log()
  masterlog = DataFrame([Int64, Float64, Float64, UTF8String, ASCIIString, UTF8String], 
    [:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name], 0)

  #REF
  dir = studypath(REF_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("REF", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])


  #MCFULL
  dir = studypath(MCFULL_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MC_FULL", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  #MCTS
  dir = studypath(MCTS_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MCTS", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  writetable(MASTERLOG_FILE, masterlog)
  masterlog
end

"""
Subsamples the collected data at 'subsample' rate to plot at a lower rate than collected
"""
function master_plot(masterlog::DataFrame; subsample::Int64=5000)
  D = masterlog 

  #aggregate over seed
  D = aggregate(D[[:nevals, :elapsed_cpu_s, :fitness, :algorithm]], [:nevals, :algorithm], 
    [mean, std, length, SEM_ymin, SEM_ymax])
  D = D[rem(D[:nevals], subsample) .== 0, :] #subsample data

  #workaround for naming in julia 0.4
  rename!(D, symbol("fitness_MathUtils.SEM_ymin"), :fitness_SEM_ymin) 
  rename!(D, symbol("fitness_MathUtils.SEM_ymax"), :fitness_SEM_ymax) 
  rename!(D, symbol("elapsed_cpu_s_MathUtils.SEM_ymin"), :elapsed_cpu_s_SEM_ymin) 
  rename!(D, symbol("elapsed_cpu_s_MathUtils.SEM_ymax"), :elapsed_cpu_s_SEM_ymax) 

  writetable(PLOTLOG_FILE, D)

  plotname = "nevals_vs_fitness"
  p = plot(D, x=:nevals, y=:fitness_mean, ymin=:fitness_SEM_ymin, ymax=:fitness_SEM_ymax, 
    color=:algorithm, Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(studypath("$plotname.tex"), 12cm, 6cm), p)
  draw(PDF(studypath("$plotname.pdf"), 12cm, 6cm), p)

  plotname = "elapsed_cpu_vs_fitness"
  p = plot(D, x=:elapsed_cpu_s_mean, y=:fitness_mean, ymin=:fitness_SEM_ymin, 
    ymax=:fitness_SEM_ymax, Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(studypath("$plotname.tex"), 12cm, 6cm), p)
  draw(PDF(studypath("$plotname.pdf"), 12cm, 6cm), p)

  plotname = "nevals_vs_elapsed_cpu"
  p = plot(D, x=:nevals, y=:elapsed_cpu_s_mean, ymin=:elapsed_cpu_s_SEM_ymin, 
    ymax=:elapsed_cpu_s_SEM_ymax, Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(studypath("$plotname.tex"), 12cm, 6cm), p)
  draw(PDF(studypath("$plotname.pdf"), 12cm, 6cm), p)
end

#Configured for single-thread at the moment...
#Start separate sessions manually to parallelize...
function study_main(; 
    b_ref::Bool=false,
    b_mc_full::Bool=false,
    b_mcts::Bool=false,
    b_ge::Bool=false
    )

    #do runs
    if b_ref
        ref = run_ref()
        combine_ref_logs()
    end

    if b_mc_full
        mc_full = run_mc_full()
        combine_mc_full_logs()
    end

    if b_mcts
        mcts = run_mcts()
        combine_mcts_logs()
    end

    #meta info logs
    combine_sweep_logs(; b_ref=b_ref, b_mc_full=b_mc_full, b_mcts=b_mcts, b_ge=b_ge)

    masterlog = master_log()

    #plot
    master_plot(masterlog)
end

end #module


#################################
# Extra code that might be revived later...

  #SA
  #dir = studypath(SA_NAME)
  #logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  #D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:start, :iter, :name])
  #D[:algorithm] = fill("SA", nrow(D))
  #D[:nevals] = map(Int64, (D[:start] - 1) * maximum(D[:iter]) + D[:iter])
  #append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])


  #MCEARLYSTOP 
  #=
  dir = studypath(MCEARLYSTOP_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MC_EARLYSTOP", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
  =#

  #MCTS_MAX
  #=
  dir = studypath(MCTSMAX_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MCTS_MAX", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
  =#

#=
function run_sa(; seed=1:5, n_starts::Int64=20)
  baseconfig = configure(ACASX_SA, "singlethread", CONFIG) #start with base config
  baseconfig[:outdir] = "./"
  baseconfig[:n_starts] = n_starts
  sweep_cfg = configure(ACASX_Compare, "sa") #study config that goes over base config
  sweep_cfg[:outdir] = studypath(SA_NAME)
  sweep_cfg[:seed] = seed
  sa_results = sweeper(acasx_sa1, SAESResult, baseconfig; sweep_cfg...)
  sa_results
end
=#

#=
function run_mc_earlystop(; seed=1:5, n_samples=10000)
  baseconfig = configure(ACASX_MC, "singlethread", CONFIG)
  baseconfig[:outdir] = "./"
  baseconfig[:earlystop] = true
  baseconfig[:n_samples] = n_samples
  sweep_cfg = configure(ACASX_Compare, "mc")
  sweep_cfg[:outdir] = studypath(MCEARLYSTOP_NAME)
  sweep_cfg[:seed] = seed
  mc_results = sweeper(acasx_mc1, MCESResult, baseconfig; sweep_cfg...)
  mc_results
end
=#

#=
function run_mcts_max(; seed=1:5, n_iters=10000)
  baseconfig = configure(ACASX_MCTS, "normal", CONFIG)
  baseconfig[:outdir] = "./"
  baseconfig[:maxmod] = true
  baseconfig[:n_iters] = n_iters
  sweep_cfg = configure(ACASX_Compare, "mcts")
  sweep_cfg[:outdir] = studypath(MCTSMAX_NAME)
  sweep_cfg[:seed] = seed
  mcts_results = sweeper(acasx_mcts, MCTSESResult, baseconfig; sweep_cfg...)
  mcts_results
end
=#

#=
function combine_sa_logs()
  dir = studypath(SA_NAME)
  logjoin(dir, "acasx_sa_log.txt", ["current_best", "elapsed_cpu_s"], 
    joinpath(dir, "subdirjoined"))
end
=#

#=
function combine_mc_earlystop_logs()
  dir = studypath(MCEARLYSTOP_NAME)
  logjoin(dir, "acasx_mc_log.txt", ["current_best", "elapsed_cpu_s"], 
    joinpath(dir, "subdirjoined"))
end
=#

#=
function combine_mcts_max_logs()
  dir = studypath(MCTSMAX_NAME)
  logjoin(dir, "acasx_mcts_log.txt", ["current_best", "elapsed_cpu_s"], 
    joinpath(dir, "subdirjoined"))
end
=#

