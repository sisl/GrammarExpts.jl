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
export run_sa, run_mc_full, run_mc_earlystop, run_mcts, run_mcts_max
export combine_sweep_logs, combine_sa_logs, combine_mc_full_logs, combine_mc_earlystop_logs, combine_mcts_logs, combine_mcts_max_logs
export master_log, master_plot

using GrammarExpts
using Sweeper
using ExprSearch: SA, MC, MCTS
using ACASX_SA, ACASX_MC, ACASX_MCTS
using LogJoiner

using RLESUtils, Loggers, MathUtils, Configure
using DataFrames
using Gadfly
import Configure.configure

const CONFIG = "nvn_libcas0100llcem"
const STUDYNAME = "ACASX_Compare"
const SA_NAME = "ACASX_SA"
const MCFULL_NAME = "ACASX_MC_full"
const MCEARLYSTOP_NAME = "ACASX_MC_earlystop"
const MCTS_NAME = "ACASX_MCTS"
const MCTSMAX_NAME = "ACASX_MCTS_MAX"

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

configure(::Type{Val{:ACASX_Compare}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

function run_sa(; seed=1:5, n_starts::Int64=20)
  baseconfig = configure(ACASX_SA, "singlethread", CONFIG) #start with base config
  baseconfig[:n_starts] = n_starts
  config = configure(ACASX_Compare, "sa") #study config that goes over base config
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, SA_NAME)
  config[:seed] = seed
  sa_results = sweeper(acasx_sa1, SAESResult, baseconfig; config...)
  sa_results
end

function run_mc_full(; seed=1:5, n_samples=10000)
  baseconfig = configure(ACASX_MC, "singlethread", CONFIG)
  baseconfig[:earlystop] = false
  baseconfig[:n_samples] = n_samples
  config = configure(ACASX_Compare, "mc")
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, MCFULL_NAME)
  config[:seed] = seed
  mc_results = sweeper(acasx_mc1, MCESResult, baseconfig; config...)
  mc_results
end

function run_mc_earlystop(; seed=1:5, n_samples=10000)
  baseconfig = configure(ACASX_MC, "singlethread", CONFIG)
  baseconfig[:earlystop] = true
  baseconfig[:n_samples] = n_samples
  config = configure(ACASX_Compare, "mc")
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, MCEARLYSTOP_NAME)
  config[:seed] = seed
  mc_results = sweeper(acasx_mc1, MCESResult, baseconfig; config...)
  mc_results
end

function run_mcts(; seed=1:5, n_iters=10000)
  baseconfig = configure(ACASX_MCTS, "normal", CONFIG)
  baseconfig[:maxmod] = false
  baseconfig[:n_iters] = n_iters
  config = configure(ACASX_Compare, "mcts")
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, MCTS_NAME)
  config[:seed] = seed
  mcts_results = sweeper(acasx_mcts, MCTSESResult, baseconfig; config...)
  mcts_results
end

function run_mcts_max(; seed=1:5, n_iters=10000)
  baseconfig = configure(ACASX_MCTS, "normal", CONFIG)
  baseconfig[:maxmod] = true
  baseconfig[:n_iters] = n_iters
  config = configure(ACASX_Compare, "mcts")
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, MCTSMAX_NAME)
  config[:seed] = seed
  mcts_results = sweeper(acasx_mcts, MCTSESResult, baseconfig; config...)
  mcts_results
end

function combine_sa_logs()
  dir = joinpath(RESULTDIR, STUDYNAME, SA_NAME)
  logjoin(dir, "acasx_sa_log.txt", ["current_best", "elapsed_cpu_s"], joinpath(dir, "subdirjoined"))
end

function combine_mc_full_logs()
  dir = joinpath(RESULTDIR, STUDYNAME, MCFULL_NAME)
  logjoin(dir, "acasx_mc_log.txt", ["current_best", "elapsed_cpu_s"], joinpath(dir, "subdirjoined"))
end

function combine_mc_earlystop_logs()
  dir = joinpath(RESULTDIR, STUDYNAME, MCEARLYSTOP_NAME)
  logjoin(dir, "acasx_mc_log.txt", ["current_best", "elapsed_cpu_s"], joinpath(dir, "subdirjoined"))
end

function combine_mcts_logs()
  dir = joinpath(RESULTDIR, STUDYNAME, MCTS_NAME)
  logjoin(dir, "acasx_mcts_log.txt", ["current_best", "elapsed_cpu_s"], joinpath(dir, "subdirjoined"))
end

function combine_mcts_max_logs()
  dir = joinpath(RESULTDIR, STUDYNAME, MCTSMAX_NAME)
  logjoin(dir, "acasx_mcts_log.txt", ["current_best", "elapsed_cpu_s"], joinpath(dir, "subdirjoined"))
end

function combine_sweep_logs()
  dir = joinpath(RESULTDIR, STUDYNAME)
  logjoin(dir, "sweeper_log.txt", ["result"], joinpath(dir, "sweepjoined"))
end

function master_log()
  masterlog = DataFrame([Int64, Float64, Float64, UTF8String, ASCIIString, UTF8String], [:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name], 0)

  #SA
  #dir = joinpath(RESULTDIR, STUDYNAME, SA_NAME)
  #logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  #D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:start, :iter, :name])
  #D[:algorithm] = fill("SA", nrow(D))
  #D[:nevals] = map(Int64, (D[:start] - 1) * maximum(D[:iter]) + D[:iter])
  #append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  #MCFULL
  dir = joinpath(RESULTDIR, STUDYNAME, MCFULL_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MC_FULL", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  #MCEARLYSTOP
  dir = joinpath(RESULTDIR, STUDYNAME, MCEARLYSTOP_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MC_EARLYSTOP", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  #MCTS
  dir = joinpath(RESULTDIR, STUDYNAME, MCTS_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MCTS", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  #MCTS_MAX
  dir = joinpath(RESULTDIR, STUDYNAME, MCTSMAX_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MCTS_MAX", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  writetable(joinpath(RESULTDIR, STUDYNAME, "masterlog.csv.gz"), masterlog)
end

function master_plot(; subsample::Int64=5000)
  dir = joinpath(RESULTDIR, STUDYNAME)
  D = readtable(joinpath(dir, "masterlog.csv.gz"))

  #aggregate over seed
  D = aggregate(D[[:nevals, :elapsed_cpu_s, :fitness, :algorithm]], [:nevals, :algorithm], [mean, std, length, SEM_ymin, SEM_ymax])
  D = D[rem(D[:nevals], subsample) .== 0, :]
  rename!(D, symbol("fitness_MathUtils.SEM_ymin"), :fitness_SEM_ymin) #workaround for naming in julia 0.4
  rename!(D, symbol("fitness_MathUtils.SEM_ymax"), :fitness_SEM_ymax) #workaround for naming in julia 0.4
  rename!(D, symbol("elapsed_cpu_s_MathUtils.SEM_ymin"), :elapsed_cpu_s_SEM_ymin) #workaround for naming in julia 0.4
  rename!(D, symbol("elapsed_cpu_s_MathUtils.SEM_ymax"), :elapsed_cpu_s_SEM_ymax) #workaround for naming in julia 0.4

  writetable(joinpath(RESULTDIR, STUDYNAME, "plotlog.csv.gz"), D)

  plotname = "nevals_vs_fitness"
  p = plot(D, x=:nevals, y=:fitness_mean, ymin=:fitness_SEM_ymin, ymax=:fitness_SEM_ymax, color=:algorithm,
           Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(joinpath(dir, "$plotname.tex"), 12cm, 6cm), p)
  draw(PDF(joinpath(dir, "$plotname.pdf"), 12cm, 6cm), p)

  plotname = "elapsed_cpu_vs_fitness"
  p = plot(D, x=:elapsed_cpu_s_mean, y=:fitness_mean, ymin=:fitness_SEM_ymin, ymax=:fitness_SEM_ymax, color=:algorithm,
           Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(joinpath(dir, "$plotname.tex"), 12cm, 6cm), p)
  draw(PDF(joinpath(dir, "$plotname.pdf"), 12cm, 6cm), p)

  plotname = "nevals_vs_elapsed_cpu"
  p = plot(D, x=:nevals, y=:elapsed_cpu_s_mean, ymin=:elapsed_cpu_s_SEM_ymin, ymax=:elapsed_cpu_s_SEM_ymax, color=:algorithm,
           Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(joinpath(dir, "$plotname.tex"), 12cm, 6cm), p)
  draw(PDF(joinpath(dir, "$plotname.pdf"), 12cm, 6cm), p)
end

#Configured for single-thread at the moment...
#Start separate sessions manually to parallelize...
function study_main()
  #do runs
  #sa = run_sa()
  mc_full = run_mc_full()
  mc_earlystop = run_mc_earlystop()
  mcts = run_mcts()
  mcts_max = run_mcts_max()

  #aggregate logs
  #combine_sa_logs()
  combine_mc_full_logs()
  combine_mc_earlystop_logs()
  combine_mcts_logs()
  combine_mcts_max_logs()
  combine_sweep_logs()
  master_log()

  #plot
  master_plot()
end

end #module
