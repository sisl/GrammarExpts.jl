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
module ACASX_SAMC_Study

export study_main
export run_sa, run_mc_full, run_mc_earlystop
export combine_sweep_logs, combine_sa_logs, combine_mc_full_logs, combine_mc_earlystop_logs, master_log, master_plot

using GrammarExpts
using Sweeper
using ExprSearch: SA, MC
using ACASX_SA, ACASX_MC
using LogJoiner

using RLESUtils: Loggers, MathUtils
using DataFrames
using Gadfly

const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")
const CONFIG = "nvn_libcas098small"
const STUDYNAME = "ACASX_SAMC_Study"
const SA_NAME = "ACASX_SA"
const MCFULL_NAME = "ACASX_MC_full"
const MCEARLYSTOP_NAME = "ACASX_MC_earlystop"

function run_sa()
  baseconfig = configure(ACASX_SA, "singlethread", CONFIG)
  baseconfig[:n_starts] = 20 #for now...
  config = configure(Sweeper, "sa_acasx_samc")
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, SA_NAME)
  config[:seed] = 1:5
  sa_results = sweeper(acasx_sa1, SAESResult, baseconfig; config...)
  sa_results
end

function run_mc_full()
  baseconfig = configure(ACASX_MC, "singlethread", CONFIG)
  baseconfig[:earlystop] = false
  baseconfig[:n_samples] = 100000 #for now...
  config = configure(Sweeper, "mc_acasx_samc")
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, MCFULL_NAME)
  config[:seed] = 1:5
  mc_results = sweeper(acasx_mc1, MCESResult, baseconfig; config...)
  mc_results
end

function run_mc_earlystop()
  baseconfig = configure(ACASX_MC, "singlethread", CONFIG)
  baseconfig[:earlystop] = true
  baseconfig[:n_samples] = 100000 #for now...
  config = configure(Sweeper, "mc_acasx_samc")
  config[:outdir] = joinpath(RESULTDIR, STUDYNAME, MCEARLYSTOP_NAME)
  config[:seed] = 1:5
  mc_results = sweeper(acasx_mc1, MCESResult, baseconfig; config...)
  mc_results
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

function combine_sweep_logs()
  dir = joinpath(RESULTDIR, STUDYNAME)
  logjoin(dir, "sweeper_log.txt", ["result"], joinpath(dir, "sweepjoined"))
end

function master_log()
  masterlog = DataFrame([Int64, Float64, Float64, UTF8String, ASCIIString, UTF8String], [:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name], 0)

  dir = joinpath(RESULTDIR, STUDYNAME, SA_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:start, :iter, :name])
  D[:algorithm] = fill("SA", nrow(D))
  D[:nevals] = map(Int64, (D[:start] - 1) * maximum(D[:iter]) + D[:iter])
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  dir = joinpath(RESULTDIR, STUDYNAME, MCFULL_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MC_FULL", nrow(D))
  rename!(D, :iter, :nevals)
  append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])

  dir = joinpath(RESULTDIR, STUDYNAME, MCEARLYSTOP_NAME)
  logs = load_log(TaggedDFLogger, joinpath(dir, "subdirjoined.txt"))
  D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
  D[:algorithm] = fill("MC_EARLYSTOP", nrow(D))
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

  writetable(joinpath(RESULTDIR, STUDYNAME, "plotlog.csv.gz"), D)

  p = plot(D, x=:nevals, y=:fitness_mean, ymin=:fitness_SEM_ymin, ymax=:fitness_SEM_ymax, color=:algorithm,
           Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(joinpath(dir, "nevals_vs_fitness.tex"), 12cm, 6cm), p)
  draw(PDF(joinpath(dir, "nevals_vs_fitness.pdf"), 12cm, 6cm), p)

  p = plot(D, x=:elapsed_cpu_s_mean, y=:fitness_mean, ymin=:fitness_SEM_ymin, ymax=:fitness_SEM_ymax, color=:algorithm,
           Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(joinpath(dir, "cputime_vs_fitness.tex"), 12cm, 6cm), p)
  draw(PDF(joinpath(dir, "cputime_vs_fitness.pdf"), 12cm, 6cm), p)

  p = plot(D, x=:nevals, y=:elapsed_cpu_s_mean, ymin=:elapsed_cpu_s_SEM_ymin, ymax=:elapsed_cpu_s_SEM_ymax, color=:algorithm,
           Guide.title(CONFIG), Geom.line, Geom.errorbar);
  draw(PGF(joinpath(dir, "nevals_vs_cputime.tex"), 12cm, 6cm), p)
  draw(PDF(joinpath(dir, "nevals_vs_cputime.pdf"), 12cm, 6cm), p)
end

function study_main()
  sa = run_sa()
  mc_full = run_mc_full()
  mc_earlystop = run_mc_earlystop()

  combine_sa_logs()
  combine_mc_full_logs()
  combine_mc_earlystop_logs()
  combine_sweep_logs()

  master_log()
  master_plot()
end

end #module
