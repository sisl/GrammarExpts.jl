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

@reexport using RLESUtils: Observers, Loggers
using Iterators

function default_logs(observer::Observer)
  logs = TaggedDFLogger()
  add_folder!(logs, "fitness", [Int64, Float64], ["iter", "fitness"])
  add_folder!(logs, "fitness5", [Int64, Int64, Float64],
              ["iter", "position", "fitness"])
  add_folder!(logs, "code", [Int64, ASCIIString], ["iter", "code"])
  add_folder!(logs, "pop_distr", [Int64, Float64, Int64, Int64, Int64],
              ["iter", "bin_center", "count", "unique_fitness", "unique_code"])
  add_folder!(logs, "pop_diversity", [Int64, Int64, Int64],
              ["iter", "unique_fitness", "unique_code"])
  add_folder!(logs, "iteration_time", [Int64, Float64],
              ["iter", "iteration_time_s"])
  add_folder!(logs, "computeinfo", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "parameters", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "result", [Float64, ASCIIString, Int64, Int64], ["fitness", "expr", "best_at_eval", "total_evals"])
  add_folder!(logs, "current_best", [Int64, Float64, ASCIIString], ["iter", "fitness", "expr"])

  add_observer(observer, "fitness", push!_f(logs, "fitness"))
  add_observer(observer, "fitness5", x -> begin
                 iter = x[1]
                 fitness = x[2:end]
                 for i in eachindex(fitness)
                   push!(logs, "fitness5", [iter, i, fitness[i]])
                 end
               end)
  add_observer(observer, "code", push!_f(logs, "code"))
  add_observer(observer, "population", x -> begin
                 iter, pop = x
                 fitness_vec = Float64[pop[i].fitness  for i = 1:length(pop)]
                 edges, counts = hist(fitness_vec, HIST_EDGES)
                 uniq_fitness = Int64[]
                 uniq_code = Int64[]
                 for (e1, e2) in partition(HIST_EDGES, 2, 1)
                   subids = filter(i -> e1 <= pop[i].fitness < e2, 1:length(pop))
                   subpop = pop[subids]
                   n_fit = length(unique(imap(i -> string(subpop[i].fitness), 1:length(subpop))))
                   n_code = length(unique(imap(i -> string(subpop[i].code), 1:length(subpop))))
                   push!(uniq_fitness, n_fit)
                   push!(uniq_code, n_code)
                 end
                 for (m, c, uf, uc) in zip(HIST_MIDS, counts, uniq_fitness, uniq_code)
                   push!(logs, "pop_distr", [iter, m, c, uf, uc])
                 end
               end)
  add_observer(observer, "population", x -> begin
                 iter, pop = x
                 n_fit = length(unique(imap(i -> string(pop[i].fitness), 1:length(pop))))
                 n_code = length(unique(imap(i -> string(pop[i].code), 1:length(pop))))
                 push!(logs, "pop_diversity", [iter, n_fit, n_code])
               end)
  add_observer(observer, "iteration_time", push!_f(logs, "iteration_time"))
  add_observer(observer, "computeinfo", push!_f(logs, "computeinfo"))
  add_observer(observer, "parameters", push!_f(logs, "parameters"))
  add_observer(observer, "result", push!_f(logs, "result"))
  add_observer(observer, "current_best", x -> begin
                 iter, fitness, expr = x
                 push!(logs, "current_best", [iter, fitness, string(expr)])
               end)

  return logs
end

function default_console!(observer::Observer)
  add_observer(observer, "verbose1", x -> println(x[1]))
  add_observer(observer, "current_best", x -> begin
                 iter, fitness, code = x
                 code = string(code)
                 code_short = take(code, 50) |> join
                 println("generation: $iter, max fitness=$(signif(fitness, 4)),",
                         "length=$(length(code)), code=$(code_short)")
               end)
  add_observer(observer, "result", x -> println("fitness=$(x[1]), expr=$(x[2])"))
end

