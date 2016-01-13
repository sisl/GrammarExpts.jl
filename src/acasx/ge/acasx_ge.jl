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

module ACASX_GE

export acasx_ge, configure

using ExprSearch.GE
using Datasets
using RLESUtils.ArrayUtils
using Reexport

const DIR = dirname(@__FILE__)

function configure(; config::Symbol=:test, data::Symbol=:dasc)
  if isdefined(:CONFIGURED)
    error("Cannot configure twice")
  end

  global const CONFIGURED = true
  println("Configure: config=$config, data=$data")

  @eval include(joinpath(DIR, "../grammar/grammar_typed/GrammarDef.jl")) #grammar

  if config == :test
    @eval include(joinpath(DIR, "test_config.jl")) #for testing
  elseif config == :normal
    @eval include(joinpath(DIR, "config.jl"))
  else
    error("config not valid ($config)")
  end

  if data == :dasc
    @eval include(joinpath(DIR, "../common/data_dasc.jl"))
  elseif data == :libcas098_small
    @eval include(joinpath(DIR, "../common/data_libcas098_small.jl"))
  else
    error("data not valid ($data)")
  end

  @eval include(joinpath(DIR, "../common/labeleddata.jl"))

  @eval import ExprSearch.GE.get_fitness
  @eval include(joinpath(DIR, "fitness.jl"))
  @eval include(joinpath(DIR, "logs.jl"))

  @eval using .GrammarDef
end


#Callbacks
#################
function define_stop()
  tracker = Float64[]
  ex = quote
    function GE.stop(iter::Int64, fitness::Float64)
      if iter == 1
        empty!($tracker)
      end
      push!($tracker, fitness)

      if length($tracker) < STOP_N
        return false
      else
        last_N = ($tracker)[end - STOP_N + 1 : end]
        return elements_equal(last_N)
      end
    end
  end
  eval(ex)
end

function define_fitness{T}(Dl::DFSetLabeled{T})
  ex = quote
    function GE.get_fitness(code::Union{Expr,Symbol})
      return get_fitness(code, $Dl)
    end
  end
  eval(ex)
end

#nmacs vs nonnmacs
function acasx_ge(outdir::AbstractString="./"; seed=1,
                  runtype::AbstractString="nmacs_vs_nonnmacs",
                  clusterdataname::AbstractString="",
                  logfileroot::AbstractString="acasx_ge_log",
                  data::DFSet=DATASET,
                  data_meta::DataFrame=DATASET_META,
                  genome_size::Int64=GENOME_SIZE,
                  pop_size::Int64=POP_SIZE,
                  maxwraps::Int64=MAXWRAPS,
                  top_percent::Float64=TOP_PERCENT,
                  prob_mutation::Float64=PROB_MUTATION,
                  mutation_rate::Float64=MUTATION_RATE,
                  default_code::Expr=DEFAULTCODE,
                  maxiterations::Int64=MAXITERATIONS)
  srand(seed)

  Dl = if runtype == "nmacs_vs_nonnmacs"
    nmacs_vs_nonnmacs(data, data_meta)
  elseif runtype == "nmac_clusters"
    clustering = dataset(manuals, clusterdataname)
    nmac_clusters(clustering, data)
  elseif runtype == "nonnmacs_extra_cluster"
    clustering = dataset(manuals, clusterdataname)
    nonnmacs_extra_cluster(clustering, data, data_meta)
  else
    error("runtype not recognized ($runtype)")
  end

  define_stop()
  define_fitness(Dl)

  grammar = create_grammar()

  observer = Observer()
  add_observer(observer, "verbose1", x -> println(x[1]))
  add_observer(observer, "best_individual", x -> begin
                 iter, fitness, code = x
                 code = string(code)
                 code_short = take(code, 50) |> join
                 println("generation: $iter, max fitness=$(signif(fitness, 4)),",
                         "length=$(length(code)), code=$(code_short)")
               end)
  add_observer(observer, "result", x -> println("fitness=$(x[1]), expr=$(x[2])"))
  logs = define_logs(observer)

  ge_observer = Observer()

  ge_params = GEESParams(grammar, genome_size, pop_size, maxwraps,
                         top_percent, prob_mutation, mutation_rate, default_code,
                         maxiterations, ge_observer, observer)

  result = exprsearch(ge_params)

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  return result
end

end #module
