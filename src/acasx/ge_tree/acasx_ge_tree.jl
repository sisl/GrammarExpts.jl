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

module ACASX_GE_Tree

export acasx_ge_tree

using DecisionTrees
using ExprSearch.GE
using Datasets
using RLESUtils: ArrayUtils, Obj2Dict
using Reexport

include("../grammar/grammar_typed/GrammarDef.jl") #grammar

include("test_config.jl") #for testing
#include("config.jl")

#pick a dataset
include("../common/data_dasc.jl")
#include("../common/data_libcas098_small.jl")

include("../common/labeleddata.jl")

import ExprSearch.GE.get_fitness
include("fitness.jl")
include("logs.jl")

using .GrammarDef

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

#FIXME: rewrite these sneak-in callbacks as a convenience macro
function define_fitness{T}(Dl::DFSetLabeled{T})
  ex = quote
    function GE.get_fitness(code::Union{Expr,Symbol})
      return get_fitness(code, $Dl)
    end
  end
  eval(ex)
end

function define_truth{T}(Dl::DFSetLabeled{T})
  ex = quote
    function DecisionTrees.get_truth(members::Vector{Int64})
      return $(Dl).labels[members]
    end
  end
  eval(ex)
end

function classify(result::GEESResult, Ds::Vector{DataFrame})
  f = to_function(result.expr)
  return map(f, Ds)
end

function define_splitter{T}(Dl::DFSetLabeled{T}, ge_params::GEESParams, logs::TaggedDFLogger)
  ex = quote
    function DecisionTrees.get_splitter(members::Vector{Int64})
      set_observers!($(ge_params).observer, $logs)
      Dl_sub = $(Dl)[members]

      define_fitness(Dl_sub)

      result = exprsearch($ge_params)

      predicts = classify(result, Dl_sub.records)
      info_gain, _, _ = get_metrics(predicts, Dl_sub.labels)

      return info_gain > 0 ? result : nothing
    end
  end
  eval(ex)
end

function define_labels{T}(Dl::DFSetLabeled{T})
  ex = quote
    function DecisionTrees.get_labels(result::SearchResult, members::Vector{Int64})
      return classify(result, $(Dl).records[members])
    end
  end
  eval(ex)
end

function train_dtree{T}(ge_params::GEESParams, Dl::DFSetLabeled{T})

  logs = define_logs()
  num_data = length(Dl)
  T1 = Bool #predict_type
  T2 = Int64 #label_type

  define_truth(Dl)
  define_splitter(Dl, ge_params, logs)
  define_labels(Dl)

  p = DTParams(num_data, MAXDEPTH, T1, T2)

  dtree = build_tree(p)
  return dtree, logs
end

#nmacs vs nonnmacs
function acasx_ge_tree(outdir::AbstractString="./"; seed=1,
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

  grammar = create_grammar()
  define_stop() #stopping criterion

  logs = define_logs()

  ge_params = GEESParams(grammar, genome_size, pop_size, maxwraps,
                         top_percent, prob_mutation, mutation_rate, default_code,
                         maxiterations, Observer(), Observer())

  dtree, logs = train_dtree(ge_params, Dl)

  outfile = joinpath(outdir, "$(logfileroot).json")
  Obj2Dict.save_obj(outfile, dtree)
  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  return dtree, logs
end

end #module
