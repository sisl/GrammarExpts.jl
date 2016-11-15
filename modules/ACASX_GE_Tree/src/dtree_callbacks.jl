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

#TODO: can all the dtree_callbacks be unified?

using DecisionTrees
using RLESUtils, Observers, Loggers
import DecisionTreeVis: get_tree, get_metric
import DecisionTrees.classify

function DecisionTrees.get_truth{T}(members::Vector{Int64},
                                 Dl::DFSetLabeled{T}, otherargs...) #userargs...
  return labels(Dl, members)
end

function classify(problem::ACASXClustering, result::GEESResult, Ds::Vector{DataFrame})
  map(D -> eval_expr(problem, result.expr, D), Ds)
end

function DecisionTrees.get_splitter{T}(members::Vector{Int64},
                                       Dl::DFSetLabeled{T}, problem::ACASXClustering,
                                       ge_params::GEESParams, logs::TaggedDFLogger,
                                       hist_edges::Range{Float64}, 
                                       hist_mids::Vector{Float64}) #userargs...
  set_observers!(ge_params.observer, logs, hist_edges, hist_mids)

  problem.Dl = Dl_sub = Dl[members] #fitness function uses problem.Dl
  result = exprsearch(ge_params, problem)

  push_members!(logs, problem, result.expr)
  @notify_observer(ge_params.observer, "expression",
                   [string(result.expr),
                    pretty_string(result.tree, FMT_PRETTY),
                    pretty_string(result.tree, FMT_NATURAL, true)])

  predicts = classify(problem, result, getrecords(Dl_sub))
  info_gain, _, _ = gini_metrics(predicts, labels(Dl_sub)) #TODO: try to get rid of this, it's being used separately in get_fitness

  return info_gain > 0 ? result : nothing
end

function push_members!{T}(logs::TaggedDFLogger, problem::ACASXClustering{T}, expr)
  decision_id = nrow(logs["members"]) > 0 ?
    maximum(logs["members"][:decision_id]) + 1 : 1
  members_true, members_false = ACASXProblem.get_members(problem, expr)
  push!(logs, "members", [join(members_true, ","), join(members_false, ","), decision_id])
end

function DecisionTrees.get_labels{T}(result::SearchResult, members::Vector{Int64},
                                     Dl::DFSetLabeled{T}, problem::ACASXClustering, otherargs...) #userargs...
  return classify(problem, result, getrecords(Dl, members))
end

DecisionTreeVis.get_tree(result::GEESResult) = result.tree
DecisionTreeVis.get_metric(result::GEESResult) = result.fitness
