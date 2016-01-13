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

using DecisionTrees

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
