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

function nmac_clusters(clustering::DataFrame, Ds::DFSet)
  ids = map(x -> parse(Int, x), names(Ds))
  labeldict = Dict{Int64,Int64}() #facilitate random access
  for row in eachrow(clustering)
    labeldict[row[:encounter_id]] = row[:label]
  end
  inds = find(x -> haskey(labeldict, x), ids)
  sublabels = map(x -> labeldict[x], ids[inds])
  subDs = Ds[inds]
  return DFSetLabeled(subDs, sublabels)
end

function nonnmacs_extra_cluster(clustering::DataFrame, Ds::DFSet, meta::DataFrame)
  ids = map(x -> parse(Int, x), names(Ds))
  labeldict = Dict{Int64,Int64}() #facilitate random access
  for row in eachrow(clustering)
    labeldict[row[:encounter_id]] = row[:label]
  end

  nmacdict = Dict{Int64,Bool}() #facilitate random access
  for row in eachrow(meta)
    nmacdict[row[:encounter_id]] = row[:nmac]
  end

  nonnmac_label = maximum(clustering[:label]) + 1
  labels = map(ids) do id
    label = if haskey(labeldict, id)
      labeldict[id]
    else
      @assert nmacdict[id] == false
      nonnmac_label
    end
    return label
  end
  return DFSetLabeled(Ds, labels)
end

function nmacs_vs_nonnmacs(Ds::DFSet, meta::DataFrame)
  ids = map(x -> parse(Int, x), names(Ds))
  nmac_ids = meta[meta[:nmac] .== true, :encounter_id]
  nonnmac_ids = meta[meta[:nmac] .== false, :encounter_id]
  labels = map(ids) do id
    label = if id in nmac_ids
      1
    elseif id in nonnmac_ids
      2
    else
      error("encounter id not found: $id")
    end
    return label
  end
  return DFSetLabeled(Ds, labels)
end
