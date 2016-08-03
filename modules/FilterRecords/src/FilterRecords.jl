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
Filter records in a DataFrame with ability to clear number of rows before and after some occurrence
"""
module FilterRecords

export filter_by_bool!, all_occurrences

using DataFrameSets
using DataFramesMeta

"""
Same as filter_by_bool! over a single dataframe D, but applies it over all records in a DFSet
"""
function filter_by_bool!(f::Function, Ds::DFSet; kwargs...)
  for D in getrecords(Ds)
    filter_by_bool!(f, D; kwargs...)
  end
end

"""
Filter rows of dataframe D according to callback f.  Rows with f returning true are filtered,
along with n_before rows before occurrence, and n_after rows after occurrence.
Safe to use typemax for n_before and n_after.
f is a callback of the form bool=f(row::DataFrameRow).
"""
function filter_by_bool!(f::Function, D::DataFrame;
                        n_before::Int64=0, #safe to use typemax for all previous
                        n_after::Int64=0) #safe to use typemax for all after
  @assert n_before >= 0 && n_after >= 0

  remrows = IntSet()
  rows = all_occurrences(f, D)
  for r in rows
    istart = max(r - n_before, 1)
    n_after = min(n_after, nrow(D)) #workaround for typemax(Int64) case
    iend = min(r + n_after, nrow(D))
    push!(remrows, istart:iend...)
  end

  if !isempty(remrows)
    deleterows!(D, [remrows...])
  end
end

"""
Returns indices of all rows in dataframe 'D' where f returns true.
f is a callback of the form bool=f(row::DataFrameRow).
"""
function all_occurrences(f::Function, D::DataFrame)
  rows = IntSet()
  for (i, r) in enumerate(eachrow(D))
    f(r) && push!(rows, i)
  end
  return rows
end

end #module
