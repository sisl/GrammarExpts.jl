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
Filters data of 'n_before' seconds before CPA and all data after CPA.  If CPA occurs prior to 't_min',
sample uniformly between 't_min' and 't_max'.
"""
module FilterNMACInfo

export script_dasc, script_libcas098small, script_libcas0100star, script_libcas0100llcem
export remove_cpa, find_cpa

using FilterRecords
using DataFrameSets
using DataFrames
using Datasets

##############################
#Add an entry here

const DASC_FILTERED = Pkg.dir("Datasets/data/dascfilt")
const LIBCAS098SMALL_FILTERED = Pkg.dir("Datasets/data/libcas098smallfilt")
const LIBCAS0100STAR_FILTERED = Pkg.dir("Datasets/data/libcas0100starfilt")
const LIBCAS0100LLCEM_FILTERED = Pkg.dir("Datasets/data/libcas0100llcemfilt")

script_dasc() = remove_cpa("dasc", DASC_FILTERED; t_min=35, n_before=5)
script_libcas098small() = remove_cpa("libcas098small", LIBCAS098SMALL_FILTERED; t_min=35, n_before=5)
script_libcas0100star() = remove_cpa("libcas0100star", LIBCAS0100STAR_FILTERED; t_min=35, n_before=5)
script_libcas0100llcem() = remove_cpa("libcas0100llcem", LIBCAS0100LLCEM_FILTERED; t_min=35, n_before=5)

##############################

function cpa_metric(r::DataFrameRow)
  return r[:abs_alt_diff] + r[:horizontal_range] / 5
end

#returns row of cpa
function find_cpa(D::DataFrame)
  minval = realmax(Float64)
  minrow = 0
  for (i, r) in enumerate(eachrow(D))
    val = cpa_metric(r)
    if val < minval
      minval = val
      minrow = i
    end
  end
  return minrow
end

function remove_cpa(dataname::AbstractString, outdir::AbstractString; kwargs...)
  Ds = dataset(dataname)
  remove_cpa!(Ds; kwargs...)
  save_csvs(outdir, Ds)
end

function remove_cpa!(Ds::DFSet; kwargs...)
  for D in records(Ds)
    remove_cpa!(D; kwargs...)
  end
end

function remove_cpa!(D::DataFrame; t_min::Int64=35, n_before::Int64=5)
  i_cpa = find_cpa(D)

  #not in acceptable range, select randomly
  if !(t_min <= i_cpa <= nrow(D))
    i_cpa = rand(t_min:nrow(D))
  end
  deleterows!(D, (i_cpa - n_before):nrow(D))
end

############
#deprecated
function isnmac(row::DataFrameRow)
  return row[:horizontal_range] <= 500.0 && row[:abs_alt_diff] <= 100.0
end

function filter_nmac_info(f::Function, dataname::AbstractString, outdir::AbstractString)
  Ds = dataset(dataname)
  filter_by_bool!(f, Ds, n_before=5, n_after=typemax(Int64))

  save_csvs(outdir, Ds)
end

#dasc_script() = filter_nmac_info(isnmac, "dasc", DASC_FILTERED)
#libcas098small_script() = filter_nmac_info(isnmac, "libcas098small", LIBCAS098SMALL_FILTERED)
############

end #module
