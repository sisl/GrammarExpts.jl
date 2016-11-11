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
Runs through the entire ACAS X pipeline from jsons to test output. Includes:
processing data from json, creating a filtered dataset, running acasx_mc_tree.
Intended use is to duplicate a small test dataset (e.g., DASC) and perform a test run.

Usage:

ptest = pipelinetest() #produces data under GrammarExpts/test/PipelineTest

#inspect Dataset and processing, if desired

cleanup(ptest) #remove created artifacts, except results
"""
module PipelineTest

export pipelinetest, cleanup

import Compat.ASCIIString
using GrammarExpts, CASJson2DataFrame, FilterRecords
using DataFrameSets
using DataFrames
using Datasets
using Base.Test

using ACASX_MC, ACASX_MC_Tree
using FilterNMACInfo

const DIR = dirname(@__FILE__)
const DATADIR = joinpath(DIR, "..", "..", "..", "data")
const DASC_JSON = joinpath(DATADIR, "dasc", "json")
const TESTDIR = joinpath(DIR, "..", "..", "..", "test", "PipelineTest")

type PTest
  dirs::Vector{ASCIIString}
end

function pipelinetest(orig_dataname::AbstractString="dasc", 
    orig_jsondir::AbstractString=DASC_JSON, 
    test_dataname::AbstractString="exampledata")
  test_csvdir = joinpath(DATADIR, test_dataname, "csv")
  test_datadir = Pkg.dir("Datasets", "data", test_dataname)

  orig_datafilt = orig_dataname * "filt"
  test_datafilt = test_dataname * "filt"
  test_datafiltdir = Pkg.dir("Datasets", "data", test_datafilt)

  #process jsons
  script_base(orig_jsondir, test_csvdir, test_datadir; 
    fromjson=true, correct_coc=true)

  #check data
  dset1 = dataset(orig_dataname)
  dset2 = dataset(test_dataname)
  for i in length(dset1)
    @test getrecords(dset1, i) == getrecords(dset2, i)
  end

  #test run
  outdir = joinpath(TESTDIR, test_dataname)
  acasx_mc(outdir=outdir, data=test_dataname)

  #generate filtered data
  remove_cpa(test_dataname, test_datafiltdir; t_min=35, n_before=5)

  #check filt data
  dset1 = dataset(orig_datafilt)
  dset2 = dataset(test_datafilt)
  for i in length(dset1)
    @test getrecords(dset1, i) == getrecords(dset2, i)
  end

  #test run on filt
  outdir = joinpath(TESTDIR, test_datafilt)
  acasx_mc_tree(outdir=outdir, data=test_datafilt)

  #for cleanup
  intermediatedir = joinpath(DATADIR, test_dataname)
  PTest([intermediatedir, test_datadir, test_datafiltdir])
end

function cleanup(ptest::PTest)
  for d in ptest.dirs
    rm(d; recursive=true)
  end
end

end #module
