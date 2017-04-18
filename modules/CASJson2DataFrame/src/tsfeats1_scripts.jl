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

export script_base, script_dasc, script_libcas098small, script_libcas0100star, script_libcas0100llcem
export process_jsons

using FilterNMACInfo

const DATADIR = joinpath(dirname(@__FILE__), "..", "..", "..", "data")

const DASC_JSON = joinpath(DATADIR, "dasc/json")
const DASC_CSV = joinpath(DATADIR, "dasc/csv")
const DASC_OUT = Pkg.dir("Datasets/data/dasc") #requires Datasets to be installed

const LIBCAS098SMALL_JSON = joinpath(DATADIR, "libcas098small/json")
const LIBCAS098SMALL_CSV = joinpath(DATADIR, "libcas098small/csv")
const LIBCAS098SMALL_OUT = Pkg.dir("Datasets/data/libcas098small")

const LIBCAS0100STAR_JSON = joinpath(DATADIR, "libcas0100star/json")
const LIBCAS0100STAR_CSV = joinpath(DATADIR, "libcas0100star/csv")
const LIBCAS0100STAR_OUT = Pkg.dir("Datasets/data/libcas0100star")

const LIBCAS0100LLCEM_JSON = joinpath(DATADIR, "libcas0100llcem/json")
const LIBCAS0100LLCEM_CSV = joinpath(DATADIR, "libcas0100llcem/csv")
const LIBCAS0100LLCEM_OUT = Pkg.dir("Datasets/data/libcas0100llcem")

#####################
#Add an entry here

#dasc set
function script_dasc(fromjson::Bool=true)
  script_base(DASC_JSON, DASC_CSV, DASC_OUT;
                 fromjson=fromjson, correct_coc=true)
end

#from APL 20151230, libcas0.9.8, MCTS iterations=500, testbatch
function script_libcas098small(fromjson::Bool=true)
  script_base(LIBCAS098SMALL_JSON, LIBCAS098SMALL_CSV, LIBCAS098SMALL_OUT;
                 fromjson=fromjson, correct_coc=true)
end

#Generated 20160413, libcas0.10.0, MCTS iterations=3000, 2ac, stardbn
function script_libcas0100star(fromjson::Bool=true)
  script_base(LIBCAS0100STAR_JSON, LIBCAS0100STAR_CSV, LIBCAS0100STAR_OUT;
                 fromjson=fromjson, correct_coc=false)
end

#Generated 20160422, libcas0.10.0, MCTS iterations=3000, 2ac, llcemdbn
function script_libcas0100llcem(fromjson::Bool=true)
  script_base(LIBCAS0100LLCEM_JSON, LIBCAS0100LLCEM_CSV, LIBCAS0100LLCEM_OUT;
                 fromjson=fromjson, correct_coc=false)
end

#####################
function script_base(jsondir::AbstractString, csvdir::AbstractString,
                        outdir::AbstractString;
                        fromjson::Bool=true, correct_coc::Bool=true,
                        verbose::Bool=true)
  if fromjson
    verbose && println("converting to csv...")
    mkpath(csvdir)
    convert2csvs(jsondir, csvdir)
  end
  tmpdir = mktempdir()
  verbose && println("tmp=$tmpdir")
  verbose && println("converting to dataframes...")
  csvs2dataframes(csvdir, tmpdir)
  if correct_coc
    verbose && println("correcting cocs...")
    correct_coc_stays!(tmpdir)
  end
  add_encounter_info!(tmpdir)

  mkpath(outdir)
  verbose && println("saving dataset...")
  make_dataset(tmpdir, jsondir, outdir)
end

"""
Convenience function to process a new dataset from directory of json files
dataname is the name of the dataset (no spaces)
jsonpath is the directory of input json files
adds 2 new datasets to Datasets/data (original and filtered NMAC)
"""
function process_jsons(dataname::AbstractString, jsonpath::AbstractString)
    csvpath = joinpath(jsonpath, "csv") 
    outpath = Pkg.dir("Datasets", "data", dataname)
    script_base(jsonpath, csvpath, outpath; fromjson=true, correct_coc=false) 

    filtname = dataname * "filt"
    filtpath = Pkg.dir("Datasets", "data", filtname)
    filter_nmac_info(FilterNMACInfo.isnmac, dataname, filtpath)
end

