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

export script_base, script_dasc, script_libcas098small, script_exampledata

const DATADIR = joinpath(dirname(@__FILE__), "..", "..", "..", "data")

const DASC_JSON = joinpath(DATADIR, "dasc/json")
const DASC_CSV = joinpath(DATADIR, "dasc/csv")
const DASC_OUT = Pkg.dir("Datasets/data/dasc") #requires Datasets to be installed
const DASC_META = Pkg.dir("Datasets/data/dasc_meta")

const LIBCAS098SMALL_JSON = joinpath(DATADIR, "libcas098small/json")
const LIBCAS098SMALL_CSV = joinpath(DATADIR, "libcas098small/csv")
const LIBCAS098SMALL_OUT = Pkg.dir("Datasets/data/libcas098small")
const LIBCAS098SMALL_META = Pkg.dir("Datasets/data/libcas098small_meta")

const EXAMPLEDATA_JSON = joinpath(DATADIR, "exampledata/json")
const EXAMPLEDATA_CSV = joinpath(DATADIR, "exampledata/csv")
const EXAMPLEDATA_OUT = Pkg.dir("Datasets/data/exampledata") #requires Datasets to be installed
const EXAMPLEDATA_META = Pkg.dir("Datasets/data/exampledata_meta")

#dasc set
function script_dasc(fromjson::Bool=true)
  script_base(DASC_JSON, DASC_CSV, DASC_OUT, DASC_META;
                 fromjson=fromjson, correct_coc=true)
end

#from APL 20151230, libcas0.9.8, MCTS iterations=500, testbatch
function script_libcas098small(fromjson::Bool=true)
  script_base(LIBCAS098SMALL_JSON, LIBCAS098SMALL_CSV, LIBCAS098SMALL_OUT, LIBCAS098SMALL_META;
                 fromjson=fromjson, correct_coc=true)
end

function script_exampledata(fromjson::Bool=true)
  script_base(EXAMPLEDATA_JSON, EXAMPLEDATA_CSV, EXAMPLEDATA_OUT, EXAMPLEDATA_META;
                 fromjson=fromjson, correct_coc=true)
end

function script_base(jsondir::AbstractString, csvdir::AbstractString,
                        datadir::AbstractString, metadir::AbstractString;
                        fromjson::Bool=true, correct_coc::Bool=true)
  if fromjson
    mkpath(csvdir)
    convert2csvs(jsondir, csvdir)
  end
  tmpdir = mktempdir()
  csvs2dataframes(csvdir, tmpdir)
  if correct_coc
    correct_coc_stays!(tmpdir)
  end

  mkpath(datadir)
  mv_files(tmpdir, datadir, name_from_id)
  add_encounter_info!(datadir)

  mkpath(metadir)
  encounter_meta(jsondir, metadir)
end
