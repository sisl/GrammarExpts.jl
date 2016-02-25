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

module Sweeper

export configure, sweeper

using GrammarExpts
using RLESUtils: ParamSweeps, Observers, Loggers, Vectorizer, FileUtils
using CPUTime
using RLESUtils.Configure
import RLESUtils.Configure.configure

const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")
const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")

configure(::Type{Val{:Sweeper}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

function sweeper(f::Function, result_type::Type, baseconfig::Dict{Symbol,Any}=Dict{Symbol,Any}();
                  outdir::AbstractString=RESULTDIR,
                  logfileroot::AbstractString="sweeper_log",
                  kwargs::Iterable...
                  )
  mkpath(outdir)
  script = KWParamSweep(f; kwargs...)

  #names and types of input parameters
  keynames = collect(keys(script))
  valtypes = map(x -> eltype(collect(x)), values(script))
  valtypes = convert(Vector{Type}, valtypes)

  observer = Observer()
  logs = TaggedDFLogger()
  add_folder!(logs, "result", vcat(valtypes, vectypes(result_type), Float64), vcat(keynames, vecnames(result_type), "cputime_s"))
  add_observer(observer, "result", push!_f(logs, "result"))

  results = map(script) do kvs
    dir = joinpath(outdir, filenamefriendly(string(kvs)))
    mkpath(dir)
    cd(dir)
    CPUtic()
    result = f(; baseconfig..., kvs...)
    cputime = CPUtoq()

    vs = map(x -> x[2], kvs) #input parameter values
    @notify_observer(observer, "result", vcat(vs, vectorize(result), cputime))

    return result
  end

  save_log(joinpath(outdir, logfileroot) * ".txt", logs)

  return results
end

end #module
