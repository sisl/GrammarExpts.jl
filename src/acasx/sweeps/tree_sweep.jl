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

using GrammarExpts
using RLESUtils: ParamSweeps, Observers, Loggers
using CPUTime

load_expt(EXPT, data=DATA, config=CONFIG, vis=VIS)

mkpath(OUTDIR)

function caller_f(func::Function, outdir::AbstractString, logfileroot::AbstractString, observer::Observer)
  f = function caller(seed::Int64, n_iters::Int64, ec::Float64)
    CPUtic()
    #make a subdirectory for logs for this run
    subdir = joinpath(OUTDIR, "$(LOGFILEROOT)_seed$(seed)_niters$(n_iters)_ec$(ec)")
    mkpath(subdir)

    dtree, logs = func(subdir, seed=seed, n_iters=n_iters, exploration_const=ec)

    rootreward = logs["result"][1,:total_reward]
    rootexpr = logs["result"][1,:expr]
    @notify_observer(observer, "result", [seed, n_iters, ec, rootreward, rootexpr])
  end
  return f
end

#observer for this study
observer = Observer()
logger = DataFrameLogger([Int64, Int64, Float64, Float64, ASCIIString],
                         ["seed", "n_iters", "exploration_const", "root_reward", "root_expr"])
add_observer(observer, "result", push!_f(logger))

