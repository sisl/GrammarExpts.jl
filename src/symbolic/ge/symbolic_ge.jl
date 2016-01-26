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

module SYMBOLIC_GE

export symbolic_ge

using ExprSearch.GE
using RLESUtils.ArrayUtils
using Reexport

import GrammarExpts.CONFIG

#defaults
if !haskey(CONFIG, :config)
  CONFIG[:config] = :test
end
if !haskey(CONFIG, :gt)
  CONFIG[:gt] = :easy
end

println("Configuring: config=$(CONFIG[:config]), gt=$(CONFIG[:gt])")

include("../common/SymbolicProblem.jl")

if CONFIG[:config] == :test
  include("test_config.jl") #for testing
elseif CONFIG[:config] == :normal
  include("config.jl")
else
  error("config not valid ($(CONFIG[:config]))")
end

const GT_FILE = if CONFIG[:gt] == :easy
  joinpath(dirname(@__FILE__), "../common/gt_easy.jl")
elseif CONFIG[:gt] == :higherorder
  joinpath(dirname(@__FILE__), "../common/gt_higherorder.jl")
else
  error("gt not valid ($(CONFIG[:gt]))")
end

include("../../logs/ge_logs.jl")

using .SymbolicProblem

function symbolic_ge(outdir::AbstractString="./"; seed=1,
                     logfileroot::AbstractString="symbolic_ge_log",
                     genome_size::Int64=GENOME_SIZE,
                     pop_size::Int64=POP_SIZE,
                     maxwraps::Int64=MAXWRAPS,
                     top_percent::Float64=TOP_PERCENT,
                     prob_mutation::Float64=PROB_MUTATION,
                     mutation_rate::Float64=MUTATION_RATE,
                     default_code=DEFAULTCODE,
                     maxiterations::Int64=MAXITERATIONS,
                     gt_file::AbstractString=GT_FILE)
  srand(seed)

  problem = Symbolic(gt_file)

  observer = Observer()
  logs = default_logs(observer)
  default_console!(observer)
  @notify_observer(observer, "parameters", ["seed", seed])
  @notify_observer(observer, "parameters", ["config", CONFIG[:config]])
  @notify_observer(observer, "parameters", ["data", CONFIG[:gt]])
  @notify_observer(observer, "parameters", ["w_len", w_len])

  ge_observer = Observer()

  ge_params = GEESParams(genome_size, pop_size, maxwraps,
                         top_percent, prob_mutation, mutation_rate, default_code,
                         maxiterations, ge_observer, observer)

  result = exprsearch(ge_params, problem)

  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)

  return result
end

end #module
