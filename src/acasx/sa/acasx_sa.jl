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

module ACASX_SA

export acasx_sa, acasx_temp_params

using ExprSearch.SA
using Reexport

import GrammarExpts.CONFIG

#defaults
if !haskey(CONFIG, :config)
  CONFIG[:config] = :test
end
if !haskey(CONFIG, :data)
  CONFIG[:data] = :dasc
end
if !haskey(CONFIG, :vis)
  CONFIG[:vis] = true
end

println("Configuring: config=$(CONFIG[:config]), data=$(CONFIG[:data]), vis=$(CONFIG[:vis])")

include("../common/ACASXProblem.jl")
using .ACASXProblem

if CONFIG[:config] == :test
  include("test_config.jl") #for testing
elseif CONFIG[:config] == :normal
  include("config.jl")
elseif CONFIG[:config] == :highest
  include("highest_config.jl")
else
  error("config not valid ($config)")
end

if CONFIG[:data] == :dasc
  include("../common/data_dasc.jl")
elseif CONFIG[:data] == :libcas098_small
  include("../common/data_libcas098_small.jl")
else
  error("data not valid ($data)")
end

#include("../../logs/sa_logs.jl")
include("../common/format.jl")

if CONFIG[:vis]
  include("../common/derivtreevis.jl") #derivation tree
end

function acasx_sa(outdir::AbstractString="./"; seed=1,
                  logfileroot::AbstractString="acasx_sa_log",
                  runtype::Symbol=:nmacs_vs_nonnmacs,
                  clusterdataname::AbstractString="",
                  data::DFSet=DATASET,
                  data_meta::DataFrame=DATASET_META,
                  maxsteps::Int64=MAXSTEPS,
                  T1::Float64=T_INIT,
                  alpha::Float64=ALPHA,
                  n_epochs::Int64=N_EPOCHS,
                  n_batches::Int64=N_BATCHES,
                  n_starts::Int64=N_STARTS,
                  observer::Observer=Observer())

  srand(seed)

  problem = ACASXClustering(runtype, data, clusterdataname, data_meta)

  add_observer(observer, "temperature", x -> begin
                 if rem(x[2], LOGINTERVAL) == 0
                   println("start=$(x[1]), i=$(x[2]), T=$(x[3])")
                 end
               end)
  add_observer(observer, "current_best", x -> begin
                 if rem(x[2], LOGINTERVAL) == 0
                   println("start=$(x[1]), i=$(x[2]), fitness=$(x[3]), expr=$(x[4])")
                 end
               end)
  #logs = default_logs(observer)
  #default_console!(observer)

  sa_params = SAESParams(maxsteps, T1, alpha, n_epochs, n_starts, observer)
  psa_params = PSAESParams(n_batches, sa_params)

  #result = exprsearch(sa_params, problem)
  result = exprsearch(psa_params, problem)

  return result
end


function acasx_temp_params(P1::Float64=0.8; seed=1,
                              n_epochs::Int64=N_EPOCHS,
                              Tfinal::Float64=1.0,
                              runtype::Symbol=:nmacs_vs_nonnmacs,
                              clusterdataname::AbstractString="",
                              data::DFSet=DATASET,
                              data_meta::DataFrame=DATASET_META,
                              maxsteps::Int64=MAXSTEPS,
                              N::Int64=1000,
                              ntrials::Int64=10)

  srand(seed)

  problem = ACASXClustering(runtype, data, clusterdataname, data_meta)
  T1, alpha, n_epochs = estimate_temp_params(problem, P1, n_epochs, Tfinal, maxsteps, N, ntrials)

  return T1, alpha, n_epochs
end

end #module
