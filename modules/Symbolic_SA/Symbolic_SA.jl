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

module Symbolic_SA

export configure, symbolic_sa, symbolic_temp_params

using ExprSearch.SA
using Reexport

using GrammarExpts
using SymbolicProblem, Configure
import Configure.configure

const CONFIGDIR = joinpath(dirname(@__FILE__), "config")

configure(::Type{Val{:Symbolic_SA}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

function symbolic_sa(outdir::AbstractString="./"; seed=1,
                     logfileroot::AbstractString="symbolic_sa_log",

                     gt_file::AbstractString="gt_easy.jl",
                     maxsteps::Int64=25,
                     T1::Float64=50.0,
                     alpha::Float64=0.8,
                     n_epochs::Int64=100,
                     n_starts::Int64=1,
                     n_threads::Int64=1,

                     loginterval::Int64=100,
                     observer::Observer=Observer())

  srand(seed)

  problem = Symbolic(gt_file)

  add_observer(observer, "temperature", x -> begin
                 if rem(x[2], loginterval) == 0
                   println("start=$(x[1]), i=$(x[2]), T=$(x[3])")
                 end
               end)
  add_observer(observer, "current_best", x -> begin
                 if rem(x[2], loginterval) == 0
                   println("start=$(x[1]), i=$(x[2]), fitness=$(x[3]), expr=$(x[4])")
                 end
               end)

  sa_params = SAESParams(maxsteps, T1, alpha, n_epochs, n_starts, observer)
  psa_params = PSAESParams(n_threads, sa_params)

  result = exprsearch(psa_params, problem)

  return result
end


function symbolic_temp_params(P1::Float64=0.8; seed=1,
                     gt_file::AbstractString=GT_FILE,
                     n_epochs::Int64=N_EPOCHS,
                     Tfinal::Float64=0.1,
                     maxsteps::Int64=MAXSTEPS,
                     N::Int64=500,
                     ntrials::Int64=10)

  srand(seed)

  problem = Symbolic(gt_file)
  T1, alpha, n_epochs = estimate_temp_params(problem, P1, n_epochs, Tfinal, maxsteps, N, ntrials)

  return T1, alpha, n_epochs
end

end #module
