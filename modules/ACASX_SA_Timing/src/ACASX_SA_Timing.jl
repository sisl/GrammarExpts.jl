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
Profiling time and memory for some of the functions in ACASXProblem and SA.
"""
module ACASX_SA_Timing

export acasx_sa_timing

using ExprSearch.SA
using DerivationTrees
using RLESUtils: SwapBuffers, Observers, Loggers, StringUtils

using GrammarExpts
using ACASXProblem

function acasx_sa_timing(problem=nothing, outdir::AbstractString="./"; seed=1,
                  logfileroot::AbstractString="acasx_sa_log",

                  runtype::Symbol=:nmacs_vs_nonnmacs,
                  data::AbstractString="dasc",
                  data_meta::AbstractString="dasc_meta",
                  manuals::AbstractString="dasc_manual",
                  clusterdataname::AbstractString="josh1",

                  maxsteps::Int64=20,
                  T1::Float64=12.184,
                  alpha::Float64=0.99976,
                  n_epochs::Int64=50,
                  n_starts::Int64=1,

                  loginterval::Int64=100,
                  vis::Bool=true,
                  observer::Observer=Observer())

  if problem == nothing
    problem = ACASXClustering(runtype, data, data_meta, manuals, clusterdataname)
  end
  p = SAESParams(maxsteps, T1, alpha, n_epochs, n_starts, observer)

  grammar = create_grammar(problem)
  tree_params = DerivTreeParams(grammar, p.maxsteps)
  s_buffer = SwapBuffer(SA.SAState(DerivationTree(tree_params)),
                        SA.SAState(DerivationTree(tree_params)))
  result = SAESResult()

  j = 1
  @printeval @time s = active(s_buffer)
  @printeval @time SA.initialize!(s, problem, result) #initialize randomly

  T = p.T1
  i = 1
  @printeval @time s = active(s_buffer)
  @printeval @time sp = inactive(s_buffer)
  @printeval @time SA.neighbor!(sp, s, problem, result)

  @printeval @time iscomplete(sp.tree)
  @printeval @time SA.accept(s, sp, T)
  @printeval @time swap!(s_buffer)

  @printeval @time SA.initialize!(s, problem) #initialize randomly
  @printeval @time SA.update!(result, s)

  @printeval @time copy!(sp, s)
  @printeval @time SA.perturb!(sp.tree)

  @printeval @time rand!(s.tree, typemax(Int64))
  @printeval @time s.expr = get_expr(s.tree)
  @show s.expr
  @printeval @time s.fitness = get_fitness(problem, s.expr)
  @printeval @time f = to_function(problem, s.expr)
  @printeval @time predicts = map(f, problem.Dl.records)
  @printeval @time _, _, ent_post = ACASXProblem.entropy_metrics(predicts, problem.Dl.labels)

  @printeval @time predicts = Array(Bool, length(problem.Dl.records))
  @printeval @time begin
    records = problem.Dl.records
    for i = 1:length(records)
      predicts[i] = f(records[i])
    end
  end
  @printeval @time predicts[1] = f(records[1])

  v1 = rand(50)
  v2 = rand(50)
  @printeval @time ACASXProblem.abs_diff_lte(v1, v2, 0.1);
  @printeval @time ACASXProblem.sign_(v1,v2);

  members = rand(1:100, 30)
  @printeval @time Dl_sub = problem.Dl[members]

  problem
end

end #module
