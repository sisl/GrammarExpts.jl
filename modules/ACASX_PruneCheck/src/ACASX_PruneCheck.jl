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

module ACASX_PruneCheck

export setup, runtest, acasx_prunecheck

using ACASXProblem, DerivationTrees

function setup(;data::AbstractString="libcas098small",
               data_meta::AbstractString="libcas098small_meta",
               maxsteps::Int64=20)

  problem = ACASXClustering(data, data_meta)
  grammar = create_grammar(problem)
  tree_params = DerivTreeParams(grammar, maxsteps)
  tree = DerivationTree(tree_params)

  return problem, tree
end

"""
Draw random samples and compare get_fitness() on full and pruned.  Produce counterexample if they differ.
"""
function runtest{T}(problem::ACASXClustering{T}, tree::DerivationTree,
                    n_samples::Int64=50000, earlystop_div::Int64=10)
  bestfit = realmax(Float64)
  for i = 1:n_samples
    rand!(tree)
    expr = get_expr(tree)

    #side-by-side comparison
    fullfit = get_fitness(problem, expr)
    prunefit = get_fitness(problem, expr, bestfit, realmax(Float64), earlystop_div)

    println("iter=$i of $(n_samples), bestfit=$(signif(bestfit, 4)), fullfit=$(signif(fullfit, 4)), prunefit=$(signif(prunefit, 4)), expr=$(string(expr))")
    if fullfit < bestfit
      bestfit = fullfit

      #on updates, these should be the same. If not, return counterexample
      if prunefit != fullfit
        return tree, expr, fullfit, prunefit
      end
    end
  end
  return tree, nothing, -1.0, -1.0
end

function acasx_prunecheck(; seed=1)
  srand(seed)
  problem, tree = setup()
  tree, expr, fullfit, prunefit = runtest(problem, tree, 50000, 10)
  @show fullfit
  @show prunefit
  @show expr
  return tree, expr, fullfit, prunefit
end

end #module
