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

### These tree modules need a refactor
"""
Create a decision tree to recursively split encounters in the ACASX Problem. GE algorithm.
Example usage: config=configure(ACASX_GE_Tree,"normal","nvn_dasc"); acasx_ge_tree(;config...)
"""
module ACASX_GE_Tree

export configure, acasx_ge_tree

import Compat.ASCIIString
using DecisionTrees
using ExprSearch.GE
using Datasets
using RLESUtils, Obj2Dict, Configure, Confusion, TreeIterators, LogSystems
using Reexport
using JLD

using GrammarExpts
using ACASXProblem
using DerivTreeVis, DecisionTreeVis, GE_Tree_Logs
import Configure.configure

include("dtree_callbacks.jl")

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")
const T1 = Bool #predict_type
const T2 = Int64 #label_type


configure(::Type{Val{:ACASX_GE_Tree}}, configs::AbstractString...) = configure_path(CONFIGDIR, configs...)

function train_dtree{T}(ge_params::GEESParams, 
    problem::ACASXClustering, 
    Dl::DFSetLabeled{T},
    maxdepth::Int64)

    logs = default_logs()
    add_folder!(logs, "members", [ASCIIString, ASCIIString, Int64], 
        ["members_true", "members_false", "decision_id"])

    num_data = length(Dl)
    
    p = DTParams(num_data, maxdepth, T1, T2)

    dtree = build_tree(p,
        Dl, problem, ge_params, logs) #userargs... #FIXME: don't do userargs this way...

  return dtree, logs
end

"""
Example call:
config=configure(ACASX_GE_Tree, "nvn_dasc", "normal")
acasx_ge_tree(; config...)
"""
function acasx_ge_tree(;outdir::AbstractString=joinpath(RESULTDIR, "./ACASX_GE_Tree"),
                        seed=1,
                        logfileroot::AbstractString="acasx_ge_tree_log",

                        #dataset
                        runtype::Symbol=:nmacs_vs_nonnmacs,
                        data::AbstractString="dasc",
                        manuals::AbstractString="dasc_manual",
                        clusterdataname::AbstractString="josh1",

                        #decision tree
                        maxdepth::Int64=3,

                        #GE
                        genome_size::Int64=20,
                        pop_size::Int64=50,
                        maxwraps::Int64=0,
                        top_keep::Float64=0.25,
                        top_seed::Float64=0.5,
                        rand_frac::Float64=0.25,
                        prob_mutation::Float64=0.2,
                        mutation_rate::Float64=0.2,
                        defaultcode::Union{Symbol,Expr}=:(eval(false)),
                        maxiterations::Int64=3,
        
                        #GE vis
                        hist_nbins::Int64=40,
                        hist_edges::Range{Float64}=linspace(0.0, 200.0, hist_nbins + 1),
                        hist_mids::Vector{Float64}=collect(Base.midpoints(hist_edges)),
                        
                        #DT vis
                        vis::Bool=true,
                        plotpdf::Bool=true,
                        limit_members::Int64=10,

                        #save tree
                        b_jldsave::Bool=true
                        )

  srand(seed)
  mkpath(outdir)

  problem = ACASXClustering(runtype, data, manuals, clusterdataname)

  logsys = GE.logsystem()
  observer = get_observer(logsys)
  ge_params = GEESParams(genome_size, pop_size, maxwraps,
                         top_keep, top_seed, rand_frac, prob_mutation, mutation_rate, defaultcode,
                         maxiterations, logsys)

  Dl = problem.Dl
  dtree, logs = train_dtree(ge_params, problem, Dl, maxdepth)

  ##################################
  #add many items to log
  push!(logs, "parameters", ["seed", seed, 0])
  push!(logs, "parameters", ["runtype", runtype, 0])
  push!(logs, "parameters", ["data", data, 0])

  #classifier performance
  members = DecisionTrees.get_members(dtree)
  p = DTParams(length(members), maxdepth, T1, T2)
  pred = map(x -> classify(p, dtree, x, Dl, problem), members)
  pred = pred .== 1 
  truth = get_truth(members, Dl)
  truth = truth .== 1
  conf_mat = ConfusionMat(pred, truth)
  add_varlist!(logs, "classifier_metrics")
  push!(logs, "classifier_metrics", ["truepos", conf_mat.truepos]) 
  push!(logs, "classifier_metrics", ["trueneg", conf_mat.trueneg])
  push!(logs, "classifier_metrics", ["falsepos", conf_mat.falsepos])
  push!(logs, "classifier_metrics", ["falseneg", conf_mat.falseneg])
  push!(logs, "classifier_metrics", ["precision", precision(conf_mat)])
  push!(logs, "classifier_metrics", ["recall", recall(conf_mat)])
  push!(logs, "classifier_metrics", ["accuracy", accuracy(conf_mat)])
  push!(logs, "classifier_metrics", ["f1_score", f1_score(conf_mat)])

  #interpretability metrics
  add_varlist!(logs, "interpretability_metrics")
  num_rules = nrow(logs["result"])
  rules = logs["result"][:expr]
  avg_rule_length = mean(map(length, rules))
  TreeIterators.get_children(node::DTNode) = collect(values(node.children))
  nodes = collect(tree_iter(dtree.root))
  push!(logs, "interpretability_metrics", ["num_rules", num_rules]) 
  push!(logs, "interpretability_metrics", ["avg_rule_length", avg_rule_length]) 
  push!(logs, "interpretability_metrics", ["num_nodes", length(nodes)]) 
  push!(logs, "interpretability_metrics", ["num_leaf", count(DecisionTrees.isleaf, nodes)]) 

  add_folder!(logs, "rule_metrics", [ASCIIString, Int64, Int64], ["expr", "deriv_tree_num_nodes", "deriv_tree_num_leafs" ])
  TreeIterators.get_children(node::DerivTreeNode) = node.children
  for node in nodes
    if node.split_rule != nothing
        root = get_derivtree(node.split_rule.tree).root
        derivnodes = collect(tree_iter(root))
        push!(logs, "rule_metrics", [string(node.split_rule.expr), length(derivnodes), count(DerivationTrees.isleaf, derivnodes)])
    end
  end
  avg_deriv_num_nodes = mean(logs["rule_metrics"][:deriv_tree_num_nodes])
  avg_deriv_num_leafs = mean(logs["rule_metrics"][:deriv_tree_num_leafs])
  push!(logs, "interpretability_metrics", ["avg_deriv_tree_num_nodes", avg_deriv_num_nodes])
  push!(logs, "interpretability_metrics", ["avg_deriv_tree_num_leafs", avg_deriv_num_leafs])

  #save logs
  outfile = joinpath(outdir, "$(logfileroot).txt")
  save_log(outfile, logs)
  ##################################

  #visualize
  if vis
    decisiontreevis(dtree, Dl, joinpath(outdir, "$(logfileroot)_vis"), limit_members,
                    FMT_PRETTY, FMT_NATURAL; plotpdf=plotpdf)
  end

  if b_jldsave
      jldfile = joinpath(outdir, "save.jld")
      save(jldfile, "dtree", dtree)
  end


  return dtree, logs
end

end #module
