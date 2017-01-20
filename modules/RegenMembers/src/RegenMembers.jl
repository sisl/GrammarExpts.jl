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
Fixes problem where members of leaf nodes are not included in members log.
This tool is used to update old results to new format using the tree
saved in save.jld
"""
module RegenMembers

export regen_members, regen_members_recursive, DecisionTrees

using Compat
import Compat.ASCIIString

using GrammarExpts
using RLESUtils, TreeIterators
import TreeIterators.get_children
using JLD, DataFrames

#################
#recreate for loading JLD, real got updated and breaks JLD
module DecisionTrees

export DTNode, DecisionTree

type DTNode{T1,T2}
    depth::Int64
    members::Vector{Int64} #indices into data starting at 1
    split_rule::Any #object used in callback for split rule
    children::Dict{T1,DTNode{T1,T2}} #key=split_rule predicts, value=child node, T1=predict label type
    label::T2 #T2=label type
    confidence::Float64
end
type DecisionTree{T1,T2}
    root::DTNode{T1,T2}
end
end #module
######################

using .DecisionTrees

get_children(node::DTNode) = collect(values(node.children))

function regen_members(jldfile::AbstractString, memberfile::AbstractString)
    dtree = load(jldfile, "dtree") 
    D = DataFrame([Int64,ASCIIString], [:decision_id,:members], 0) 
    node_id = 1
    for node in tree_iter(dtree.root)
        push!(D, [node_id, string(node.members)])
        node_id += 1
    end
    writetable(memberfile, D)
    D
end

function regen_members_recursive(topdir::AbstractString, jldfile::AbstractString,
    memberfile::AbstractString)
    for (root, dirs, files) in walkdir(topdir)
        fin = joinpath(root, jldfile) 
        if isfile(fin)
            fout = joinpath(root, memberfile)
            regen_members(fin, fout)
        end
    end
end

end #module
