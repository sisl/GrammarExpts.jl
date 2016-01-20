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

using TreeToJSON
using TikzQTrees
using RLESUtils.LatexUtils

function decisiontreevis{T}(dtree::DecisionTree, Dl::DFSetLabeled{T}, fileroot::AbstractString)
  get_depth(tree::DecisionTree) = get_depth(tree.root)
  get_depth(node::DTNode) = node.depth
  get_children(tree::DecisionTree) = get_children(tree.root)
  get_children(node::DTNode) = node.children

  get_name(tree::DecisionTree) = get_name(tree.root)
  function get_name(node::DTNode)
    members = sort(Dl.names[node.members], by=x->parse(Int64, x))
    members_text = if length(members) <= LIMIT_MEMBERS
      "members=" * join(members, ",")
    else
      "members=" * join(members[1:LIMIT_MEMBERS], ",") * ", and $(length(members)-LIMIT_MEMBERS) more."
    end
    label = "label=$(node.label)"
    confidence = "confidence=" * string(signif(node.confidence, 3))
    if node.split_rule != nothing
      tree = get_tree(node.split_rule)
      expr = string(get_expr(tree))
      pretty = pretty_string(tree, FMT_PRETTY)
      natural = pretty_string(tree, FMT_NATURAL, true)
      score = "fitness (lower is better)=" * string(signif(get_metric(node.split_rule), 4))
    else
      expr = pretty = natural = "none"
      score = "fitness (lower is better)=none"
    end
    text = join([members_text, label, confidence, expr, pretty, natural, score], "\\\\")
    return text
  end

  viscalls = VisCalls(get_name, get_children, get_depth)
  write_json(dtree, viscalls, "$(fileroot)_decisiontree.json")
  plottree("$(fileroot)_decisiontree.json", outfileroot="$(fileroot)_decisiontree")
end
