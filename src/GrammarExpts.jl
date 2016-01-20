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

module GrammarExpts

export load_expt

using Reexport

global CONFIG
const EXPTDIR = dirname(@__FILE__)

#load experiments dynamically
#keeps the experiments separate, so that they don't clash at compile time
#esp the overloads
#pass keyword arguments as config into the loaded module
function load_expt(s::Symbol; kwargs...)
  global CONFIG = Dict{Symbol,Any}(kwargs)
  load_expt(Val{s})
end

function load_expt(::Type{Val{:acasx_mcts}})
  @eval include(joinpath(EXPTDIR, "acasx/mcts/acasx_mcts.jl"))
  @eval @reexport using .ACASX_MCTS
end

function load_expt(::Type{Val{:acasx_mcts2}})
  @eval include(joinpath(EXPTDIR, "acasx/mcts2/acasx_mcts2.jl"))
  @eval @reexport using .ACASX_MCTS2
end

function load_expt(::Type{Val{:acasx_ge}})
  @eval include(joinpath(EXPTDIR, "acasx/ge/acasx_ge.jl"))
  @eval @reexport using .ACASX_GE
end

function load_expt(::Type{Val{:acasx_ge_tree}})
  @eval include(joinpath(EXPTDIR, "acasx/ge_tree/acasx_ge_tree.jl"))
  @eval @reexport using .ACASX_GE_Tree
end

function load_expt(::Type{Val{:acasx_mcts_tree}})
  @eval include(joinpath(EXPTDIR, "acasx/mcts_tree/acasx_mcts_tree.jl"))
  @eval @reexport using .ACASX_MCTS_Tree
end

function load_expt(::Type{Val{:acasx_mcts2_tree}})
  @eval include(joinpath(EXPTDIR, "acasx/mcts2_tree/acasx_mcts2_tree.jl"))
  @eval @reexport using .ACASX_MCTS2_Tree
end

function load_expt(::Type{Val{:symbolic_mcts}})
  @eval include(joinpath(EXPTDIR, "symbolic/mcts/symbolic_mcts.jl"))
  @eval @reexport using .SYMBOLIC_MCTS
end

function load_expt(::Type{Val{:symbolic_ge}})
  @eval include(joinpath(EXPTDIR, "symbolic/ge/symbolic_ge.jl"))
  @eval @reexport using .SYMBOLIC_GE
end

function load_expt(::Type{Val{:ant_ge}})
  @eval include(joinpath(EXPTDIR, "ant/ge/ant_ge.jl"))
  @eval @reexport using .ANT_GE
end

function load_expt(::Type{Val{:ant_mcts}})
  @eval include(joinpath(EXPTDIR, "ant/mcts/ant_mcts.jl"))
  @eval @reexport using .ANT_MCTS
end

load_expt{T}(::Type{Val{T}}) = error("experiment not defined")

end # module
