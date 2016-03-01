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
Default logs for SA_Tree
"""
module SA_Tree_Logs

export default_logs, set_observers!

using DataFrames #nrow
using ExprSearch, DerivationTrees #get_expr
using Reexport
using RLESUtils
@reexport using Observers, Loggers

function default_logs()
  logs = TaggedDFLogger()

  add_folder!(logs, "parameters", [ASCIIString, Any, Int64], ["parameter", "value", "decision_id"])
  add_folder!(logs, "computeinfo", [ASCIIString, Any, Int64], ["parameter", "value", "decision_id"])
  add_folder!(logs, "result", [Float64, ASCIIString, Int64, Int64], ["fitness", "expr", "total_evals", "decision_id"])
  add_folder!(logs, "expression", [ASCIIString, ASCIIString, ASCIIString, Int64], ["raw", "pretty", "natural", "decision_id"])

  return logs
end

function set_observers!(par_observer::Observer, observer::Observer, logs::TaggedDFLogger, loginterval::Int64)
  empty!(par_observer)
  empty!(observer)
  ####################
  #print out observers

  add_observer(observer, "verbose1", x -> println(x...))
  add_observer(observer, "current_best", x -> begin
                 starti, iter, fitness, expr = x
                 if rem(iter, loginterval) == 0
                   println("start=$starti, i=$iter, fitness=$fitness, expr=$expr")
                 end
               end)

  ###################
  #log observers
  decision_id = nrow(logs["result"]) > 0 ?
    maximum(logs["result"][:decision_id]) + 1 : 1
  add_observer(par_observer, "parameters", append_push!_f(logs, "parameters", decision_id))
  add_observer(par_observer, "computeinfo", append_push!_f(logs, "computeinfo", decision_id))
  add_observer(par_observer, "result", append_push!_f(logs, "result", decision_id))
  add_observer(par_observer, "expression", append_push!_f(logs, "expression", decision_id))
end



end #module
