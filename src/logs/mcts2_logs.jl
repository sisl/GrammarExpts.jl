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

@reexport using RLESUtils: Observers, Loggers

function default_logs(observer::Observer)
  logs = TaggedDFLogger()
  add_folder!(logs, "parameters", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "computeinfo", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "cputime", [Int64, Float64], ["step", "cputime_s"])
  add_folder!(logs, "result", [Float64, ASCIIString, Int64, Int64], ["total_reward", "expr", "best_at_eval", "total_evals"])
  add_folder!(logs, "expression", [ASCIIString, ASCIIString, ASCIIString], ["raw", "pretty", "natural"])
  add_folder!(logs, "current_best", [Int64, Float64, ASCIIString, ASCIIString], ["iteration", "reward", "state", "expr"])

  add_observer(observer, "parameters", push!_f(logs, "parameters"))
  add_observer(observer, "computeinfo", push!_f(logs, "computeinfo"))
  add_observer(observer, "cputime", push!_f(logs, "cputime"))
  add_observer(observer, "result", push!_f(logs, "result"))
  add_observer(observer, "expression", push!_f(logs, "expression"))
  add_observer(observer, "current_best", x -> begin
                 i, reward, state = x
                 if rem(i, LOGINTERVAL) == 0
                   push!(logs, "current_best", [i, reward, string(state.past_actions), string(get_expr(state))])
                 end
               end)

  return logs
end

function default_console!(observer::Observer)
  #console out
  add_observer(observer, "verbose1", x -> println(x[1]))
  add_observer(observer, "action", x -> println("step=$(x[1]), action=$(x[2])"))
  add_observer(observer, "result", x -> println("total_reward=$(x[1]), expr=$(x[2]), best_at_eval=$(x[3]), total_evals=$(x[4])"))
  add_observer(observer, "current_best", x -> begin
                 i, reward, state = x
                 if rem(i, 100) == 0
                   println("step $i: best_reward=$reward, best_state=$(state.past_actions)")
                 end
               end)
end
