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

module SA_Logs

export default_logs, default_console!, default_logs1

using Reexport
@reexport using RLESUtils: Observers, Loggers

function default_logs(par_observer::Observer)
  logs = TaggedDFLogger()
  add_folder!(logs, "parameters", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "computeinfo", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "result", [Float64, ASCIIString, Int64], ["fitness", "expr", "total_evals"])

  add_observer(par_observer, "parameters", push!_f(logs, "parameters"))
  add_observer(par_observer, "computeinfo", push!_f(logs, "computeinfo"))
  add_observer(par_observer, "result", push!_f(logs, "result"))

  return logs
end

function default_console!(observer::Observer, loginterval::Int64)
  add_observer(observer, "verbose1", x -> println(x...))
  add_observer(observer, "current_best", x -> begin
                 starti, iter, fitness, expr = x
                 if rem(iter, loginterval) == 0
                   println("start=$starti, i=$iter, fitness=$fitness, expr=$expr")
                 end
               end)
  add_observer(observer, "temperature", x -> begin
                 starti, iter, T = x
                 if rem(iter, loginterval) == 0
                   println("start=$starti, i=$iter, T=$T")
                 end
               end)
end

function default_logs1(observer::Observer, loginterval::Int64)
  logs = TaggedDFLogger()
  add_folder!(logs, "parameters", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "computeinfo", [ASCIIString, Any], ["parameter", "value"])
  add_folder!(logs, "result", [Float64, ASCIIString, Int64, Int64], ["fitness", "expr", "best_at_eval", "total_evals"])
  add_folder!(logs, "current_best", [Int64, Int64, Float64, ASCIIString], ["start", "iter", "fitness", "expr"])
  add_folder!(logs, "elapsed_cpu_s", [Int64, Int64, Float64], ["start", "iter", "elapsed_cpu_s"])

  add_observer(observer, "parameters", push!_f(logs, "parameters"))
  add_observer(observer, "computeinfo", push!_f(logs, "computeinfo"))
  add_observer(observer, "result", push!_f(logs, "result"))
  add_observer(observer, "current_best", x -> begin
                 iter = x[2]
                 if rem(iter, loginterval) == 0
                   push!(logs, "current_best", x)
                 end
               end)
  add_observer(observer, "elapsed_cpu_s", x -> begin
                 iter = x[2]
                 if rem(iter, loginterval) == 0
                   push!(logs, "elapsed_cpu_s", x)
                 end
               end)

  return logs
end

end #module

