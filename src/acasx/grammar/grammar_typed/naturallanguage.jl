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

const U_LTE = "<=" #"\u2264"
const U_GTE = ">=" #"\u2265"
const U_IMPLIES = "=>" #"\u21D2"

function get_format_pretty{T<:AbstractString}(colnames::Vector{T})
  fmt = Format()
  for s in ["&&", "||", "&", "|"]
    fmt[s] = bin_infix
  end
  fmt[".=="] = (cmd, args) -> bin_infix("=", args)
  fmt[".<"] = (cmd, args) -> bin_infix("<", args)
  fmt[".<="] = (cmd, args) -> bin_infix("$(U_LTE)", args)
  fmt["D"] = (cmd, args) -> "$(colnames[parse(Int,args[2])])"
  fmt["sn"] = (cmd, args) -> "sign($(args[1])) = sign($(args[2]))"
  fmt["dfeq"] = (cmd, args) -> "$(args[1]) - $(args[2]) = $(args[3])"
  fmt["dflt"] = (cmd, args) -> "$(args[1]) - $(args[2]) < $(args[3])"
  fmt["dfle"] = (cmd, args) -> "$(args[1]) - $(args[2]) $(U_LTE) $(args[3])"
  fmt["ctlt"] = (cmd, args) -> "count($(args[1])) < $(args[2])"
  fmt["ctle"] = (cmd, args) -> "count($(args[1])) $(U_LTE) $(args[2])"
  fmt["ctgt"] = (cmd, args) -> "count($(args[1])) > $(args[2])"
  fmt["ctge"] = (cmd, args) -> "count($(args[1])) $(U_GTE) $(args[2])"
  fmt["cteq"] = (cmd, args) -> "count($(args[1])) = $(args[2])"
  fmt["Y"] = (cmd, args) -> "$(args[1]) $(U_IMPLIES) $(args[2])"

  return fmt
end

function get_format_natural{T<:AbstractString}(colnames::Vector{T})
  fmt = Format()
  fmt["&&"] = (cmd, args) -> bin_infix("and", args)
  fmt["||"] = (cmd, args) -> bin_infix("or", args)
  fmt["&"] = (cmd, args) -> bin_infix("and", args)
  fmt["|"] = (cmd, args) -> bin_infix("or", args)
  fmt[".=="] = (cmd, args) -> bin_infix("equals", args)
  fmt[".<"] = (cmd, args) -> bin_infix("is less than", args)
  fmt[".<="] = (cmd, args) -> bin_infix("is less than or equal to", args)
  fmt["D"] = (cmd, args) -> "$(colnames[parse(Int, args[2])])"
  fmt["sn"] = (cmd, args) -> "the sign of $(args[1]) is equal to the sign of $(args[2])"
  fmt["dfeq"] = (cmd, args) -> "the difference between $(args[1]) and $(args[2]) is equal to $(args[3])"
  fmt["dflt"] = (cmd, args) -> "the difference between $(args[1]) and $(args[2]) is less than $(args[3])"
  fmt["dfle"] = (cmd, args) -> "the difference between $(args[1]) and $(args[2]) is less than or equal to $(args[3])"
  fmt["ctlt"] = (cmd, args) -> "the number of times $(args[1]) is less than $(args[2])"
  fmt["ctle"] = (cmd, args) -> "the number of times $(args[1]) is less than or equal to $(args[2])"
  fmt["ctgt"] = (cmd, args) -> "the number of times $(args[1]) is greater than $(args[2])"
  fmt["ctge"] = (cmd, args) -> "the number of times $(args[1]) is greater than or equal to $(args[2])"
  fmt["cteq"] = (cmd, args) -> "the number of times $(args[1]) is equal to $(args[2])"
  fmt["Y"] = (cmd, args) -> "$(args[1]) implies that $(args[2])"
  fmt["G"] = (cmd, args) -> "for all time, $(args[1])"
  fmt["F"] = (cmd, args) -> "at some point, $(args[1])"
  fmt["!"] = (cmd, args) -> "it is not true that $(args[1])"
  return fmt
end

