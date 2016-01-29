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

using Gadfly
using DataFrames
using RLESUtils.Loggers

function param_reward_avg(sweeplog::AbstractString)
  D = readtable(sweeplog)
  D1 = D[:,[:seed,:n_iters,:exploration_const,:best_reward]]
  D2 = aggregate(D1,[:n_iters, :exploration_const], [mean, std])

  ymins = D2[:, :best_reward_mean] - D2[:, :best_reward_std] ./ sqrt(2 * (D2[:, :seed_mean] - 1))
  ymaxs = D2[:, :best_reward_mean] + D2[:, :best_reward_std] ./ sqrt(2 * (D2[:, :seed_mean] - 1))
  p = plot(D2, x="exploration_const", y="best_reward_mean", ymin=ymins, ymax=ymaxs,
           Geom.point, Geom.line, Geom.errorbar)
  draw(PDF("ec_reward_avg.pdf", 4inch, 3inch), p)
end

function nevals_reward_avg(funclogfile::AbstractString)
  D = readtable(funclogfile)
  D1 = D[:,[:seed, :exploration_const, :n_evals, :best_reward]]
  D2 = aggregate(D1,[:exploration_const, :n_evals], [mean, std])

  ymins = D2[:, :best_reward_mean] - D2[:, :best_reward_std] ./ sqrt(2 * (D2[:, :seed_mean] - 1))
  ymaxs = D2[:, :best_reward_mean] + D2[:, :best_reward_std] ./ sqrt(2 * (D2[:, :seed_mean] - 1))
  p = plot(D2, x="n_evals", y="best_reward_mean", ymin=ymins, ymax=ymaxs, color="exploration_const",
           Geom.point, Geom.line, Geom.errorbar)
  draw(PDF("nevals_reward_avg.pdf", 11inch, 8.5inch), p)
end
