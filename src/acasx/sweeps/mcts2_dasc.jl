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

const EXPT = :acasx_mcts2
const DATA = :dasc
const CONFIG = :normal
const VIS = true
const MCTSTREEVIS = false
const LOGINTERVAL = 500

const OUTDIR = Pkg.dir("GrammarExpts/results/acasxmcts2_dasc")
const LOGFILEROOT = "acasxmcts2_dasc"

using GrammarExpts
load_expt(EXPT, data=DATA, config=CONFIG, vis=VIS)

include("../../sweeps/mcts2_sweep.jl")

f = caller_f(acasx_mcts2, OUTDIR, LOGFILEROOT, observer, funclogger)
script = ParamSweep(f)

push!(script, 1:5) #seed
push!(script, [20000]) #n_iters
push!(script, [0.0]) #[500.0, 1500.0, 2000.0]) #ec

textfile(joinpath(OUTDIR, "description.txt"), expt=EXPT, data=DATA, config=CONFIG, vis=VIS,
         outdir=OUTDIR, logfileroot=LOGFILEROOT, script=dump2string(script))

run(script)

#save logs
save_log(joinpath(OUTDIR, "$(LOGFILEROOT)_log"), logger)
save_log(joinpath(OUTDIR, "$(LOGFILEROOT)_funclog"), funclogger)

#plot
include("../../plots/mcts2_sweep_plot.jl")
param_reward_avg(joinpath(OUTDIR, "$(LOGFILEROOT)_log.csv.gz"))
nevals_reward_avg(joinpath(OUTDIR, "$(LOGFILEROOT)_funclog.csv.gz"))
