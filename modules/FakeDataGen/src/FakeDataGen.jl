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
Generate some fake data into datasets for grammar experiments
"""
module FakeDataGen

export generate_fake_data

using DataFrames

const DATAPATH = Pkg.dir("Datasets/data")
const BIN_SYNTH_NAME = "bin_synth"

function generate_fake_data(dataset::AbstractString, args...)
    generate_fake_data(Val{symbol(dataset)}, args...)
end

"""
bin_synth dataset
"""
function generate_fake_data(::Type{Val{symbol(BIN_SYNTH_NAME)}}, n_feats::Int64=20, 
    n_samples::Int64=5000)
    mkpath(joinpath(DATAPATH, BIN_SYNTH_NAME))
    D = convert(Array{Int64}, rand(Bool, n_samples, n_feats))
    colnames = [symbol("x$i") for i = 1:n_feats]
    feats = DataFrame(D)
    names!(feats, colnames)
    filename = joinpath(DATAPATH, BIN_SYNTH_NAME, "feats.csv.gz") 
    writetable(filename, feats)

    #labels
    labels = DataFrame()
    labels[:x1] = feats[:x1] #x1
    labels[:x2] = feats[:x2] #x2
    labels[:x1_and_x3] = feats[:x1] & feats[:x3] #x1 and x3
    labels[:x2_or_x4] = feats[:x2] | feats[:x4] #x2 or x4
    labels[:x1_and_x4] = feats[:x1] & feats[:x4] #x1 and x4
    labels[:x2_or_x3] = feats[:x2] | feats[:x3] #x2 or x3
    filename = joinpath(DATAPATH, BIN_SYNTH_NAME, "labels.csv.gz") 
    writetable(filename, labels)
end

end
