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

import Compat.ASCIIString
using RLESUtils, DataFrameSets
using DataFrames
using Datasets

const DATAPATH = Pkg.dir("Datasets/data")
const BIN_SYNTH_NAME = "bin_synth"
const BIN_TS_SYNTH_NAME = "bin_ts_synth"
const VHDIST_NAME = "vhdist"
const VDIST_NAME = "vdist"

function generate_fake_data(dataset::AbstractString; kwargs...)
    generate_fake_data(Val{Symbol(dataset)}; kwargs...)
end

"""
bin_synth dataset
synthetic data with
static binary features
"""
function generate_fake_data(::Type{Val{Symbol(BIN_SYNTH_NAME)}}; 
    n_feats::Int64=20, 
    n_samples::Int64=5000)
    
    mkpath(joinpath(DATAPATH, BIN_SYNTH_NAME))
    D = convert(Array{Int64}, rand(Bool, n_samples, n_feats))
    colnames = [Symbol("x$i") for i = 1:n_feats]
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

"""
bin_ts_synth dataset
synthetic data with
binary time series features
"""
function generate_fake_data(::Type{Val{Symbol(BIN_TS_SYNTH_NAME)}}; 
    n_feats::Int64=20, 
    n_time::Int64=50,
    n_samples::Int64=5000)

    outdir = joinpath(DATAPATH, BIN_TS_SYNTH_NAME)
    mkpath(outdir)

    records = DataFrame[]
    colnames = [Symbol("x$i") for i = 1:n_feats]
    for j = 1:n_samples
        B = rand(Bool, n_time, n_feats)
        for i = 1:size(B, 2)
            r = rand(1:3)
            if r == 1
                B[:,i] = false
            elseif r == 2
                B[:,i] = true
            end
        end
        A = convert(Array{Int64}, B)
        D = DataFrame(A)
        names!(D, colnames)
        push!(records, D)
    end
    Ds = DFSet(records)
    meta = getmeta(Ds)

    #labels, saved into meta df
    meta[:F_x1] = [reduce(|, r[:x1]) for r in records] #F(x1)
    meta[:G_x1] = [reduce(&, r[:x1]) for r in records] #G(x1)
    meta[:F_x2] = [reduce(|, r[:x2]) for r in records] #F(x2)
    meta[:G_x2] = [reduce(&, r[:x2]) for r in records] #G(x2)
    meta[:F_x3] = [reduce(|, r[:x3]) for r in records] #F(x3)
    meta[:G_x3] = [reduce(&, r[:x3]) for r in records] #G(x3)
    meta[:F_x4] = [reduce(|, r[:x4]) for r in records] #F(x4)
    meta[:G_x4] = [reduce(&, r[:x4]) for r in records] #G(x4)
    meta[:F_x5] = [reduce(|, r[:x5]) for r in records] #F(x5)
    meta[:G_x5] = [reduce(&, r[:x5]) for r in records] #G(x5)

    #F(x1 & x3)
    meta[:F_x1_and_x3] = [reduce(|, r[:x1] & r[:x3]) for r in records] 
    #F(x1 | x3)
    meta[:F_x1_or_x3] = [reduce(|, r[:x1] | r[:x3]) for r in records] 
    #G(x1 & x3)
    meta[:G_x1_and_x3] = [reduce(&, r[:x1] & r[:x3]) for r in records] 
    #G(x1 | x3)
    meta[:G_x1_or_x3] = [reduce(&, r[:x1] | r[:x3]) for r in records] 
    #F(x2 & x4)
    meta[:F_x2_and_x4] = [reduce(|, r[:x2] & r[:x4]) for r in records] 
    #F(x2 | x4)
    meta[:F_x2_or_x4] = [reduce(|, r[:x2] | r[:x4]) for r in records] 
    #G(x2 & x4)
    meta[:G_x2_and_x4] = [reduce(&, r[:x2] & r[:x4]) for r in records] 
    #G(x2 | x4)
    meta[:G_x2_or_x4] = [reduce(&, r[:x2] | r[:x4]) for r in records] 
    #F(x1 & x4)
    meta[:F_x1_and_x4] = [reduce(|, r[:x1] & r[:x4]) for r in records] 
    #F(x1 | x4)
    meta[:F_x1_or_x4] = [reduce(|, r[:x1] | r[:x4]) for r in records] 
    #G(x1 & x4)
    meta[:G_x1_and_x4] = [reduce(&, r[:x1] & r[:x4]) for r in records] 
    #G(x1 | x4)
    meta[:G_x1_or_x4] = [reduce(&, r[:x1] | r[:x4]) for r in records] 

    save_csvs(outdir, Ds) 
    Ds
end

"""
vertical and horizontal miss distances from ACASX dataset
real-valued time series features
actually not fake data...
"""
function generate_fake_data(::Type{Val{Symbol(VHDIST_NAME)}}; 
    cas_data_name::ASCIIString="dasc", 
    vdist_name::Symbol=:abs_alt_diff,
    hdist_name::Symbol=:horizontal_range,
    nmac_name::Symbol=:nmac,
    outname::ASCIIString=VHDIST_NAME
    )

    outdir = joinpath(DATAPATH, outname)
    mkpath(outdir)

    cas_data = dataset(cas_data_name)

    records = DataFrame[]
    colnames = [:vdist, :hdist]
    for r in getrecords(cas_data)
        D = r[[vdist_name, hdist_name]] 
        names!(D, colnames)
        push!(records, D)
    end

    Ds = DFSet(getmeta(cas_data), records)
    save_csvs(outdir, Ds)
end

"""
vertical distances from ACASX dataset
real-valued time series features
actually not fake data...
"""
function generate_fake_data(::Type{Val{Symbol(VDIST_NAME)}}; 
    cas_data_name::ASCIIString="dasc", 
    vdist_name::Symbol=:abs_alt_diff,
    nmac_name::Symbol=:nmac,
    nmac_thresh::Float64=100.0,
    outname::ASCIIString=VDIST_NAME
    )

    outdir = joinpath(DATAPATH, outname)
    mkpath(outdir)

    cas_data = dataset(cas_data_name)
    meta = metadf(length(cas_data))

    records = DataFrame[]
    nmac = Bool[]
    for r in getrecords(cas_data)
        D = DataFrame()
        D[:vdist] = r[vdist_name] 
        push!(records, D)
        push!(nmac, any(D[:vdist] .< nmac_thresh))
    end

    meta[nmac_name] = nmac
    Ds = DFSet(meta, records)
    save_csvs(outdir, Ds)
end

end #module
