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

module ACASXProblem

export ACASXClustering, create_grammar, get_fitness, to_function
export FMT_PRETTY, FMT_NATURAL

using Datasets
using ExprSearch
using Devectorize
import ExprSearch: ExprProblem, create_grammar, get_fitness

include("labeleddata.jl")
include("infogain.jl")
include("format.jl")

#fitness
const W_ENT = 100.0 #entropy
const W_LEN = 0.05 #

typealias RealVec Union{DataArray{Float64,1}, Vector{Float64}}

type ACASXClustering{T} <: ExprProblem
  Dl::DFSetLabeled{T}
  w_ent::Float64
  w_len::Float64
end

function ACASXClustering(runtype::Symbol, dataname::AbstractString, meta_name::AbstractString,
                         manuals::AbstractString, clustername::AbstractString,
                         w_ent::Float64=W_ENT, w_len::Float64=W_LEN)
  out = if runtype == :nmacs_vs_nonnmacs
    ACASXClustering(dataname, meta_name, w_ent, w_len)
  elseif runtype == :nmac_clusters
    ACASXClustering(dataname, manuals, clustername, w_ent, w_len)
  elseif runtype == :nonnmacs_extra_cluster
    ACASXClustering(dataname, meta_name, manuals, clustername, w_ent, w_len)
  else
    error("Runtype not defined ($runtype)")
  end
  return out
end

#nmacs vs non-nmacs
function ACASXClustering(dataname::AbstractString,
                         meta_name::AbstractString,
                         w_ent::Float64=W_ENT, w_len::Float64=W_LEN)

  data = dataset(dataname)
  data_meta = dataset(meta_name, "encounter_meta")

  Dl = nmacs_vs_nonnmacs(data, data_meta)
  return ACASXClustering(Dl, w_ent, w_len)
end

#clusterings only
function ACASXClustering(dataname::AbstractString,
                         manuals::AbstractString,
                         clustername::AbstractString,
                         w_ent::Float64=W_ENT, w_len::Float64=W_LEN)

  data = dataset(dataname)
  clustering = dataset(manuals, clustername)

  Dl =  nmac_clusters(clustering, data)
  return ACASXClustering(Dl, w_ent, w_len)
end

#clusterings with nonnmacs as extra cluster
function ACASXClustering(dataname::AbstractString,
                         meta_name::AbstractString,
                         manuals::AbstractString,
                         clustername::AbstractString,
                         w_ent::Float64=W_ENT, w_len::Float64=W_LEN)

  data = dataset(dataname)
  data_meta = dataset(meta_name, "encounter_meta")
  clustering = dataset(manuals, clustername)

  Dl =  nonnmacs_extra_cluster(clustering, data, data_meta)
  return ACASXClustering(Dl, w_ent, w_len)
end

function ExprSearch.create_grammar(problem::ACASXClustering)
  @grammar grammar begin
    start = bin

    bin = always | eventually | implies | count
    always = Expr(:call, :G, bin_vec) #global
    eventually = Expr(:call, :F, bin_vec) #future
    implies = Expr(:call, :Y, bin_vec, bin_vec)
    count = Expr(:call, :ctlt, bin_vec, timestep) | Expr(:call, :ctle, bin_vec, timestep) | Expr(:call, :ctgt, bin_vec, timestep) | Expr(:call, :ctge, bin_vec, timestep) | Expr(:call, :cteq, bin_vec, timestep)

    #produces a bin_vec
    bin_vec = bin_feat | and | or | not  | eq | lt | lte | abseq | abslt | abslte | diff_eq | diff_lt | diff_lte | sign | absdiff_eq | absdiff_lt | absdiff_lte
    and = Expr(:call, :&, bin_vec, bin_vec)
    or = Expr(:call, :|, bin_vec, bin_vec)
    not = Expr(:call, :!, bin_vec)

    #equal
    eq = vrate_eq | altdiff_eq | chi_angle_eq | psi_angle_eq | sr_eq | tds_eq | timer_eq | psid_eq | v_eq | alt_eq | abs_altdiff_eq
    vrate_eq = Expr(:comparison, vrate_feat, :.==, vrate_val) | Expr(:comparison, vrate_feat, :.==, vrate_feat)
    altdiff_eq = Expr(:comparison, altdiff_feat, :.==, altdiff_val)
    chi_angle_eq = Expr(:comparison, chi_angle_feat, :.==, angle_val) | Expr(:comparison, chi_angle_feat, :.==, chi_angle_feat)
    psi_angle_eq = Expr(:comparison, psi_angle_feat, :.==, psi_angle_feat)
    sr_eq = Expr(:comparison, sr_feat, :.==, sr_val)
    tds_eq = Expr(:comparison, tds_feat, :.==, tds_val) | Expr(:comparison, tds_feat, :.==, tds_feat)
    timer_eq = Expr(:comparison, timer_feat, :.==, timer_val) | Expr(:comparison, timer_feat, :.==, timer_feat)
    psid_eq = Expr(:comparison, psid_feat, :.==, psid_val) | Expr(:comparison, psid_feat, :.==, psid_feat)
    v_eq = Expr(:comparison, v_feat, :.==, v_val) | Expr(:comparison, v_feat, :.==, v_feat)
    alt_eq = Expr(:comparison, alt_feat, :.==, alt_val) | Expr(:comparison, alt_feat, :.==, alt_feat)
    abs_altdiff_eq = Expr(:comparison, abs_altdiff_feat, :.==, abs_altdiff_val)

    #absolute equals
    abseq = vrate_abseq | altdiff_abseq | chi_angle_abseq | psi_angle_abseq | sr_abseq | tds_abseq | timer_abseq | psid_abseq | v_abseq | alt_abseq
    vrate_abseq = Expr(:call, :abeq, vrate_feat, vrate_val) | Expr(:call, :abeq, vrate_feat, vrate_feat)
    altdiff_abseq = Expr(:call, :abeq, altdiff_feat, altdiff_val)
    chi_angle_abseq = Expr(:call, :abeq, chi_angle_feat, angle_val) | Expr(:call, :abeq, chi_angle_feat, chi_angle_feat)
    psi_angle_abseq = Expr(:call, :abeq, psi_angle_feat, psi_angle_feat)
    sr_abseq = Expr(:call, :abeq, sr_feat, sr_val)
    tds_abseq = Expr(:call, :abeq, tds_feat, tds_val) | Expr(:call, :abeq, tds_feat, tds_feat)
    timer_abseq = Expr(:call, :abeq, timer_feat, timer_val) | Expr(:call, :abeq, timer_feat, timer_feat)
    psid_abseq = Expr(:call, :abeq, psid_feat, psid_val) | Expr(:call, :abeq, psid_feat, psid_feat)
    v_abseq = Expr(:call, :abeq, v_feat, v_val) | Expr(:call, :abeq, v_feat, v_feat)
    alt_abseq = Expr(:call, :abeq, alt_feat, alt_val) | Expr(:call, :abeq, alt_feat, alt_feat)

    #less than
    lt = vrate_lt | altdiff_lt  | chi_angle_lt | psi_angle_lt | sr_lt | tds_lt | timer_lt | psid_lt | v_lt | alt_lt | abs_altdiff_lt
    vrate_lt = Expr(:comparison, vrate_feat, :.<, vrate_val) | Expr(:comparison, vrate_feat, :.<, vrate_feat)
    altdiff_lt = Expr(:comparison, altdiff_feat, :.<, altdiff_val)
    chi_angle_lt = Expr(:comparison, chi_angle_feat, :.<, angle_val) | Expr(:comparison, chi_angle_feat, :.<, chi_angle_feat)
    psi_angle_lt = Expr(:comparison, psi_angle_feat, :.<, psi_angle_feat)
    sr_lt = Expr(:comparison, sr_feat, :.<, sr_val)
    tds_lt = Expr(:comparison, tds_feat, :.<, tds_val) | Expr(:comparison, tds_feat, :.<, tds_feat)
    timer_lt = Expr(:comparison, timer_feat, :.<, timer_val) | Expr(:comparison, timer_feat, :.<, timer_feat)
    psid_lt = Expr(:comparison, psid_feat, :.<, psid_val) | Expr(:comparison, psid_feat, :.<, psid_feat)
    v_lt = Expr(:comparison, v_feat, :.<, v_val) | Expr(:comparison, v_feat, :.<, v_feat)
    alt_lt = Expr(:comparison, alt_feat, :.<, alt_val) | Expr(:comparison, alt_feat, :.<, alt_feat)
    abs_altdiff_lt = Expr(:comparison, abs_altdiff_feat, :.<, abs_altdiff_val)

    #absolute less than
    abslt = vrate_abslt | altdiff_abslt  | chi_angle_abslt | psi_angle_abslt | sr_abslt | tds_abslt | timer_abslt | psid_abslt | v_abslt | alt_abslt
    vrate_abslt = Expr(:call, :ablt, vrate_feat, vrate_val) | Expr(:call, :ablt, vrate_feat, vrate_feat)
    altdiff_abslt = Expr(:call, :ablt, altdiff_feat, altdiff_val)
    chi_angle_abslt = Expr(:call, :ablt, chi_angle_feat, angle_val) | Expr(:call, :ablt, chi_angle_feat, chi_angle_feat)
    psi_angle_abslt = Expr(:call, :ablt, psi_angle_feat, psi_angle_feat)
    sr_abslt = Expr(:call, :ablt, sr_feat, sr_val)
    tds_abslt = Expr(:call, :ablt, tds_feat, tds_val) | Expr(:call, :ablt, tds_feat, tds_feat)
    timer_abslt = Expr(:call, :ablt, timer_feat, timer_val) | Expr(:call, :ablt, timer_feat, timer_feat)
    psid_abslt = Expr(:call, :ablt, psid_feat, psid_val) | Expr(:call, :ablt, psid_feat, psid_feat)
    v_abslt = Expr(:call, :ablt, v_feat, v_val) | Expr(:call, :ablt, v_feat, v_feat)
    alt_abslt = Expr(:call, :ablt, alt_feat, alt_val) | Expr(:call, :ablt, alt_feat, alt_feat)

    #less then or equal
    lte = vrate_lte | altdiff_lte | chi_angle_lte | psi_angle_lte | sr_lte | tds_lte | timer_lte | psid_lte | v_lte | alt_lte | abs_altdiff_lte
    vrate_lte = Expr(:comparison, vrate_feat, :.<=, vrate_val) | Expr(:comparison, vrate_feat, :.<=, vrate_feat)
    altdiff_lte = Expr(:comparison, altdiff_feat, :.<=, altdiff_val)
    chi_angle_lte = Expr(:comparison, chi_angle_feat, :.<=, angle_val) | Expr(:comparison, chi_angle_feat, :.<=, chi_angle_feat)
    psi_angle_lte = Expr(:comparison, psi_angle_feat, :.<=, psi_angle_feat)
    sr_lte = Expr(:comparison, sr_feat, :.<=, sr_val)
    tds_lte = Expr(:comparison, tds_feat, :.<=, tds_val) | Expr(:comparison, tds_feat, :.<=, tds_feat)
    timer_lte = Expr(:comparison, timer_feat, :.<=, timer_val) | Expr(:comparison, timer_feat, :.<=, timer_feat)
    psid_lte = Expr(:comparison, psid_feat, :.<=, psid_val) | Expr(:comparison, psid_feat, :.<=, psid_feat)
    v_lte = Expr(:comparison, v_feat, :.<=, v_val) | Expr(:comparison, v_feat, :.<=, v_feat)
    alt_lte = Expr(:comparison, alt_feat, :.<=, alt_val) | Expr(:comparison, alt_feat, :.<=, alt_feat)
    abs_altdiff_lte = Expr(:comparison, abs_altdiff_feat, :.<=, abs_altdiff_val)

    #abs less then or equal
    abslte = vrate_abslte | altdiff_abslte | chi_angle_abslte | psi_angle_abslte | sr_abslte | tds_abslte | timer_abslte | psid_abslte | v_abslte | alt_abslte
    vrate_abslte = Expr(:call, :able, vrate_feat, vrate_val) | Expr(:call, :able, vrate_feat, vrate_feat)
    altdiff_abslte = Expr(:call, :able, altdiff_feat, altdiff_val)
    chi_angle_abslte = Expr(:call, :able, chi_angle_feat, angle_val) | Expr(:call, :able, chi_angle_feat, chi_angle_feat)
    psi_angle_abslte = Expr(:call, :able, psi_angle_feat, psi_angle_feat)
    sr_abslte = Expr(:call, :able, sr_feat, sr_val)
    tds_abslte = Expr(:call, :able, tds_feat, tds_val) | Expr(:call, :able, tds_feat, tds_feat)
    timer_abslte = Expr(:call, :able, timer_feat, timer_val) | Expr(:call, :able, timer_feat, timer_feat)
    psid_abslte = Expr(:call, :able, psid_feat, psid_val) | Expr(:call, :able, psid_feat, psid_feat)
    v_abslte = Expr(:call, :able, v_feat, v_val) | Expr(:call, :able, v_feat, v_feat)
    alt_abslte = Expr(:call, :able, alt_feat, alt_val) | Expr(:call, :able, alt_feat, alt_feat)

    #sign
    sign = vrate_sign | chi_angle_sign | psid_sign
    vrate_sign = Expr(:call, :sn, vrate_feat, vrate_feat)
    chi_angle_sign = Expr(:call, :sn, chi_angle_feat, chi_angle_feat)
    psid_sign = Expr(:call, :sn, psid_feat, psid_feat)

    #difference is equal
    diff_eq = vrate_diff_eq | chi_angle_diff_eq | psi_angle_diff_eq | tds_diff_eq | timer_diff_eq | psid_diff_eq | v_diff_eq
    vrate_diff_eq = Expr(:call, :dfeq, vrate_feat, vrate_feat, vrate_val)
    chi_angle_diff_eq = Expr(:call, :dfeq, chi_angle_feat, chi_angle_feat, angle_val)
    psi_angle_diff_eq = Expr(:call, :dfeq, psi_angle_feat, psi_angle_feat, angle_val)
    tds_diff_eq = Expr(:call, :dfeq, tds_feat, tds_feat, tds_val)
    timer_diff_eq = Expr(:call, :dfeq, timer_feat, timer_feat, timer_val)
    psid_diff_eq = Expr(:call, :dfeq, psid_feat, psid_feat, psid_val)
    v_diff_eq = Expr(:call, :dfeq, v_feat, v_feat, v_val)

    #abs difference is equal
    absdiff_eq = vrate_absdiff_eq | chi_angle_absdiff_eq | psi_angle_absdiff_eq | tds_absdiff_eq | timer_absdiff_eq | psid_absdiff_eq | v_absdiff_eq
    vrate_absdiff_eq = Expr(:call, :adfeq, vrate_feat, vrate_feat, vrate_val)
    chi_angle_absdiff_eq = Expr(:call, :adfeq, chi_angle_feat, chi_angle_feat, angle_val)
    psi_angle_absdiff_eq = Expr(:call, :adfeq, psi_angle_feat, psi_angle_feat, angle_val)
    tds_absdiff_eq = Expr(:call, :adfeq, tds_feat, tds_feat, tds_val)
    timer_absdiff_eq = Expr(:call, :adfeq, timer_feat, timer_feat, timer_val)
    psid_absdiff_eq = Expr(:call, :adfeq, psid_feat, psid_feat, psid_val)
    v_absdiff_eq = Expr(:call, :adfeq, v_feat, v_feat, v_val)

    #difference is less than some value.  Values may not be appropriate after taking difference
    diff_lt = vrate_diff_lt | chi_angle_diff_lt | psi_angle_diff_lt | tds_diff_lt | timer_diff_lt | psid_diff_lt | v_diff_lt
    vrate_diff_lt = Expr(:call, :dflt, vrate_feat, vrate_feat, vrate_val)
    chi_angle_diff_lt = Expr(:call, :dflt, chi_angle_feat, chi_angle_feat, angle_val)
    psi_angle_diff_lt = Expr(:call, :dflt, psi_angle_feat, psi_angle_feat, angle_val)
    tds_diff_lt = Expr(:call, :dflt, tds_feat, tds_feat, tds_val)
    timer_diff_lt = Expr(:call, :dflt, timer_feat, timer_feat, timer_val)
    psid_diff_lt = Expr(:call, :dflt, psid_feat, psid_feat, psid_val)
    v_diff_lt = Expr(:call, :dflt, v_feat, v_feat, v_val)

    #abs difference is less than some value.  Values may not be appropriate after taking difference
    absdiff_lt = vrate_absdiff_lt | chi_angle_absdiff_lt | psi_angle_absdiff_lt | tds_absdiff_lt | timer_absdiff_lt | psid_absdiff_lt | v_absdiff_lt
    vrate_absdiff_lt = Expr(:call, :adflt, vrate_feat, vrate_feat, vrate_val)
    chi_angle_absdiff_lt = Expr(:call, :adflt, chi_angle_feat, chi_angle_feat, angle_val)
    psi_angle_absdiff_lt = Expr(:call, :adflt, psi_angle_feat, psi_angle_feat, angle_val)
    tds_absdiff_lt = Expr(:call, :adflt, tds_feat, tds_feat, tds_val)
    timer_absdiff_lt = Expr(:call, :adflt, timer_feat, timer_feat, timer_val)
    psid_absdiff_lt = Expr(:call, :adflt, psid_feat, psid_feat, psid_val)
    v_absdiff_lt = Expr(:call, :adflt, v_feat, v_feat, v_val)

    #difference is less than or equal to some value.  Values may not be appropriate after taking difference
    diff_lte = vrate_diff_lte | chi_angle_diff_lte | psi_angle_diff_lte | tds_diff_lte | timer_diff_lte | psid_diff_lte | v_diff_lte
    vrate_diff_lte = Expr(:call, :dfle, vrate_feat, vrate_feat, vrate_val)
    chi_angle_diff_lte = Expr(:call, :dfle, chi_angle_feat, chi_angle_feat, angle_val)
    psi_angle_diff_lte = Expr(:call, :dfle, psi_angle_feat, psi_angle_feat, angle_val)
    tds_diff_lte = Expr(:call, :dfle, tds_feat, tds_feat, tds_val)
    timer_diff_lte = Expr(:call, :dfle, timer_feat, timer_feat, timer_val)
    psid_diff_lte = Expr(:call, :dfle, psid_feat, psid_feat, psid_val)
    v_diff_lte = Expr(:call, :dfle, v_feat, v_feat, v_val)

    #absolute difference is less than or equal to some value.
    absdiff_lte = vrate_absdiff_lte | chi_angle_absdiff_lte | psi_angle_absdiff_lte | tds_absdiff_lte | timer_absdiff_lte | psid_absdiff_lte | v_absdiff_lte
    vrate_absdiff_lte = Expr(:call, :adfle, vrate_feat, vrate_feat, vrate_val)
    chi_angle_absdiff_lte = Expr(:call, :adfle, chi_angle_feat, chi_angle_feat, angle_val)
    psi_angle_absdiff_lte = Expr(:call, :adfle, psi_angle_feat, psi_angle_feat, angle_val)
    tds_absdiff_lte = Expr(:call, :adfle, tds_feat, tds_feat, tds_val)
    timer_absdiff_lte = Expr(:call, :adfle, timer_feat, timer_feat, timer_val)
    psid_absdiff_lte = Expr(:call, :adfle, psid_feat, psid_feat, psid_val)
    v_absdiff_lte = Expr(:call, :adfle, v_feat, v_feat, v_val)

    #read features
    bin_feat = Expr(:ref, :D, :(:), bin_feat_id)
    vrate_feat = Expr(:ref, :D, :(:), vrate_feat_id)
    altdiff_feat = Expr(:ref, :D, :(:), altdiff_feat_id)
    abs_altdiff_feat = Expr(:ref, :D, :(:), abs_altdiff_feat_id)
    angle_feat = Expr(:ref, :D, :(:), angle_feat_id)
    psi_angle_feat = Expr(:ref, :D, :(:), psi_angle_feat_id)
    chi_angle_feat = Expr(:ref, :D, :(:), chi_angle_feat_id)
    sr_feat = Expr(:ref, :D, :(:), sr_feat_id)
    tds_feat = Expr(:ref, :D, :(:), tds_feat_id)
    timer_feat = Expr(:ref, :D, :(:), timer_feat_id)
    psid_feat = Expr(:ref, :D, :(:), psid_feat_id)
    v_feat = Expr(:ref, :D, :(:), v_feat_id)
    alt_feat = Expr(:ref, :D, :(:), alt_feat_id)

    #indices of each type
    bin_feat_id = 1 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 23 | 24 | 25 | 26 | 27 | 28 | 30 | 31 | 32 | 38 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 60 | 61 | 62 | 63 | 64 | 65 | 67 | 68 | 69 | 75
    vrate_feat_id = 2 | 22 | 34 | 39 | 59 | 71
    altdiff_feat_id = 3 | 40
    abs_altdiff_feat_id = 76
    angle_feat_id = psi_angle_feat_id | chi_angle_feat_id
    psi_angle_feat_id = 4 | 41
    chi_angle_feat_id = 6 | 43
    sr_feat_id = 5 | 42 | 77
    tds_feat_id = 29 | 66
    timer_feat_id = 33 | 70
    psid_feat_id = 35 | 72
    v_feat_id = 36 | 73
    alt_feat_id = 37 | 74

    #values of each type
    vrate_val = -50 | -40 | -30 | -20 | -10 | -1 | 0 | 1 | 10 | 20 | 30 | 40 | 50
    altdiff_val = -2000 | -1500 | -1000 | -500 | -250 | -100 | -50 | -25 | -10 | -5 | -1 | 0 | 1 | 5 | 10 | 25 | 50 | 100 | 250 | 500 | 1000 | 1500 | 2000
    abs_altdiff_val = 0 | 1 | 5 | 10 | 25 | 50 | 100 | 250 | 500 | 1000 | 1500 | 2000
    angle_val = -180 | -135 | -90 | -45 | 0 | 45 | 90 | 135 | 180
    sr_val = 30000 | 25000 | 20000 | 15000 | 10000 | 7500 | 5000 | 2500 | 1000 | 500 | 250 | 100 | 50 | 25 | 10 | 1 | 0
    tds_val = 0 | 50 | 100 | 250 | 400 | 1000 | 1200
    timer_val = 0:5
    psid_val = -10:10
    v_val = 50 | 75 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500
    alt_val = 1000 | 2000 | 3000 | 5000 | 10000 | 15000 | 18000 | 20000 | 25000 | 29000
    timestep = 0:50
  end

  #automatically determine real vs bool columns from DataFrame
  #bin_ids, _ = feat_type_ids(D)
  #bin_terms = map(GrammaticalEvolution.Terminal, bin_ids)
  #grammar.rules[:bin_feat_id] = OrRule("bin_feat_id", bin_terms, nothing)

  return grammar
end

abs_eq(v::RealVec, b::Real) = @devec r = abs(v) .== b
abs_lte(v::RealVec, b::Real) = @devec r = abs(v) .<= b
abs_lt(v::RealVec, b::Real) = @devec r = abs(v) .< b

abs_eq(v1::RealVec, v2::RealVec) = @devec r = abs(v1) .== abs(v2)
abs_lte(v1::RealVec, v2::RealVec) = @devec r = abs(v1) .<= abs(v2)
abs_lt(v1::RealVec, v2::RealVec) = @devec r = abs(v1) .< abs(v2)

diff_eq(v1::RealVec, v2::RealVec, b::Real) = @devec r = (v1 - v2) .== b
diff_lte(v1::RealVec, v2::RealVec, b::Real) = @devec r = (v1 - v2) .<= b
diff_lt(v1::RealVec, v2::RealVec, b::Real) = @devec r = (v1 - v2) .< b

abs_diff_eq(v1::RealVec, v2::RealVec, b::Real) = @devec r = abs(v1 - v2) .== b
abs_diff_lte(v1::RealVec, v2::RealVec, b::Real) = @devec r = abs(v1 - v2) .<= b
abs_diff_lt(v1::RealVec, v2::RealVec, b::Real) = @devec r = abs(v1 - v2) .< b

eventually(v::AbstractVector{Bool}) = any(v)
globally(v::AbstractVector{Bool}) = all(v)
implies(v1::AbstractVector{Bool}, v2::AbstractVector{Bool}) = all(!v1 | v2)

sign_(v1::RealVec, v2::RealVec) = @devec r = (sign(v1) .* sign(v2)) .>= 0.0 #same sign, 0 matches any sign

count_eq(v::AbstractVector{Bool}, b::Real) = count(identity, v) == b
count_lt(v::AbstractVector{Bool}, b::Real) = count(identity, v) < b
count_lte(v::AbstractVector{Bool}, b::Real) = count(identity, v) <= b
count_gt(v::AbstractVector{Bool}, b::Real) = count(identity, v) > b
count_gte(v::AbstractVector{Bool}, b::Real) = count(identity, v) >= b

#shorthands used in grammar to reduce impact on code length
abeq = abs_eq
able = abs_lte
ablt = abs_lt
dfeq = diff_eq
dfle = diff_lte
dflt = diff_lt
adfeq = abs_diff_eq
adfle = abs_diff_lte
adflt = abs_diff_lt
F = eventually
G = globally
Y = implies
sn = sign_ #avoid conflict with Base.sign
ctlt = count_lt
ctle = count_lte
ctgt = count_gt
ctge = count_gte
cteq = count_eq

function to_function(problem::ACASXClustering, expr)
  @eval f(D) = $expr
  return f
end

function ExprSearch.get_fitness{T}(problem::ACASXClustering{T}, expr)
  Dl = problem.Dl

  codelen = length(string(expr))

  f = to_function(problem, expr)
  predicts = Array(Bool, length(Dl.records))
  for i = 1:length(Dl.records)
    predicts[i] = f(Dl.records[i])
  end
  _, _, ent_post = get_metrics(predicts, Dl.labels)
  return problem.w_ent * ent_post + problem.w_len * codelen
end

function ExprSearch.get_fitness{T}(problem::ACASXClustering{T}, expr,
                                   thresh::Float64, default::Float64)
  Dl = problem.Dl
  codelen = length(string(expr))
  f = to_function(problem, expr)

  maxent = (thresh - problem.w_len * codelen) / problem.w_len

  predicts = Array(Bool, length(Dl.records))
  for i = 1:length(Dl.records)
    predicts[i] = f(Dl.records[i])

    #early exit
    ent = computebound()
    if ent > maxent
      return default
    end
  end

  #do full calc
  _, _, ent_post = get_metrics(predicts, Dl.labels)
  return problem.w_ent * ent_post + problem.w_len * codelen
end

end #module
