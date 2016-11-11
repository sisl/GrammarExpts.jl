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

import Compat.ASCIIString
using DerivationTrees

#note: tds columns are not used, but left in the data
const COLNAMES = [
  "RA_1",
  "vert_rate_1",
  "alt_diff_1",
  "psi_1",
  "intr_sr_1",
  "intr_chi_1",
  "intr_vrc0_1",
  "intr_vrc1_1",
  "intr_vrc2_1",
  "cc0_1",
  "cc1_1",
  "cc2_1",
  "vc0_1",
  "vc1_1",
  "vc2_1",
  "ua0_1",
  "ua1_1",
  "ua2_1",
  "da0_1",
  "da1_1",
  "da2_1",
  "target_rate_1",
  "crossing_1",
  "alarm_1",
  "alert_1",
  "intr_out_vrc0_1",
  "intr_out_vrc1_1",
  "intr_out_vrc2_1",
  "intr_out_tds_1",
  "response_none_1",
  "response_stay_1",
  "response_follow_1",
  "response_timer_1",
  "response_h_d_1",
  "response_psi_d_1",
  "v_1",
  "h_1",
  "RA_2",
  "vert_rate_2",
  "alt_diff_2",
  "psi_2",
  "intr_sr_2",
  "intr_chi_2",
  "intr_vrc0_2",
  "intr_vrc1_2",
  "intr_vrc2_2",
  "cc0_2",
  "cc1_2",
  "cc2_2",
  "vc0_2",
  "vc1_2",
  "vc2_2",
  "ua0_2",
  "ua1_2",
  "ua2_2",
  "da0_2",
  "da1_2",
  "da2_2",
  "target_rate_2",
  "crossing_2",
  "alarm_2",
  "alert_2",
  "intr_out_vrc0_2",
  "intr_out_vrc1_2",
  "intr_out_vrc2_2",
  "intr_out_tds_2",
  "response_none_2",
  "response_stay_2",
  "response_follow_2",
  "response_timer_2",
  "response_h_d_2",
  "response_psi_d_2",
  "v_2",
  "h_2",
  "converging",
  "abs_alt_diff",
  "horizontal_range"]

function get_map(num_aircraft::Int64 = 2)
  D = Dict{ASCIIString,ASCIIString}()
  for i = 1:num_aircraft
    D["RA_$i"] = "RA is active on aircraft $i"
    D["vert_rate_$i"] = "aircraft $i's vertical rate"
    D["alt_diff_$i"] = "altitude difference relative to aircraft $i"
    D["psi_$i"] = "aircraft $i's heading angle"
    D["intr_sr_$i"] = "aircraft $i's intruder slant range"
    D["intr_chi_$i"] = "aircraft $i's intruder bearing"
    D["intr_vrc0_$i"] = "aircraft $i's input vrc bit 0 is set"
    D["intr_vrc1_$i"] = "aircraft $i's input vrc bit 1 is set"
    D["intr_vrc2_$i"] = "aircraft $i's input vrc bit 2 is set"
    D["cc0_$i"] = "aircraft $i's cc bit 0 is set"
    D["cc1_$i"] = "aircraft $i's cc bit 1 is set"
    D["cc2_$i"] = "aircraft $i's cc bit 2 is set"
    D["vc0_$i"] = "aircraft $i's vc bit 0 is set"
    D["vc1_$i"] = "aircraft $i's vc bit 1 is set"
    D["vc2_$i"] = "aircraft $i's vc bit 2 is set"
    D["ua0_$i"] = "aircraft $i's ua bit 0 is set"
    D["ua1_$i"] = "aircraft $i's ua bit 1 is set"
    D["ua2_$i"] = "aircraft $i's ua bit 2 is set"
    D["da0_$i"] = "aircraft $i's da bit 0 is set"
    D["da1_$i"] = "aircraft $i's da bit 1 is set"
    D["da2_$i"] = "aircraft $i's da bit 2 is set"
    D["target_rate_$i"] = "aircraft $i's RA target rate"
    D["crossing_$i"] = "crossing RA is issued to aircraft $i"
    D["alarm_$i"] = "RA alarm occurs on aircraft $i"
    D["alert_$i"] = "RA alert occurs aircraft $i"
    D["intr_out_vrc0_$i"] = "aircraft $i's output vrc bit 0 is set"
    D["intr_out_vrc1_$i"] = "aircraft $i's output vrc bit 1 is set"
    D["intr_out_vrc2_$i"] = "aircraft $i's output vrc bit 2 is set"
    D["intr_out_tds_$i"] = "aircraft $i's output tds"
    D["response_none_$i"] = "pilot $i is flying intended trajectory"
    D["response_stay_$i"] = "pilot $i is responding to previous RA"
    D["response_follow_$i"] = "pilot $i is responding to current RA"
    D["response_timer_$i"] = "number of seconds remaining in pilot $i's response delay"
    D["response_h_d_$i"] = "pilot $i's commanded vertical rate"
    D["response_psi_d_$i"] = "pilot $i's commanded turn rate"
    D["v_$i"] = "aircraft $i's velocity"
    D["h_$i"] = "aircraft $i's altitude"
  end
  D["converging"] = "aircraft are converging"
  D["abs_alt_diff"] = "absolute altitude difference"
  D["horizontal_range"] = "horizontal range"
  return D
end

const U_LTE = "<=" #"\u2264"
const U_GTE = ">=" #"\u2265"
const U_IMPLIES = "=>"

function get_format_pretty{T<:AbstractString}(colnames::Vector{T})
  fmt = Format()

  fmt["always"] = (cmd, args) -> "G($(args[1]))"
  fmt["eventually"] = (cmd, args) -> "F($(args[1]))"

  bin_infix(cmd, args, insym) = "($(args[1]) $insym $(args[2]))"
  fmt["and"] = (cmd, args) -> bin_infix(cmd, args, "&")
  fmt["or"] = (cmd, args) -> bin_infix(cmd, args, "|")
  fmt["not"] = (cmd, args) -> "!($(args[1]))"
  fmt["implies"] = (cmd, args) -> "$(args[1]) => $(args[2])"

  count_op(cmd, args, op) = "(count($(args[1])) $op $(args[2]))"
  fmt["count.1.1.1.1"] = (cmd, args) -> count_op(cmd, args, "<")
  fmt["count.1.1.1.2"] = (cmd, args) -> count_op(cmd, args, "<=")
  fmt["count.1.1.2"] = (cmd, args) -> count_op(cmd, args, ">")
  fmt["count.1.2"] = (cmd, args) -> count_op(cmd, args, ">=")
  fmt["count.2"] = (cmd, args) -> count_op(cmd, args, "==")

  bin_infix_eq(cmd, args) = bin_infix(cmd, args, ".==")
  bin_infix_lt(cmd, args) = bin_infix(cmd, args, ".<")
  bin_infix_lte(cmd, args) = bin_infix(cmd, args, ".<=")
  fmt["vrate_eq.1"] = bin_infix_eq
  fmt["vrate_eq.2"] = bin_infix_eq
  fmt["altdiff_eq"] = bin_infix_eq
  fmt["chi_angle_eq.1"] = bin_infix_eq
  fmt["chi_angle_eq.2"] = bin_infix_eq
  fmt["psi_angle_eq"] = bin_infix_eq
  fmt["sr_eq"] = bin_infix_eq
  fmt["timer_eq.1"] = bin_infix_eq
  fmt["timer_eq.2"] = bin_infix_eq
  fmt["psid_eq.1"] = bin_infix_eq
  fmt["psid_eq.2"] = bin_infix_eq
  fmt["v_eq.1"] = bin_infix_eq
  fmt["v_eq.2"] = bin_infix_eq
  fmt["alt_eq.1"] = bin_infix_eq
  fmt["alt_eq.2"] = bin_infix_eq
  fmt["abs_altdiff_eq"] = bin_infix_eq
  fmt["vrate_lt.1"] = bin_infix_lt
  fmt["vrate_lt.2"] = bin_infix_lt
  fmt["altdiff_lt"] = bin_infix_lt
  fmt["chi_angle_lt.1"] = bin_infix_lt
  fmt["chi_angle_lt.2"] = bin_infix_lt
  fmt["psi_angle_lt"] = bin_infix_lt
  fmt["sr_lt"] = bin_infix_lt
  fmt["timer_lt.1"] = bin_infix_lt
  fmt["timer_lt.2"] = bin_infix_lt
  fmt["psid_lt.1"] = bin_infix_lt
  fmt["psid_lt.2"] = bin_infix_lt
  fmt["v_lt.1"] = bin_infix_lt
  fmt["v_lt.2"] = bin_infix_lt
  fmt["alt_lt.1"] = bin_infix_lt
  fmt["alt_lt.2"] = bin_infix_lt
  fmt["abs_altdiff_lt"] = bin_infix_lt
  fmt["vrate_lte.1"] = bin_infix_lte
  fmt["vrate_lte.2"] = bin_infix_lte
  fmt["altdiff_lte"] = bin_infix_lte
  fmt["chi_angle_lte.1"] = bin_infix_lte
  fmt["chi_angle_lte.2"] = bin_infix_lte
  fmt["psi_angle_lte"] = bin_infix_lte
  fmt["sr_lte"] = bin_infix_lte
  fmt["timer_lte.1"] = bin_infix_lte
  fmt["timer_lte.2"] = bin_infix_lte
  fmt["psid_lte.1"] = bin_infix_lte
  fmt["psid_lte.2"] = bin_infix_lte
  fmt["v_lte.1"] = bin_infix_lte
  fmt["v_lte.2"] = bin_infix_lte
  fmt["alt_lte.1"] = bin_infix_lte
  fmt["alt_lte.2"] = bin_infix_lte
  fmt["abs_altdiff_lte"] = bin_infix_lte

  abs_bin_infix_op(cmd, args, op) = "(abs($(args[1])) $op $(args[2]))"
  abs_bin_infix_eq(cmd, args) = abs_bin_infix_op(cmd, args, ".==")
  abs_bin_infix_lt(cmd, args) = abs_bin_infix_op(cmd, args, ".<")
  abs_bin_infix_lte(cmd, args) = abs_bin_infix_op(cmd, args, ".<=")
  fmt["vrate_abseq.1"] = abs_bin_infix_eq
  fmt["vrate_abseq.2"] = abs_bin_infix_eq
  fmt["altdiff_abseq"] = abs_bin_infix_eq
  fmt["chi_angle_abseq.1"] = abs_bin_infix_eq
  fmt["chi_angle_abseq.2"] = abs_bin_infix_eq
  fmt["psi_angle_abseq"] = abs_bin_infix_eq
  fmt["sr_abseq"] = abs_bin_infix_eq
  fmt["timer_abseq.1"] = abs_bin_infix_eq
  fmt["timer_abseq.2"] = abs_bin_infix_eq
  fmt["psid_abseq.1"] = abs_bin_infix_eq
  fmt["psid_abseq.2"] = abs_bin_infix_eq
  fmt["v_abseq.1"] = abs_bin_infix_eq
  fmt["v_abseq.2"] = abs_bin_infix_eq
  fmt["alt_abseq.1"] = abs_bin_infix_eq
  fmt["alt_abseq.2"] = abs_bin_infix_eq
  fmt["vrate_abslt.1"] = abs_bin_infix_lt
  fmt["vrate_abslt.2"] = abs_bin_infix_lt
  fmt["altdiff_abslt"] = abs_bin_infix_lt
  fmt["chi_angle_abslt.1"] = abs_bin_infix_lt
  fmt["chi_angle_abslt.2"] = abs_bin_infix_lt
  fmt["psi_angle_abslt"] = abs_bin_infix_lt
  fmt["sr_abslt"] = abs_bin_infix_lt
  fmt["timer_abslt.1"] = abs_bin_infix_lt
  fmt["timer_abslt.2"] = abs_bin_infix_lt
  fmt["psid_abslt.1"] = abs_bin_infix_lt
  fmt["psid_abslt.2"] = abs_bin_infix_lt
  fmt["v_abslt.1"] = abs_bin_infix_lt
  fmt["v_abslt.2"] = abs_bin_infix_lt
  fmt["alt_abslt.1"] = abs_bin_infix_lt
  fmt["alt_abslt.2"] = abs_bin_infix_lt
  fmt["vrate_abslte.1"] = abs_bin_infix_lte
  fmt["vrate_abslte.2"] = abs_bin_infix_lte
  fmt["altdiff_abslte"] = abs_bin_infix_lte
  fmt["chi_angle_abslte.1"] = abs_bin_infix_lte
  fmt["chi_angle_abslte.2"] = abs_bin_infix_lte
  fmt["psi_angle_abslte"] = abs_bin_infix_lte
  fmt["sr_abslte"] = abs_bin_infix_lte
  fmt["timer_abslte.1"] = abs_bin_infix_lte
  fmt["timer_abslte.2"] = abs_bin_infix_lte
  fmt["psid_abslte.1"] = abs_bin_infix_lte
  fmt["psid_abslte.2"] = abs_bin_infix_lte
  fmt["v_abslte.1"] = abs_bin_infix_lte
  fmt["v_abslte.2"] = abs_bin_infix_lte
  fmt["alt_abslte.1"] = abs_bin_infix_lte
  fmt["alt_abslte.2"] = abs_bin_infix_lte

  diff_infix_op(cmd, args, op) = "($(args[1]) - $(args[2]) $op $(args[3]))"
  diff_infix_eq(cmd, args) = diff_infix_op(cmd, args, ".==")
  diff_infix_lt(cmd, args) = diff_infix_op(cmd, args, ".<")
  diff_infix_lte(cmd, args) = diff_infix_op(cmd, args, ".<=")
  fmt["vrate_diff_eq"] = diff_infix_eq
  fmt["chi_angle_diff_eq"] = diff_infix_eq
  fmt["psi_angle_diff_eq"] = diff_infix_eq
  fmt["timer_diff_eq"] = diff_infix_eq
  fmt["psid_diff_eq"] = diff_infix_eq
  fmt["v_diff_eq"] = diff_infix_eq
  fmt["vrate_diff_lt"] = diff_infix_lt
  fmt["chi_angle_diff_lt"] = diff_infix_lt
  fmt["psi_angle_diff_lt"] = diff_infix_lt
  fmt["timer_diff_lt"] = diff_infix_lt
  fmt["psid_diff_lt"] = diff_infix_lt
  fmt["v_diff_lt"] = diff_infix_lt
  fmt["vrate_diff_lte"] = diff_infix_lte
  fmt["chi_angle_diff_lte"] = diff_infix_lte
  fmt["psi_angle_diff_lte"] = diff_infix_lte
  fmt["timer_diff_lte"] = diff_infix_lte
  fmt["psid_diff_lte"] = diff_infix_lte
  fmt["v_diff_lte"] = diff_infix_lte

  abs_diff_infix_op(cmd, args, op) = "(abs($(args[1]) - $(args[2])) $op $(args[3]))"
  abs_diff_infix_eq(cmd, args) = abs_diff_infix_op(cmd, args, ".==")
  abs_diff_infix_lt(cmd, args) = abs_diff_infix_op(cmd, args, ".<")
  abs_diff_infix_lte(cmd, args) = abs_diff_infix_op(cmd, args, ".<=")
  fmt["vrate_absdiff_eq"] = abs_diff_infix_eq
  fmt["chi_angle_absdiff_eq"] = abs_diff_infix_eq
  fmt["psi_angle_absdiff_eq"] = abs_diff_infix_eq
  fmt["timer_absdiff_eq"] = abs_diff_infix_eq
  fmt["psid_absdiff_eq"] = abs_diff_infix_eq
  fmt["v_absdiff_eq"] = abs_diff_infix_eq
  fmt["vrate_absdiff_lt"] = abs_diff_infix_lt
  fmt["chi_angle_absdiff_lt"] = abs_diff_infix_lt
  fmt["psi_angle_absdiff_lt"] = abs_diff_infix_lt
  fmt["timer_absdiff_lt"] = abs_diff_infix_lt
  fmt["psid_absdiff_lt"] = abs_diff_infix_lt
  fmt["v_absdiff_lt"] = abs_diff_infix_lt
  fmt["vrate_absdiff_lte"] = abs_diff_infix_lte
  fmt["chi_angle_absdiff_lte"] = abs_diff_infix_lte
  fmt["psi_angle_absdiff_lte"] = abs_diff_infix_lte
  fmt["timer_absdiff_lte"] = abs_diff_infix_lte
  fmt["psid_absdiff_lte"] = abs_diff_infix_lte
  fmt["v_absdiff_lte"] = abs_diff_infix_lte

  feat(cmd, args) = "$(colnames[parse(Int, args[1])])"
  fmt["bin_feat"] = feat
  fmt["vrate_feat"] = feat
  fmt["altdiff_feat"] = feat
  fmt["abs_altdiff_feat"] = feat
  fmt["angle_feat"] = feat
  fmt["psi_angle_feat"] = feat
  fmt["chi_angle_feat"] = feat
  fmt["sr_feat"] = feat
  fmt["timer_feat"] = feat
  fmt["psid_feat"] = feat
  fmt["v_feat"] = feat
  fmt["alt_feat"] = feat

  valunit(cmd, args, unit) = "$(args[1]) $unit"
  fmt["vrate_val"] = (cmd, args) -> valunit(cmd, args, "ft/s")
  fmt["altdiff_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["abs_altdiff_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["angle_val"] = (cmd, args) -> valunit(cmd, args, "deg")
  fmt["sr_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["timer_val"] = (cmd, args) -> valunit(cmd, args, "s")
  fmt["psid_val"] = (cmd, args) -> valunit(cmd, args, "deg/s")
  fmt["v_val"] = (cmd, args) -> valunit(cmd, args, "ft/s")
  fmt["alt_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["timestep"] = (cmd, args) -> valunit(cmd, args, "s")

  bin_sign(cmd, args) = "(sign($(args[1])) .== sign($(args[2])))"
  fmt["vrate_sign"] = bin_sign
  fmt["chi_angle_sign"] = bin_sign
  fmt["psid_sign"] = bin_sign

  return fmt
end

function get_format_natural{T<:AbstractString}(colnames::Vector{T})
  fmt = Format()

  fmt["always"] = (cmd, args) -> "for all time, $(args[1])"
  fmt["eventually"] = (cmd, args) -> "at some point, $(args[1])"

  bin_infix(cmd, args, insym) = "[$(args[1]) $insym $(args[2])]"
  fmt["and"] = (cmd, args) -> bin_infix(cmd, args, "and")
  fmt["or"] = (cmd, args) -> bin_infix(cmd, args, "or")
  fmt["not"] = (cmd, args) -> "[it is not true that $(args[1])]"
  fmt["implies"] = (cmd, args) -> "whenever $(args[1]), it is also true that $(args[2])"

  count_op(cmd, args, op) = "[the number of times $(args[1]) $op $(args[2])]"
  fmt["count.1.1.1.1"] = (cmd, args) -> count_op(cmd, args, "is less than")
  fmt["count.1.1.1.2"] = (cmd, args) -> count_op(cmd, args, "is less than or equal to")
  fmt["count.1.1.2"] = (cmd, args) -> count_op(cmd, args, "is greater than")
  fmt["count.1.2"] = (cmd, args) -> count_op(cmd, args, "is greater than or equal to")
  fmt["count.2"] = (cmd, args) -> count_op(cmd, args, "equals")

  bin_infix_eq(cmd, args) = bin_infix(cmd, args, "equals")
  bin_infix_lt(cmd, args) = bin_infix(cmd, args, "is less than")
  bin_infix_lte(cmd, args) = bin_infix(cmd, args, "is less than or equal to")
  fmt["vrate_eq.1"] = bin_infix_eq
  fmt["vrate_eq.2"] = bin_infix_eq
  fmt["altdiff_eq"] = bin_infix_eq
  fmt["chi_angle_eq.1"] = bin_infix_eq
  fmt["chi_angle_eq.2"] = bin_infix_eq
  fmt["psi_angle_eq"] = bin_infix_eq
  fmt["sr_eq"] = bin_infix_eq
  fmt["timer_eq.1"] = bin_infix_eq
  fmt["timer_eq.2"] = bin_infix_eq
  fmt["psid_eq.1"] = bin_infix_eq
  fmt["psid_eq.2"] = bin_infix_eq
  fmt["v_eq.1"] = bin_infix_eq
  fmt["v_eq.2"] = bin_infix_eq
  fmt["alt_eq.1"] = bin_infix_eq
  fmt["alt_eq.2"] = bin_infix_eq
  fmt["abs_altdiff_eq"] = bin_infix_eq
  fmt["vrate_lt.1"] = bin_infix_lt
  fmt["vrate_lt.2"] = bin_infix_lt
  fmt["altdiff_lt"] = bin_infix_lt
  fmt["chi_angle_lt.1"] = bin_infix_lt
  fmt["chi_angle_lt.2"] = bin_infix_lt
  fmt["psi_angle_lt"] = bin_infix_lt
  fmt["sr_lt"] = bin_infix_lt
  fmt["timer_lt.1"] = bin_infix_lt
  fmt["timer_lt.2"] = bin_infix_lt
  fmt["psid_lt.1"] = bin_infix_lt
  fmt["psid_lt.2"] = bin_infix_lt
  fmt["v_lt.1"] = bin_infix_lt
  fmt["v_lt.2"] = bin_infix_lt
  fmt["alt_lt.1"] = bin_infix_lt
  fmt["alt_lt.2"] = bin_infix_lt
  fmt["abs_altdiff_lt"] = bin_infix_lt
  fmt["vrate_lte.1"] = bin_infix_lte
  fmt["vrate_lte.2"] = bin_infix_lte
  fmt["altdiff_lte"] = bin_infix_lte
  fmt["chi_angle_lte.1"] = bin_infix_lte
  fmt["chi_angle_lte.2"] = bin_infix_lte
  fmt["psi_angle_lte"] = bin_infix_lte
  fmt["sr_lte"] = bin_infix_lte
  fmt["timer_lte.1"] = bin_infix_lte
  fmt["timer_lte.2"] = bin_infix_lte
  fmt["psid_lte.1"] = bin_infix_lte
  fmt["psid_lte.2"] = bin_infix_lte
  fmt["v_lte.1"] = bin_infix_lte
  fmt["v_lte.2"] = bin_infix_lte
  fmt["alt_lte.1"] = bin_infix_lte
  fmt["alt_lte.2"] = bin_infix_lte
  fmt["abs_altdiff_lte"] = bin_infix_lte

  abs_bin_infix_op(cmd, args, op) = "[the absolute value of $(args[1]) $op $(args[2]))]"
  abs_bin_infix_eq(cmd, args) = abs_bin_infix_op(cmd, args, "equals")
  abs_bin_infix_lt(cmd, args) = abs_bin_infix_op(cmd, args, "is less than")
  abs_bin_infix_lte(cmd, args) = abs_bin_infix_op(cmd, args, "is less than or equal to")
  fmt["vrate_abseq.1"] = abs_bin_infix_eq
  fmt["vrate_abseq.2"] = abs_bin_infix_eq
  fmt["altdiff_abseq"] = abs_bin_infix_eq
  fmt["chi_angle_abseq.1"] = abs_bin_infix_eq
  fmt["chi_angle_abseq.2"] = abs_bin_infix_eq
  fmt["psi_angle_abseq"] = abs_bin_infix_eq
  fmt["sr_abseq"] = abs_bin_infix_eq
  fmt["timer_abseq.1"] = abs_bin_infix_eq
  fmt["timer_abseq.2"] = abs_bin_infix_eq
  fmt["psid_abseq.1"] = abs_bin_infix_eq
  fmt["psid_abseq.2"] = abs_bin_infix_eq
  fmt["v_abseq.1"] = abs_bin_infix_eq
  fmt["v_abseq.2"] = abs_bin_infix_eq
  fmt["alt_abseq.1"] = abs_bin_infix_eq
  fmt["alt_abseq.2"] = abs_bin_infix_eq
  fmt["vrate_abslt.1"] = abs_bin_infix_lt
  fmt["vrate_abslt.2"] = abs_bin_infix_lt
  fmt["altdiff_abslt"] = abs_bin_infix_lt
  fmt["chi_angle_abslt.1"] = abs_bin_infix_lt
  fmt["chi_angle_abslt.2"] = abs_bin_infix_lt
  fmt["psi_angle_abslt"] = abs_bin_infix_lt
  fmt["sr_abslt"] = abs_bin_infix_lt
  fmt["timer_abslt.1"] = abs_bin_infix_lt
  fmt["timer_abslt.2"] = abs_bin_infix_lt
  fmt["psid_abslt.1"] = abs_bin_infix_lt
  fmt["psid_abslt.2"] = abs_bin_infix_lt
  fmt["v_abslt.1"] = abs_bin_infix_lt
  fmt["v_abslt.2"] = abs_bin_infix_lt
  fmt["alt_abslt.1"] = abs_bin_infix_lt
  fmt["alt_abslt.2"] = abs_bin_infix_lt
  fmt["vrate_abslte.1"] = abs_bin_infix_lte
  fmt["vrate_abslte.2"] = abs_bin_infix_lte
  fmt["altdiff_abslte"] = abs_bin_infix_lte
  fmt["chi_angle_abslte.1"] = abs_bin_infix_lte
  fmt["chi_angle_abslte.2"] = abs_bin_infix_lte
  fmt["psi_angle_abslte"] = abs_bin_infix_lte
  fmt["sr_abslte"] = abs_bin_infix_lte
  fmt["timer_abslte.1"] = abs_bin_infix_lte
  fmt["timer_abslte.2"] = abs_bin_infix_lte
  fmt["psid_abslte.1"] = abs_bin_infix_lte
  fmt["psid_abslte.2"] = abs_bin_infix_lte
  fmt["v_abslte.1"] = abs_bin_infix_lte
  fmt["v_abslte.2"] = abs_bin_infix_lte
  fmt["alt_abslte.1"] = abs_bin_infix_lte
  fmt["alt_abslte.2"] = abs_bin_infix_lte

  diff_infix_op(cmd, args, op) = "[the difference between $(args[1]) and $(args[2]) $op $(args[3])]"
  diff_infix_eq(cmd, args) = diff_infix_op(cmd, args, "equals")
  diff_infix_lt(cmd, args) = diff_infix_op(cmd, args, "is less than")
  diff_infix_lte(cmd, args) = diff_infix_op(cmd, args, "is less than or equal to")
  fmt["vrate_diff_eq"] = diff_infix_eq
  fmt["chi_angle_diff_eq"] = diff_infix_eq
  fmt["psi_angle_diff_eq"] = diff_infix_eq
  fmt["timer_diff_eq"] = diff_infix_eq
  fmt["psid_diff_eq"] = diff_infix_eq
  fmt["v_diff_eq"] = diff_infix_eq
  fmt["vrate_diff_lt"] = diff_infix_lt
  fmt["chi_angle_diff_lt"] = diff_infix_lt
  fmt["psi_angle_diff_lt"] = diff_infix_lt
  fmt["timer_diff_lt"] = diff_infix_lt
  fmt["psid_diff_lt"] = diff_infix_lt
  fmt["v_diff_lt"] = diff_infix_lt
  fmt["vrate_diff_lte"] = diff_infix_lte
  fmt["chi_angle_diff_lte"] = diff_infix_lte
  fmt["psi_angle_diff_lte"] = diff_infix_lte
  fmt["timer_diff_lte"] = diff_infix_lte
  fmt["psid_diff_lte"] = diff_infix_lte
  fmt["v_diff_lte"] = diff_infix_lte

  abs_diff_infix_op(cmd, args, op) = "[the absolute difference between $(args[1]) and $(args[2]) $op $(args[3])]"
  abs_diff_infix_eq(cmd, args) = abs_diff_infix_op(cmd, args, "equals")
  abs_diff_infix_lt(cmd, args) = abs_diff_infix_op(cmd, args, "is less than")
  abs_diff_infix_lte(cmd, args) = abs_diff_infix_op(cmd, args, "is less than or equal to")
  fmt["vrate_absdiff_eq"] = abs_diff_infix_eq
  fmt["chi_angle_absdiff_eq"] = abs_diff_infix_eq
  fmt["psi_angle_absdiff_eq"] = abs_diff_infix_eq
  fmt["timer_absdiff_eq"] = abs_diff_infix_eq
  fmt["psid_absdiff_eq"] = abs_diff_infix_eq
  fmt["v_absdiff_eq"] = abs_diff_infix_eq
  fmt["vrate_absdiff_lt"] = abs_diff_infix_lt
  fmt["chi_angle_absdiff_lt"] = abs_diff_infix_lt
  fmt["psi_angle_absdiff_lt"] = abs_diff_infix_lt
  fmt["timer_absdiff_lt"] = abs_diff_infix_lt
  fmt["psid_absdiff_lt"] = abs_diff_infix_lt
  fmt["v_absdiff_lt"] = abs_diff_infix_lt
  fmt["vrate_absdiff_lte"] = abs_diff_infix_lte
  fmt["chi_angle_absdiff_lte"] = abs_diff_infix_lte
  fmt["psi_angle_absdiff_lte"] = abs_diff_infix_lte
  fmt["timer_absdiff_lte"] = abs_diff_infix_lte
  fmt["psid_absdiff_lte"] = abs_diff_infix_lte
  fmt["v_absdiff_lte"] = abs_diff_infix_lte

  feat(cmd, args) = "[$(colnames[parse(Int, args[1])])]"
  fmt["bin_feat"] = feat
  fmt["vrate_feat"] = feat
  fmt["altdiff_feat"] = feat
  fmt["abs_altdiff_feat"] = feat
  fmt["angle_feat"] = feat
  fmt["psi_angle_feat"] = feat
  fmt["chi_angle_feat"] = feat
  fmt["sr_feat"] = feat
  fmt["timer_feat"] = feat
  fmt["psid_feat"] = feat
  fmt["v_feat"] = feat
  fmt["alt_feat"] = feat

  valunit(cmd, args, unit) = "$(args[1]) $unit"
  fmt["vrate_val"] = (cmd, args) -> valunit(cmd, args, "ft/s")
  fmt["altdiff_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["abs_altdiff_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["angle_val"] = (cmd, args) -> valunit(cmd, args, "deg")
  fmt["sr_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["timer_val"] = (cmd, args) -> valunit(cmd, args, "s")
  fmt["psid_val"] = (cmd, args) -> valunit(cmd, args, "deg/s")
  fmt["v_val"] = (cmd, args) -> valunit(cmd, args, "ft/s")
  fmt["alt_val"] = (cmd, args) -> valunit(cmd, args, "ft")
  fmt["timestep"] = (cmd, args) -> valunit(cmd, args, "s")

  bin_sign(cmd, args) = "[the sign of $(args[1]) is equal to the sign of $(args[2])]"
  fmt["vrate_sign"] = bin_sign
  fmt["chi_angle_sign"] = bin_sign
  fmt["psid_sign"] = bin_sign

  return fmt
end

const DESCRIP_MAP = get_map()
const COLNAMES_FULL = map(x -> DESCRIP_MAP[x], COLNAMES)
const FMT_PRETTY = get_format_pretty(COLNAMES)
const FMT_NATURAL = get_format_natural(COLNAMES_FULL)
