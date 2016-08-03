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

function nmac_clusters(clustering::DataFrame, Ds::DFSet)
    meta = getmeta(Ds)
    D = join(meta, clustering, on=[:encounter_id])
    D = D[D[:nmac], :] #nmacs only
    ids = D[:id]
    labels = D[:label]
    Dl = DFSetLabeled(Ds[ids], labels)
    Dl
end

function nonnmacs_extra_cluster(clustering::DataFrame, Ds::DFSet)
    Dl1 = nmac_clusters(clustering, Ds) #cluster labeled nmacs
    
    meta = getmeta(Ds)
    D = meta[!meta[:nmac], :] #non-nmacs only
    extra_label = maximum(clustering[:label]) + 1 #highest cluster label + 1
    ids = D[:id]
    labels = fill(extra_label, length(ids))
    Dl2 = DFSetLabeled(Ds[ids], labels)
    Dl = vcat(Dl1, Dl2)
    Dl
end

function nmacs_vs_nonnmacs(Ds::DFSet)
    meta = getmeta(Ds)
    labels = map(b -> b ? 1 : 2, meta[:nmac]) #remap to 1=nmac,2=nonnmac
    labels = convert(Vector{Int64}, labels)
    Dl = DFSetLabeled(Ds, labels)
    Dl
end
