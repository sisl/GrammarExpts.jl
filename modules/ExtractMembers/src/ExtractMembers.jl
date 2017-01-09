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
Extract and organize json files into subdirectories according to decision id and 
members from clustering results.
Use RLESCAS json_update_recur (if needed) and recursive_plot to generate the plots
recursively on the subdirectories.
"""
module ExtractMembers

export extract_members

using DataFrames

const DATADIR = Pkg.dir("Datasets/data")
const METAFILE = "_META.csv.gz"

#TODO: remove depedence on metafile
function extract_members(memberfile::AbstractString, jsondir::AbstractString, 
    jsonroot::AbstractString, dataset::AbstractString, outdir::AbstractString, 
    N::Int64; jsonext::AbstractString="json.gz")

    metafile = joinpath(DATADIR, dataset, METAFILE)
    meta = readtable(metafile)
    D = readtable(memberfile)
    for i = 1:nrow(D)
        id = D[i, 1]
        members = string2array(D[i, 2])
        subdir = joinpath(outdir, string(id))
        mkpath(subdir)
        for j = 1:N
            m = members[j]
            enc = meta[m, :encounter_id]
            fname = "$jsonroot$enc.$jsonext"
            @show fname
            @show isfile(joinpath(jsondir,fname))
            cp(joinpath(jsondir, fname), joinpath(subdir, fname))
        end
    end
end

function string2array(str_array::AbstractString; T::Type=Int64)
    str = replace(str_array, "[", "")
    str = replace(str, "]", "")
    A = split(str, ',')
    A = map(s->parse(T, s), A)
    A
end

end
