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

using GrammarExpts, ACASXProblem
using ExprSearch, DerivationTrees
using RLESUtils, CPUTimeUtils, Loggers, Observers
using DataFrameSets

"""
reps=inner loop reps before taking a time
trials=number of times to take
so the number of evals is trials*rep
"""
function test1(; 
    data::AbstractString="dasc",
    n_trials::Int64=50,
    n_reps::Int64=2500,
    verbose::Bool=true,
    b_get_fitness::Bool=false,
    b_emulated_fitness::Bool=false,
    b_predicts::Bool=false,
    b_emulated_predicts::Bool=false)

    problem = ACASXClustering(data)
    grammar = get_grammar(problem)
    p = DerivTreeParams(grammar, 20) 
    tree = DerivationTree(p) 

    #setup logs
    logs = TaggedDFLogger()
    add_folder!(logs, "params", [ASCIIString, Any], [:name, :value])
    add_folder!(logs, "rand", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "get_expr", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "get_fitness", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "to_function", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "to_function2", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "emulated_fitness", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "predicts", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "emulated_predicts", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "emulated_predicts2", [Int64, Float64], [:trial, :t_elapsed])
    add_folder!(logs, "metrics", [Int64, Float64], [:trial, :t_elapsed])
    observer = Observer()
    add_observer(observer, "params", push!_f(logs, "params"))
    add_observer(observer, "rand", push!_f(logs, "rand"))
    add_observer(observer, "get_expr", push!_f(logs, "get_expr"))
    add_observer(observer, "get_fitness", push!_f(logs, "get_fitness"))
    add_observer(observer, "to_function", push!_f(logs, "to_function"))
    add_observer(observer, "to_function2", push!_f(logs, "to_function2"))
    add_observer(observer, "emulated_fitness", push!_f(logs, "emulated_fitness"))
    add_observer(observer, "predicts", push!_f(logs, "predicts"))
    add_observer(observer, "emulated_predicts", push!_f(logs, "emulated_predicts"))
    add_observer(observer, "emulated_predicts2", push!_f(logs, "emulated_predicts2"))
    add_observer(observer, "metrics", push!_f(logs, "metrics"))

    @notify_observer(observer, "params", ["data", data])
    @notify_observer(observer, "params", ["n_trials", n_trials])
    @notify_observer(observer, "params", ["n_reps", n_reps])

    #time rand!
    println("timing rand!")
    for trial = 1:n_trials
        tstart = CPUtime_start()
        for i = 1:n_reps
            #########################
            rand!(tree, retries=typemax(Int64))
            #########################
        end
        t_elapsed = CPUtime_elapsed_s(tstart)
        @notify_observer(observer, "rand", [trial, t_elapsed]) 
    end
    #time get_expr
    println("timing get_expr")
    for trial = 1:n_trials
        tstart = CPUtime_start()
        for i = 1:n_reps
            #########################
            expr = get_expr(tree)
            #########################
        end
        t_elapsed = CPUtime_elapsed_s(tstart)
        @notify_observer(observer, "get_expr", [trial, t_elapsed]) 
    end
    expr = get_expr(tree)

    if b_get_fitness
        #time get_fitness
        println("timing get_fitness")
        for trial = 1:n_trials
            tstart = CPUtime_start()
            for i = 1:n_reps
                #########################
                fitness = get_fitness(problem, expr)
                #########################
            end
            t_elapsed = CPUtime_elapsed_s(tstart)
            @notify_observer(observer, "get_fitness", [trial, t_elapsed]) 
        end
    end
    #time to_function
    println("timing to_function")
    for trial = 1:n_trials
        tstart = CPUtime_start()
        for i = 1:n_reps
            #########################
            f = to_function(problem, expr)
            #########################
        end
        t_elapsed = CPUtime_elapsed_s(tstart)
        @notify_observer(observer, "to_function", [trial, t_elapsed]) 
    end
    #time to_function2
    D_1 = getrecords(problem.Dl, 1) #only the first record
    println("timing to_function2")
    for trial = 1:n_trials
        tstart = CPUtime_start()
        for i = 1:n_reps
            #########################
            rand!(tree, retries=typemax(Int64))
            expr = get_expr(tree)
            f = to_function(problem, expr)
            f(D_1)
            #########################
        end
        t_elapsed = CPUtime_elapsed_s(tstart)
        @notify_observer(observer, "to_function2", [trial, t_elapsed]) 
    end

    if b_emulated_fitness
        #time emulated_fitness
        println("timing emulated_fitness")
        for trial = 1:n_trials
            tstart = CPUtime_start()
            for i = 1:n_reps
                #########################
                rand!(tree, retries=typemax(Int64))
                expr = get_expr(tree)
                predicts = get_predicts(problem, expr)
                #########################
            end
            t_elapsed = CPUtime_elapsed_s(tstart)
            @notify_observer(observer, "emulated_fitness", [trial, t_elapsed]) 
        end
    end

    if b_predicts
        #time predicts
        println("timing predicts")
        for trial = 1:n_trials
            rand!(tree, retries=typemax(Int64))
            expr = get_expr(tree)
            println("trial=$trial, expr=", string(expr))
            tstart = CPUtime_start()
            for i = 1:n_reps
                #########################
                predicts = get_predicts(problem, expr)
                #########################
            end
            t_elapsed = CPUtime_elapsed_s(tstart)
            @notify_observer(observer, "predicts", [trial, t_elapsed]) 
        end
    end

    if b_emulated_predicts
        #time emulated_predicts
        println("timing emulated_predicts")
        for trial = 1:n_trials
            rand!(tree, retries=typemax(Int64))
            expr = get_expr(tree)
            println("trial=$trial, expr=", string(expr))
            tstart = CPUtime_start()
            for i = 1:n_reps
                #########################
                Dl = problem.Dl
                f = to_function(problem, expr)
                predicts = Array(Bool, length(Dl))
                for i = 1:length(Dl)
                    predicts[i] = f(getrecords(Dl, i))
                end
                #########################
            end
            t_elapsed = CPUtime_elapsed_s(tstart)
            @notify_observer(observer, "emulated_predicts", [trial, t_elapsed]) 
        end
    end

    #time emulated_predicts2
    println("timing emulated_predicts2")
    for trial = 1:n_trials
        rand!(tree, retries=typemax(Int64))
        expr = get_expr(tree)
        Dl = problem.Dl
        f = to_function(problem, expr)
        println("trial=$trial, expr=", string(expr))
        tstart = CPUtime_start()
        for i = 1:n_reps
            #########################
            predicts = Array(Bool, length(Dl))
            for i = 1:length(Dl)
                predicts[i] = f(getrecords(Dl, i))
            end
            #########################
        end
        t_elapsed = CPUtime_elapsed_s(tstart)
        @notify_observer(observer, "emulated_predicts2", [trial, t_elapsed]) 
    end

    predicts = get_predicts(problem, expr)
    #time metrics
    println("timing metrics")
    for trial = 1:n_trials
        tstart = CPUtime_start()
        for i = 1:n_reps
            #########################
            _, _, metric = gini_metrics(predicts, problem.Dl.labels)
            #########################
        end
        t_elapsed = CPUtime_elapsed_s(tstart)
        @notify_observer(observer, "metrics", [trial, t_elapsed]) 
    end

    save_log("timingresults.txt", logs)
    logs
end
