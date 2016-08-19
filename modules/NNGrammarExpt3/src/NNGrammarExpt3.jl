#*****************************************************************************
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
Experiment 3
test combined time series and binary logical operations
"""
module NNGrammarExpt3

export circuit_fgandor, restarts

using TFTools
using Datasets
using TensorFlow
using TensorFlow.CoreTypes
import TensorFlow.API: l2_loss, AdamOptimizer, cast, round_, reshape_,
    reduce_max, reduce_min, pack, minimum_, maximum_, transpose_, reduce_sum,
    moments, gradients
using StatsBase
using RLESUtils, Confusion

using PyCall
@pyimport tensorflow as tf

function restarts(f::Function, N::Int64; kwargs...)
   [f(; kwargs...) for i = 1:N]
end

using Debug
@debug function circuit_fgandor(;
    dataname::AbstractString="bin_ts_synth",
    labelfield::AbstractString="F_x1_and_x3",
    learning_rate::Float64=0.001,
    max_training_epochs::Int64=500,
    target_cost::Float64=0.001,
    batch_size::Int64=5000,
    embed_hidden_units::Vector{Int64}=Int64[50, 30, 15],
    mux_hidden_units::Vector{Int64}=Int64[50,30],
    display_step::Int64=1,
    b_debug::Bool=false,
    nshow::Int64=20)

    Ds = dataset(dataname) #DFSet
    data_set = TFDataset(Ds, getmeta(Ds)[symbol(labelfield)])

    # Construct model
    (n_examples, n_steps, n_feats) = size(Ds)
    n_featsflat = n_steps * n_feats 

    # inputs
    feats = Placeholder(DT_FLOAT32, [-1, n_steps, n_feats])
    inputs = Tensor(feats)
    feats_flat = reshape_(feats, Tensor([-1, n_featsflat]))

    # common (embedding) layer
    #embed_in = feats_flat 
    #embed_blk = ReluStack(embed_in, embed_hidden_units)
    #embed_out = out(embed_blk)
    
    # mux select input
    muxselect = feats_flat #embed layer not helping, disable for now...

    #softness parameter
    softness = collect(linspace(0.1, 1.0, max_training_epochs))
    softness_pl = Placeholder(DT_FLOAT32, [1])

    #toggle hard or soft output
    harden_pl = Placeholder(DT_BOOL, [1])

    # x mux
    x_muxin = inputs 
    x_mux = SoftMux(n_feats, mux_hidden_units, x_muxin, muxselect, Tensor(harden_pl), Tensor(softness_pl))
    x_muxout = out(x_mux) 

    # y mux
    y_muxin = inputs 
    y_mux = SoftMux(n_feats, mux_hidden_units, y_muxin, muxselect, Tensor(harden_pl), Tensor(softness_pl))
    y_muxout = out(y_mux) 

    # logical op block
    ops1_in = (x_muxout, y_muxout)
    ops1_list = [ops1_And, ops1_Or]
    ops1_blk = SoftOpsMux(ops1_in, ops1_list, mux_hidden_units, muxselect, Tensor(harden_pl), Tensor(softness_pl))
    ops1_out = out(ops1_blk) 

    # temporal op block 
    ops2_in = (ops1_out,)
    ops2_list = [ops2_F, ops2_G]
    ops2_blk = SoftOpsMux(ops2_in, ops2_list, mux_hidden_units, muxselect, Tensor(harden_pl), Tensor(softness_pl))
    ops2_out = out(ops2_blk) 

    # outputs
    pred = ops2_out
    labels = Placeholder(DT_FLOAT32, [-1]) 

    #take nnout of batch, compute moments, take variance component and sum
    x_var = reduce_sum(moments(x_mux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    y_var = reduce_sum(moments(y_mux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    ops1_var = reduce_sum(moments(ops1_blk.softmux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    ops2_var = reduce_sum(moments(ops2_blk.softmux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    sum_var = x_var + y_var + constant(10.0) .* ops1_var + constant(10.0) .* ops2_var
    
    # Define loss and optimizer
    #cost = l2_loss(pred - labels) # Squared loss
    cost = l2_loss(pred - labels) + constant(50.0) .* sum_var

    #optimizer
    #optimizer = minimize(AdamOptimizer(learning_rate), cost) # Adam Optimizer
    opt = Optimizer(tf.train[:GradientDescentOptimizer](learning_rate))
    gvs = opt.x[:compute_gradients](cost.x) 
    capped_gvs = [(tf.nn[:l2_normalize](tf.clip_by_value(grad, -1.0, 1.0), 0), var) for (grad, var) in gvs]
    optimizer = Operation(opt.x[:apply_gradients](capped_gvs))

    #compiled hardselect
    ckt = Circuit([
        x_mux, y_mux,
        ops1_blk, 
        ops2_blk, 
        ], 
        Vector{ASCIIString}[
        recordcolnames(Ds),
        recordcolnames(Ds),
        ["and", "or"],
        ["F", "G"]])
    
    # Initializing the variables
    init = initialize_all_variables()
    
    # Rock and roll
    sess = Session()
    run(sess, init)

    # Training cycle
    for epoch in 1:max_training_epochs
        avg_cost = 0.0
        total_batch = div(num_examples(data_set), batch_size)
    
        # Loop over all batches
        for i in 1:total_batch
            batch_xs, batch_ys = next_batch(data_set, batch_size)
            fd = FeedDict(feats => batch_xs, labels => batch_ys, softness_pl => [softness[epoch]], harden_pl => [false])
            # Fit training using batch data
            run(sess, optimizer, fd)

            #debug
            #@show run(sess, pred - labels, fd)
            #/debug 
            
            # Compute average loss
            batch_average_cost = run(sess, cost, fd)
            avg_cost += batch_average_cost / (total_batch * batch_size)
        end
    
        # Display logs per epoch step
        if epoch % display_step == 0
            fd = FeedDict(feats => data_set.X[1,:,:], labels => [data_set.Y[1]], softness_pl => [softness[epoch]], harden_pl => [false])
            softsel = softselect_by_example(sess, ckt, fd)
            x_grad = run(sess, Tensor(gradients(cost, x_mux.nnout)), fd)
            y_grad = run(sess, Tensor(gradients(cost, y_mux.nnout)), fd)
            a1_grad = run(sess, Tensor(gradients(cost, ops1_blk.softmux.nnout)), fd)
            t1_grad = run(sess, Tensor(gradients(cost, ops2_blk.softmux.nnout)), fd)
            println("x_sel=", softsel[1][1])
            println("x_grad=",x_grad[1])
            println("y_sel=", softsel[1][2])
            println("y_grad=",y_grad[1])
            println("a1_sel=", softsel[1][3])
            println("a1_grad=",a1_grad[1])
            println("t1_sel=", softsel[1][4])
            println("t1_grad=",t1_grad[1])
            @bp
            println("Epoch $(epoch)  cost=$(avg_cost)")
            if avg_cost < target_cost
                break;
            end
        end
    end
    println("Optimization Finished")
    
    # Test model
    correct_prediction = (round_(pred) == labels)
    accuracy = mean(cast(correct_prediction, DT_FLOAT32))
    
    #reload data_set to recover original order
    data_set = TFDataset(Ds, getmeta(Ds)[symbol(labelfield)])
    X = data_set.X 
    Y = data_set.Y

    #soft metrics
    fd = FeedDict(feats => data_set.X, labels => data_set.Y, softness_pl => [softness[end]], harden_pl => [false])
    Ypred_soft = run(sess, round_(pred), fd)
    acc_soft = run(sess, accuracy, fd)
    stringout = simplestring(sess, ckt, fd; order=[4,1,3,2])
    top5 = topstrings(stringout, 5)
    softsel = softselect_by_example(sess, ckt, fd)
    #grads = run(sess, Tensor([f1_grad, v1_grad, a1_grad, t1_grad]), fd)
    conf = confusion(Ypred_soft.==1.0, Y.==1.0)
    conf_indices = confusion_indices(Ypred_soft.==1.0, Y.==1.0)

    #hard metrics
    fd = FeedDict(feats => data_set.X, labels => data_set.Y, softness_pl => [softness[end]], harden_pl => [true])
    Ypred_hard = run(sess, round_(pred), fd)
    acc_hard = run(sess, accuracy, fd)

    println("Soft Accuracy:", acc_soft)
    println("Hard Accuracy:", acc_hard)
    println(top5)
    d=Dict{ASCIIString,Any}()
    @bp 

    top5, acc_hard
end

function ops2_F(x::Tensor)
   reduce_max(x, Tensor(1))
end

function ops2_G(x::Tensor)
   reduce_min(x, Tensor(1))
end

function ops1_And(x::Tensor, y::Tensor)
    minimum_(x, y) #element-wise
end

function ops1_Or(x::Tensor, y::Tensor)
    maximum_(x, y) #element-wise
end

end #module
