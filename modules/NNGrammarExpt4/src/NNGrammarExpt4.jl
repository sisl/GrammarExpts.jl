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
Experiment 4
test combined time series and binary logical operations
"""
module NNGrammarExpt4

export circuit4, restarts

import Compat.ASCIIString
using TFTools
using Datasets
using TensorFlow
using TensorFlow.CoreTypes #DT_FLOAT32, DT_BOOL, PyVectorType
using TensorFlow.API: l2_loss, AdamOptimizer, GradientDescentOptimizer, cast, 
    round_, reshape_, l2_normalize, RMSPropOptimizer,
    reduce_max, reduce_min, minimum_, maximum_, transpose_, less_, greater,
    expand_dims, tile, shape, mul, reduce_sum, moments, conv2d, truncated_normal,
    concat, sub_, matmul, clip_by_value
using TensorFlow.API.TfNn: sigmoid
using StatsBase
using RLESUtils, Confusion

using PyCall
@pyimport tensorflow as tf

function restarts(f::Function, N::Int64; kwargs...)
   [f(; kwargs...) for i = 1:N]
end

function circuit4(;
    dataname::AbstractString="vhdist3",
    labelfield::AbstractString="nmac", #"F_x1_lt_100_and_x2_lt_500",
    learning_rate::Float64=0.001,
    max_training_epochs::Int64=500,
    target_cost::Float64=0.001,
    batch_size::Int64=4820,
    mux_hidden_units::Vector{Int64}=Int64[50,30],
    display_step::Int64=1,
    b_debug::Bool=false,
    nshow::Int64=1)

    Dl = dataset(dataname, :nmac; transform=x->Float32(x)) #DFSetLabeled
    val_inputs = constant(Float32[0, 50, 100, 250, 500, 1000, 3000, 5000])
    data_set = TFDataset(Dl)

    # Construct model
    (n_examples, n_steps, n_feats) = size(Dl)
    n_featsflat = n_steps * n_feats 
    n_vals = get_shape(val_inputs)[1]

    # inputs
    feats = Placeholder(DT_FLOAT32, [-1, n_steps, n_feats])
    inputs = Tensor(feats)

    normalizer = Normalizer(data_set; converttype=Float32)
    normed_input = normalize01(normalizer, inputs)
    feats_flat = reshape_(normed_input, constant(Int32[-1, n_featsflat]))

    #softness parameter
    softness = collect(linspace(0.1, 1.0, max_training_epochs))
    softness_pl = Placeholder(DT_FLOAT32, [1])

    #toggle hard or soft output
    harden_pl = Placeholder(DT_BOOL, [1])

    # common (embedding) layer
    #embed_in = constant(rand(Float32, 1, 5))
    #embed_hidden_units = [5]
    #embed_blk = ReluStack(embed_in, embed_hidden_units)
    #embed_out = out(embed_blk)

    # mux select input
    #muxselect = feats_flat #simple flatten
    #muxselect = constant(ones(Float32, 1, 1)) #constant 1
    #muxselect = embed_out #relustack embedding

    overrides = zeros(Int64, 8)
    #overrides = [1, 2, 3, 5, 1, 1, 1, 1]
    #overrides = [0, 2, 0, 5, 0, 1, 1, 0]

    #convolution stuff
    HT = 3 
    WT = 2
    STRIDES = PyVectorType([1, 1, 1, 1])
    filt_weights = Variable(truncated_normal(constant([HT, WT, 1, 1]), constant(0.0), constant(5e-2)))
    conv1 = conv2d(expand_dims(Tensor(normed_input), constant(3)), Tensor(filt_weights), STRIDES, "SAME")
    muxselect = reshape_(conv1, constant(Int32[-1, n_featsflat]))

    # f1 feat select
    f1_in = inputs 
    f1_mux = SoftMux(n_feats, mux_hidden_units, f1_in, muxselect, Tensor(harden_pl), Tensor(softness_pl); override=overrides[1])
    f1_out = out(f1_mux) 

    # f2 feat select
    f2_in = inputs 
    f2_mux = SoftMux(n_feats, mux_hidden_units, f2_in, muxselect, Tensor(harden_pl), Tensor(softness_pl); override=overrides[2])
    f2_out = hardout(f2_mux) 

    # v1 value select
    #v1_in = val_inputs 
    n_vals=1
    v1_in = constant(Float32[100.0])
    v1_mux = SoftMux(n_vals, mux_hidden_units, v1_in, muxselect, Tensor(harden_pl); override=overrides[3])
    v1_out = out(v1_mux) 

    # v2 value select
    #v2_in = val_inputs 
    n_vals = 1
    v2_in = constant(Float32[500.0])
    v2_mux = SoftMux(n_vals, mux_hidden_units, v2_in, muxselect, Tensor(harden_pl); override=overrides[4])
    v2_out = hardout(v2_mux) 

    compare_ops = [op_lt, op_gt]
    logical_ops = [op_and, op_or]
    temporal_ops = [op_F, op_G]

    # a1 float op block
    a1_in = (f1_out, v1_out)
    a1_blk = SoftOpsMux(a1_in, compare_ops, mux_hidden_units, muxselect, Tensor(harden_pl), Tensor(softness_pl); opargs=Any[Tensor(softness_pl)], override=overrides[5])
    a1_out = out(a1_blk) 

    # a2 float op block
    a2_in = (f2_out, v2_out)
    a2_blk = SoftOpsMux(a2_in, compare_ops, mux_hidden_units, muxselect, Tensor(harden_pl), Tensor(softness_pl); opargs=Any[Tensor(softness_pl)], override=overrides[6])
    a2_out = hardout(a2_blk) 

    # l1 logical op block
    l1_in = (a1_out, a2_out)
    l1_blk = SoftOpsMux(l1_in, logical_ops, mux_hidden_units, muxselect, Tensor(harden_pl), Tensor(softness_pl); override=overrides[7])
    l1_out = hardout(l1_blk) 

    # t1 temporal op block
    t1_in = (l1_out,)
    #t1_in = (constant(rand(Float32, shape(l1_out))), )
    t1_blk = SoftOpsMux(t1_in, temporal_ops, mux_hidden_units, muxselect, Tensor(harden_pl), Tensor(softness_pl); override=overrides[8])
    t1_out = out(t1_blk) 

    # outputs
    pred = t1_out
    labels = Placeholder(DT_FLOAT32, [-1]) 
    
    #take nnout of batch, compute moments, take variance component and sum
    f1_var = reduce_sum(moments(f1_mux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    v1_var = reduce_sum(moments(v1_mux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    f2_var = reduce_sum(moments(f2_mux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    v2_var = reduce_sum(moments(v2_mux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    a1_var = reduce_sum(moments(a1_blk.softmux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    a2_var = reduce_sum(moments(a2_blk.softmux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    l1_var = reduce_sum(moments(l1_blk.softmux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    t1_var = reduce_sum(moments(t1_blk.softmux.nnout, Tensor(Int32[0]))[2], Tensor(0))
    sum_var = f1_var + v1_var + f2_var + v2_var + constant(300.0) .* a1_var + constant(300.0) .* a2_var + constant(300.0) .* l1_var + constant(300.0) .* t1_var

    #packed_nnouts = concat(Tensor(1), Tensor([f1_mux.nnout, v1_mux.nnout, f2_mux.nnout, v2_mux.nnout, a1_blk.softmux.nnout, a2_blk.softmux.nnout, l1_blk.softmux.nnout, t1_blk.softmux.nnout]))
    #m = matmul(packed_nnouts, packed_nnouts, false, true)
    #allpairs_cossim = reduce_sum(sub_(constant(8.0), m))

    # Define loss and optimizer
    #cost = l2_loss(pred - labels) # Squared loss
    cost = l2_loss(pred - labels) + constant(1.0) .* sum_var
    #cost = l2_loss(pred - labels) + constant(1.0) .* allpairs_cossim


    #optimizer
    #optimizer = minimize(AdamOptimizer(learning_rate), cost) 
    opt = Optimizer(tf.train[:GradientDescentOptimizer](learning_rate))
    gvs = opt.x[:compute_gradients](cost.x) 
    capped_gvs = [(tf.nn[:l2_normalize](tf.clip_by_value(grad, -1.0, 1.0), 0), var) for (grad, var) in gvs]
    optimizer = Operation(opt.x[:apply_gradients](capped_gvs))

    #compiled hardselect
    ckt = Circuit([
        f1_mux, v1_mux,
        f2_mux, v2_mux,
        a1_blk, a2_blk, 
        l1_blk, 
        t1_blk, 
        ], 
        Vector{ASCIIString}[
        ["alt", "range"],
        ["0", "50", "100", "250", "500", "1000", "3000", "5000"],
        ["alt", "range"],
        ["0", "50", "100", "250", "500", "1000", "3000", "5000"],
        ["<", ">"],
        ["<", ">"],
        ["and", "or"],
        ["F", "G"]])

    #debug
    f1_grad = gradient_tensor(cost, f1_mux.weight)
    v1_grad = gradient_tensor(cost, v1_mux.weight)
    f2_grad = gradient_tensor(cost, f2_mux.weight)
    v2_grad = gradient_tensor(cost, v2_mux.weight)
    a1_grad = gradient_tensor(cost, a1_blk.softmux.weight)
    a2_grad = gradient_tensor(cost, a2_blk.softmux.weight)
    l1_grad = gradient_tensor(cost, l1_blk.softmux.weight)
    t1_grad = gradient_tensor(cost, t1_blk.softmux.weight)
    #/debug
    
    # Iniuializing the variables
    init = initialize_all_variables()
    
    # Rock and roll
    println("Optimization Start: $(now())")
    sess = Session()
    run(sess, init)

    #debug
    #fd = FeedDict(feats => data_set.X, labels => data_set.Y)
    #run(sess, normed_input, fd)
    #/debug

    # Training cycle
    for epoch in 1:max_training_epochs
        avg_cost = 0.0
        total_batch = div(num_examples(data_set), batch_size)

        #Loop over all batches
        for i in 1:total_batch
            batch_xs, batch_ys = next_batch(data_set, batch_size)
            fd = FeedDict(feats => batch_xs, labels => batch_ys, softness_pl => [softness[epoch]], harden_pl => [false])
            # Fit training using batch data
            run(sess, optimizer, fd)

            #debug
            #softsel = softselect_by_example(sess, ckt, fd)
            #grads = run(sess, Tensor([f1_grad, v1_grad, f2_grad, v2_grad, a1_grad, a2_grad, l1_grad, t1_grad]), fd)
            #conv1_out = run(sess, conv1, fd)
            #@show run(sess, pred, fd)
            #@show run(sess, pred - labels, fd)
            #@show run(sess, muxselect, fd)
            #@show run(sess, f1_grad, fd)
            #@show run(sess, v1_grad, fd)
            #/debug

            # Compute average loss
            batch_average_cost = run(sess, cost, fd)
            avg_cost += batch_average_cost / (total_batch * batch_size)
        end
    
        # Display logs per epoch step
        if epoch % display_step == 0
            #softsel = softselect_by_example(sess, ckt, fd)
            #grads = run(sess, Tensor([f1_grad, v1_grad, f2_grad, v2_grad, a1_grad, a2_grad, l1_grad, t1_grad]), fd)
            #println(softsel)
            #println(capped_gvs)
            #grads = Tensor([grad for (grad, var) in capped_gvs])
            println("Epoch $(epoch)  cost=$(avg_cost)")
            #println("Norm=$(norm(grads))")
            if avg_cost < Float32(target_cost)
                break;
            end
        end
    end
    println("Optimization Finished: $(now())")
    
    # Test model
    correct_prediction = (round_(pred) == labels)
    # Calculate accuracy
    accuracy = mean(cast(correct_prediction, DT_FLOAT32))
    
    #reload data_set to recover original order (next_batch will internally shuffle)
    data_set = TFDataset(Dl)
    X = data_set.X
    Y = data_set.Y

    #soft metrics
    fd = FeedDict(feats => data_set.X, labels => data_set.Y, softness_pl => [softness[end]], harden_pl => [false])
    Ypred_soft = run(sess, round_(pred), fd)
    acc_soft = run(sess, accuracy, fd)
    stringout = simplestring(sess, ckt, fd; order=[8,1,5,2,7,3,6,4])
    top5 = topstrings(stringout, 5)
    softsel = softselect_by_example(sess, ckt, fd)
    grads = run(sess, Tensor([f1_grad, v1_grad, f2_grad, v2_grad, a1_grad, a2_grad, l1_grad, t1_grad]), fd)
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

    top5, acc_hard
end

op_F(x::Tensor) = reduce_max(x, Tensor(1))
op_G(x::Tensor) = reduce_min(x, Tensor(1))
op_and(x::Tensor, y::Tensor) = minimum_(x, y) #element-wise
op_or(x::Tensor, y::Tensor) = maximum_(x, y) #element-wise

#TODO: move these to a central location
function op_gt(x::Tensor, y::Tensor, W::Tensor=constant(1.0/100.0)) 
    tmp = expand_dims(y, Tensor(1))
    ytiled = tile(tmp, Tensor([1, get_shape(x)[2]]))
    result = sigmoid(mul(W, x - ytiled))
    result
end
function op_lt(x::Tensor, y::Tensor, W::Tensor=constant(1.0/100.0))  
    tmp = expand_dims(y, Tensor(1))
    ytiled = tile(tmp, Tensor([1, get_shape(x)[2]]))
    result = sigmoid(mul(W, ytiled - x))
end

end #module
