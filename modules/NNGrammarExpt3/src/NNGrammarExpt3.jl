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
import TensorFlow: DT_FLOAT32
import TensorFlow.API: l2_loss, AdamOptimizer, cast, round_, reshape_,
    reduce_max, reduce_min, pack, minimum_, maximum_, transpose_
using StatsBase

function restarts(f::Function, N::Int64; kwargs...)
   [f(; kwargs...) for i = 1:N]
end

function circuit_fgandor(;
    dataname::AbstractString="bin_ts_synth",
    labelfield::AbstractString="F_x1_and_x3",
    learning_rate::Float64=0.002,
    max_training_epochs::Int64=400,
    target_cost::Float64=0.001,
    batch_size::Int64=1000,
    embed_hidden_units::Vector{Int64}=Int64[50, 30, 15],
    mux_hidden_units::Vector{Int64}=Int64[30,15],
    display_step::Int64=10,
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
    embed_in = feats_flat 
    embed_blk = ReluStack(embed_in, embed_hidden_units)
    embed_out = out(embed_blk)
    
    # mux select input
    muxselect = feats_flat #embed layer not helping, disable for now...

    # x mux
    x_muxin = inputs 
    x_mux = SoftMux(n_feats, mux_hidden_units, x_muxin, muxselect)
    x_muxout = out(x_mux) 

    # y mux
    y_muxin = inputs 
    y_mux = SoftMux(n_feats, mux_hidden_units, y_muxin, muxselect)
    y_muxout = out(y_mux) 

    # logical op block
    ops1_in = (x_muxout, y_muxout)
    ops1_list = [ops1_And, ops1_Or]
    ops1_blk = OpsBlock(ops1_in, ops1_list)
    ops1_out = out(ops1_blk) 

    # logical op mux
    op1_muxin = ops1_out 
    op1_mux = SoftMux(num_ops(ops1_blk), mux_hidden_units, op1_muxin, muxselect)
    op1_muxout = out(op1_mux) 

    # temporal op block 
    ops2_in = (op1_muxout,)
    ops2_list = [ops2_F, ops2_G]
    ops2_blk = OpsBlock(ops2_in, ops2_list)
    ops2_out = out(ops2_blk) 

    # temporal op mux
    op2_muxin = ops2_out
    op2_mux = SoftMux(num_ops(ops2_blk), mux_hidden_units, op2_muxin, muxselect)
    op2_muxout = out(op2_mux) 

    # outputs
    pred = op2_muxout
    labels = Placeholder(DT_FLOAT32, [-1]) 
    
    # Define loss and optimizer
    cost = l2_loss(pred - labels) # Squared loss
    optimizer = minimize(AdamOptimizer(learning_rate), cost) # Adam Optimizer

    #compiled hardselect
    hardselects = transpose_(pack(Tensor([hardselect(x_mux), hardselect(y_mux), hardselect(op1_mux), hardselect(op2_mux)])))
    
    # Initializing the variables
    init = initialize_all_variables()
    
    # Rock and roll
    sess = Session()
    run(sess, init)

    #debug 
    #fd = FeedDict(feats => data_set.X, labels => data_set.Y)
    #@bp
    #tmp=1
    #run(sess, x_mux.nnout, fd)
    #run(sess, x_mux.hardselect, fd)
    #run(sess, x_muxout, fd)
    #run(sess, ops2_out, fd)
    #run(sess, op1_muxout, fd)
    #run(sess, cost, fd)
    #/debug
    
    # Training cycle
    for epoch in 1:max_training_epochs
        avg_cost = 0.0
        total_batch = div(num_examples(data_set), batch_size)
    
        # Loop over all batches
        for i in 1:total_batch
            batch_xs, batch_ys = next_batch(data_set, batch_size)
            fd = FeedDict(feats => batch_xs, labels => batch_ys)
            # Fit training using batch data
            run(sess, optimizer, fd)
            # Compute average loss
            batch_average_cost = run(sess, cost, fd)
            avg_cost += batch_average_cost / (total_batch * batch_size)
        end
    
        # Display logs per epoch step
        if epoch % display_step == 0
            println("Epoch $(epoch)  cost=$(avg_cost)")
            if avg_cost < target_cost
                break;
            end
        end
    end
    println("Optimization Finished")
    
    # Test model
    correct_prediction = (round_(pred) == labels)
    # Calculate accuracy
    accuracy = mean(cast(correct_prediction, DT_FLOAT32))
    fd = FeedDict(feats => data_set.X, labels => data_set.Y)
    acc = run(sess, accuracy, fd)
    
    #reload data_set to recover original order
    data_set = TFDataset(Ds, Dlabels[symbol(labelfield)])
    fd = FeedDict(feats => data_set.X, labels => data_set.Y)
    db_x = data_set.X 
    db_labels = data_set.Y
    db_hardselects = run(sess, hardselects, fd)
    xnames = colnames(Ds)
    op1names = ["&", "|"]
    op2names = ["F", "G"]
    hs = db_hardselects
    stringout = ["$(op2names[hs[i,4]+1])($(xnames[hs[i,1]+1]) $(op1names[hs[i,3]+1]) $(xnames[hs[i,2]+1]))" for i = 1:n_examples]
    cmap = countmap(stringout)
    cmap1 = collect(cmap)
    maxentry = sort(collect(cmap1), by=kv->kv[2], rev=true)[1:5]
    maxentry = map(x->(x...), maxentry)

    if b_debug
        @show db_hardselects[1:nshow,:] #n_examples by hardselects
    end

    @show cmap
    println(maxentry)
    println("Accuracy:", acc)

    cmap, maxentry, acc
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
