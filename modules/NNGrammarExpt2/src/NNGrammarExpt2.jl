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
Experiment 2
test time series possibilities
"""
module NNGrammarExpt2

export circuit_fg

using TFTools
using Datasets
using TensorFlow
import TensorFlow: DT_FLOAT32
import TensorFlow.API: l2_loss, AdamOptimizer, cast, round_, reshape_,
    reduce_max, reduce_min
using StatsBase

function circuit_fg(;
    featsname::AbstractString="bin_ts_synth_feats",
    labelsname::AbstractString="bin_ts_synth_labels",
    labelfile::AbstractString="labels",
    labelfield::AbstractString="F_x1",
    learning_rate::Float64=0.002,
    max_training_epochs::Int64=1000,
    target_cost::Float64=0.001,
    batch_size::Int64=1000,
    hidden_units::Vector{Int64}=Int64[30,10],
    display_step::Int64=1,
    b_debug::Bool=false)

    Dfeats = dataset(featsname) #DFSet
    Dlabels = dataset(labelsname, labelfile)
    @assert length(Dfeats) == nrow(Dlabels) #sanity check, num examples should be same

    data_set = TFDataset(Dfeats, Dlabels[Symbol(labelfield)])

    # Construct model
    (n_examples, n_steps, n_feats) = size(Dfeats)
    n_select = n_steps * n_feats 

    # inputs
    feats = Placeholder(DT_FLOAT32, [-1, n_steps, n_feats])
    muxselect = reshape_(feats, Tensor([-1, n_select]))
    inputs = Tensor(feats)

    # x mux
    x_muxin = inputs 
    x_mux = SoftMux(n_feats, n_select, hidden_units, x_muxin, muxselect)
    x_muxout = out(x_mux) 

    # op block
    ops_in = (x_muxout,)
    ops_list = [ops_F, ops_G]
    ops_blk = OpsBlock(ops_in, ops_list)
    ops_out = out(ops_blk) 

    # op mux
    op_muxin = ops_out 
    op_mux = SoftMux(num_ops(ops_blk), n_select, hidden_units, op_muxin, muxselect)
    op_muxout = out(op_mux) 

    # outputs
    pred = op_muxout
    labels = Placeholder(DT_FLOAT32, [-1]) 
    
    # Define loss and optimizer
    cost = l2_loss(pred - labels) # Squared loss
    optimizer = minimize(AdamOptimizer(learning_rate), cost) # Adam Optimizer
    
    # Initializing the variables
    init = initialize_all_variables()
    
    # Rock and roll
    sess = Session()
    #try
        run(sess, init)

        #debug 
        #fd = FeedDict(feats => data_set.X, labels => data_set.Y)
        #@bp
        #tmp=1
        #run(sess, x_mux.nnout, fd)
        #run(sess, x_mux.hardselect, fd)
        #run(sess, x_muxout, fd)
        #run(sess, ops_out, fd)
        #run(sess, op_muxout, fd)
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
        println("Accuracy:", acc)

        if b_debug
            #reload data_set to recover original order
            data_set = TFDataset(Dfeats, Dlabels[Symbol(labelfield)])
            db_x = data_set.X 
            db_labels = data_set.Y
            db_xmux = run(sess, x_muxout, fd)
            db_xmux_nnout = run(sess, x_mux.nnout, fd)
            db_xmux_hardselect = run(sess, x_mux.hardselect, fd)
            db_opmux = run(sess, op_muxout, fd)
            db_opmux_nnout = run(sess, op_mux.nnout, fd)
            db_opmux_hardselect = run(sess, op_mux.hardselect, fd)
            db_pred = run(sess, pred, fd)
            #combine these into a single call to prevent multiple forward passes
            #run(sess, [x_muxout, ops_out, op_muxout, cost], fd)

            NSHOW = 20
            @show db_x[1:NSHOW]
            @show db_xmux[1:NSHOW]
            @show db_xmux_nnout[1:NSHOW,1:7]
            @show db_opmux[1:NSHOW]
            @show db_opmux_nnout[1:NSHOW,1:2]
            @show db_pred[1:NSHOW]
            @show db_labels[1:NSHOW]
            @show db_xmux_hardselect[1:NSHOW]
            @show db_opmux_hardselect[1:NSHOW]
            xnames = colnames(getrecords(Dfeats)[1])
            opnames = ["F", "G"]
            x = map(i -> xnames[i+1], db_xmux_hardselect)
            op = map(i -> opnames[i+1], db_opmux_hardselect)
            stringout = ["$(op[i]),$(x[i])" for i = 1:n_examples]
            @show countmap(stringout)
            println("Accuracy:", acc)
        end
    #finally
        #close(sess)
    #end
end

function ops_F(x::Tensor)
   reduce_max(x, Tensor(1))
end

function ops_G(x::Tensor)
   reduce_min(x, Tensor(1))
end

end #module
