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

module NNGrammar

export circuit_andor

using TFTools
using Datasets
using TensorFlow
import TensorFlow: DT_FLOAT32
import TensorFlow.API: l2_loss, AdamOptimizer, cast, round_, maximum_, minimum_

# Parameters
const LEARNING_RATE = 0.005
const TRAINING_EPOCHS = 500
const BATCH_SIZE = 1000
const DISPLAY_STEP = 1

# Network parameters
const HIDDEN_UNITS = [10, 10, 10] 

using Debug
@debug function circuit_andor(datname::AbstractString="bin_synth",
    featfile::AbstractString="feats", labelfile::AbstractString="labels";
    labelfield::AbstractString="x1_and_x3",
    b_debug::Bool=false)

    Dfeats = dataset(datname, featfile)
    Dlabels = dataset(datname, labelfile)
    n_feats = ncol(Dfeats)
    @assert nrow(Dfeats) == nrow(Dlabels)

    data_set = TFDataset(Dfeats, Dlabels[symbol(labelfield)])

    # Construct model

    # inputs
    muxselect = Placeholder(DT_FLOAT32, [-1, n_feats])
    inputs = Placeholder(DT_FLOAT32, [-1, n_feats])

    # x mux
    x_muxin = inputs 
    x_mux = Softmux(n_feats, n_feats, HIDDEN_UNITS, Tensor(x_muxin), Tensor(muxselect))
    x_muxout = out(x_mux) 

    # y mux
    y_muxin = inputs 
    y_mux = Softmux(n_feats, n_feats, HIDDEN_UNITS, Tensor(y_muxin), Tensor(muxselect))
    y_muxout = out(y_mux) 

    # op block
    ops_in = (x_muxout, y_muxout)
    ops_list = [ops_And, ops_Or]
    ops_blk = OpsBlock(ops_in, ops_list)
    ops_out = out(ops_blk) 

    # op mux
    op_muxin = ops_out 
    op_mux = Softmux(num_ops(ops_blk), n_feats, HIDDEN_UNITS, Tensor(op_muxin), Tensor(muxselect))
    op_muxout = out(op_mux) 

    # outputs
    pred = op_muxout
    labels = Placeholder(DT_FLOAT32, [-1]) 
    
    # Define loss and optimizer
    cost = l2_loss(pred - labels) # Squared loss
    optimizer = minimize(AdamOptimizer(LEARNING_RATE), cost) # Adam Optimizer
    
    # Initializing the variables
    init = initialize_all_variables()
    
    # Rock and roll
    sess = Session()
    try
        run(sess, init)

        #debug 
        #fd = FeedDict(inputs => data_set.X, muxselect => data_set.X, labels => data_set.Y)
        #@bp
        #tmp=1
        #run(sess, x_muxout, fd)
        #run(sess, y_muxout, fd)
        #run(sess, ops_out, fd)
        #run(sess, op_muxout, fd)
        #run(sess, cost, fd)
        #/debug
        
        # Training cycle
        for epoch in 1:TRAINING_EPOCHS
            avg_cost = 0.0
            total_batch = div(num_examples(data_set), BATCH_SIZE)
        
            # Loop over all batches
            for i in 1:total_batch
                batch_xs, batch_ys = next_batch(data_set, BATCH_SIZE)
                fd = FeedDict(inputs => batch_xs, muxselect => batch_xs, labels => batch_ys)
                # Fit training using batch data
                run(sess, optimizer, fd)
                # Compute average loss
                batch_average_cost = run(sess, cost, fd)
                avg_cost += batch_average_cost / (total_batch * BATCH_SIZE)
            end
        
            # Display logs per epoch step
            if epoch % DISPLAY_STEP == 0
                println("Epoch $(epoch)  cost=$(avg_cost)")
            end
        end
        println("Optimization Finished")
        
        # Test model
        correct_prediction = (round_(pred) == labels)
        # Calculate accuracy
        accuracy = mean(cast(correct_prediction, DT_FLOAT32))
        fd = FeedDict(inputs => data_set.X, muxselect => data_set.X, labels => data_set.Y)
        acc = run(sess, accuracy, fd)
        println("Accuracy:", acc)

        if b_debug
            db_x = data_set.X 
            db_xmux = run(sess, x_muxout, fd)
            db_xmux_nnout = run(sess, x_mux.nnout, fd)
            db_xmux_hardselect = run(sess, x_mux.hardselect, fd)
            db_ymux = run(sess, y_muxout, fd)
            db_ymux_nnout = run(sess, y_mux.nnout, fd)
            db_ymux_hardselect = run(sess, y_mux.hardselect, fd)
            db_opmux = run(sess, op_muxout, fd)
            db_opmux_nnout = run(sess, op_mux.nnout, fd)
            db_opmux_hardselect = run(sess, op_mux.hardselect, fd)
            db_ypred = run(sess, pred, fd)
            db_labels = data_set.Y

            NSHOW = 20
            @show db_x[1:NSHOW]
            @show db_xmux[1:NSHOW]
            @show db_xmux_nnout[1:NSHOW,1:7]
            @show db_ymux[1:NSHOW]
            @show db_ymux_nnout[1:NSHOW,1:7]
            @show db_ymux[1:NSHOW]
            @show db_opmux[1:NSHOW]
            @show db_opmux_nnout[1:NSHOW,1:2]
            @show db_xmux_hardselect[1:NSHOW]
            @show db_ymux_hardselect[1:NSHOW]
            @show db_opmux_hardselect[1:NSHOW]
            @show db_ypred[1:NSHOW]
            @show db_labels[1:NSHOW]
            println("Accuracy:", acc)
        end
    finally
        close(sess)
    end
end

function ops_And(x::Tensor, y::Tensor)
    minimum_(x, y) #alternatively, use tf.logical_and?
end

function ops_Or(x::Tensor, y::Tensor)
    maximum_(x, y) #alternatively, use tf.logical_or?
end

end #module
