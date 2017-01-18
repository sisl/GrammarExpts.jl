using RLESUtils,RunUtils,IFTTTUtils

const COMP = gethostname()
const DATA = "098filt_10K"

template_gp(i) = 
"""
using GrammarExpts,ACASX_GP_Tree,RLESUtils,IFTTTUtils
config=configure(ACASX_GP_Tree,"normal","nvn_libcas098smallfilt_10K")
acasx_gp_tree(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare_Tree/ACASX_GP_Tree/seed$i"), config...)
sendifttt(;value1="gp_tree,$i", value2=$COMP, value3=$DATA)
"""

template_ge(i) =
"""
using GrammarExpts,ACASX_GE_Tree,RLESUtils,IFTTTUtils
config=configure(ACASX_GE_Tree,"normal","nvn_libcas098smallfilt_10K")
acasx_ge_tree(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare_Tree/ACASX_GE_Tree/seed$i"), config...)
sendifttt(;value1="ge_tree,$i", value2=$COMP, value3=$DATA)
"""

template_mc(i) =
"""
using GrammarExpts,ACASX_MC_Tree,RLESUtils,IFTTTUtils
config=configure(ACASX_MC_Tree,"normal","nvn_libcas098smallfilt_10K")
acasx_mc_tree(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare_Tree/ACASX_MC_Tree/seed$i"), config...)
sendifttt(;value1="mc_tree,$i", value2=$COMP, value3=$DATA)
"""

template_mcts(i) =
"""
using GrammarExpts,ACASX_MCTS_Tree,RLESUtils,IFTTTUtils
config=configure(ACASX_MCTS_Tree,"normal","nvn_libcas098smallfilt_10K")
acasx_mcts(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare_Tree/ACASX_MCTS_Tree/seed$i"), config...)
sendifttt(;value1="mcts_tree,$i", value2=$COMP, value3=$DATA)
"""

notifydone() = sendifttt(;value1="$COMP done")

A = JuliaSource[]
append!(A, [JuliaSource(template_ge(i)) for i=11:13])
append!(A, [JuliaSource(template_gp(i)) for i=11:13])
append!(A, [JuliaSource(template_mc(i)) for i=11:13])
append!(A, [JuliaSource(template_mcts(i)) for i=1:3])
append!(A, [JuliaSource(template_mcts(i)) for i=11:13])


