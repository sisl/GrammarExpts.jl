using RLESUtils,RunUtils,IFTTTUtils

const COMP = gethostname()
const DATA = "098_10K"

template_gp(i) = 
"""
using GrammarExpts,ACASX_GP,RLESUtils,IFTTTUtils
config=configure(ACASX_GP,"normal","nvn_libcas098small_10K")
acasx_gp(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_GP/seed$i"), config...)
sendifttt(;value1="gp,$i", value2="$COMP", value3="$DATA")
"""

template_ge(i) =
"""
using GrammarExpts,ACASX_GE,RLESUtils,IFTTTUtils
config=configure(ACASX_GE,"normal","nvn_libcas098small_10K")
acasx_ge(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_GE/seed$i"), config...)
sendifttt(;value1="ge,$i", value2="$COMP", value3="$DATA")
"""

template_mc(i) =
"""
using GrammarExpts,ACASX_MC,RLESUtils,IFTTTUtils
config=configure(ACASX_MC,"normal","nvn_libcas098small_10K")
acasx_mc1(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_MC/seed$i"), config...)
sendifttt(;value1="mc,$i", value2="$COMP", value3="$DATA")
"""

template_mcts(i) =
"""
using GrammarExpts,ACASX_MCTS,RLESUtils,IFTTTUtils
config=configure(ACASX_MCTS,"normal","nvn_libcas098small_10K")
acasx_mcts(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_MCTS/seed$i"), config...)
sendifttt(;value1="mcts,$i", value2="$COMP", value3="$DATA")
"""

notifydone() = sendifttt(;value1="$COMP done")

A = JuliaSource[]
append!(A, [JuliaSource(template_ge(i)) for i=1:20])
append!(A, [JuliaSource(template_gp(i)) for i=1:20])
append!(A, [JuliaSource(template_mc(i)) for i=1:20])
append!(A, [JuliaSource(template_mcts(i)) for i=1:20])


