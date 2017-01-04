using RLESUtils,RunUtils

const COMP = gethostname()
const DATA = "098filt"

template_gp(i) = 
"""
using GrammarExpts,ACASX_GP,RLESUtils,IFTTTUtils
config=configure(ACASX_GP,"normal","nvn_libcas098smallfilt")
acasx_gp(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_GP/seed$i"), config...)
sendifttt(;value1="gp,$i", value2=$COMP, value3=$DATA)
"""

template_ge(i) =
"""
using GrammarExpts,ACASX_GE,RLESUtils,IFTTTUtils
config=configure(ACASX_GE,"normal","nvn_libcas098smallfilt")
acasx_ge(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_GE/seed$i"), config...)
sendifttt(;value1="ge,$i", value2=$COMP, value3=$DATA)
"""

template_mc(i) =
"""
using GrammarExpts,ACASX_MC,RLESUtils,IFTTTUtils
config=configure(ACASX_MC,"normal","nvn_libcas098smallfilt")
acasx_mc(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_MC/seed$i"), config...)
sendifttt(;value1="mc,$i", value2=$COMP, value3=$DATA)
"""

template_mcts(i) =
"""
using GrammarExpts,ACASX_MCTS,RLESUtils,IFTTTUtils
config=configure(ACASX_MCTS,"normal","nvn_libcas098smallfilt")
acasx_mcts(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_MCTS/seed$i"), config...)
sendifttt(;value1="mcts,$i", value2=$COMP, value3=$DATA)
"""

A = JuliaSource[]
append!(A, [JuliaSource(template_ge(i)) for i=12:15])
append!(A, [JuliaSource(template_gp(i)) for i=11:15])
append!(A, [JuliaSource(template_mc(i)) for i=11:15])
append!(A, [JuliaSource(template_mcts(i)) for i=11:15])


