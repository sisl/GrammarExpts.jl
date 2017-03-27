using RLESUtils, RunUtils, IFTTTUtils

const COMP = gethostname()
const DATA = "nvn_dascfilt"
const CONFIG = "normal"
const I_START = 1 
const I_END = 20 

template_gp(i) =
"""
using GrammarExpts,ACASX_GP,RLESUtils,IFTTTUtils
config=configure(ACASX_GP,"$CONFIG","$DATA")
acasx_gp(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_GP/seed$i"), config...)
sendifttt(;value1="gp,$i", value2="$COMP", value3="$DATA")
"""

template_ge(i) =
"""
using GrammarExpts,ACASX_GE,RLESUtils,IFTTTUtils
config=configure(ACASX_GE,"$CONFIG","$DATA")
acasx_ge(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_GE/seed$i"), config...)
sendifttt(;value1="ge,$i", value2="$COMP", value3="$DATA")
"""

template_mc(i) =
"""
using GrammarExpts,ACASX_MC,RLESUtils,IFTTTUtils
config=configure(ACASX_MC,"$CONFIG","$DATA")
acasx_mc1(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_MC/seed$i"), config...)
sendifttt(;value1="mc,$i", value2="$COMP", value3="$DATA")
"""

template_mcts(i) =
"""
using GrammarExpts,ACASX_MCTS,RLESUtils,IFTTTUtils
config=configure(ACASX_MCTS,"$CONFIG","$DATA")
acasx_mcts(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_MCTS/seed$i"), config...)
sendifttt(;value1="mcts,$i", value2="$COMP", value3="$DATA")
"""

template_ce(i) =
"""
using GrammarExpts,ACASX_CE,RLESUtils,IFTTTUtils
config=configure(ACASX_CE,"$CONFIG","$DATA")
acasx_ce(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_Compare/ACASX_CE/seed$i"), config...)
sendifttt(;value1="ce,$i", value2="$COMP", value3="$DATA")
"""

notifydone() = sendifttt(;value1="$COMP done")

A = JuliaSource[]
append!(A, [JuliaSource(template_ce(i)) for i=I_START:I_END])
append!(A, [JuliaSource(template_ge(i)) for i=I_START:I_END])
append!(A, [JuliaSource(template_gp(i)) for i=I_START:I_END])
append!(A, [JuliaSource(template_mc(i)) for i=I_START:I_END])
append!(A, [JuliaSource(template_mcts(i)) for i=I_START:I_END])

#addprocs(4)
#include("script.jl")
#fs=pmap(julia_process, A); notifydone()
