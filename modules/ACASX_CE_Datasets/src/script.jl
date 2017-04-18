using RLESUtils, RunUtils, IFTTTUtils

const COMP = gethostname()
const CONFIG = "normal"
const DATA = ["dasc", "dascfilt"]#, "libcas098small_10K", "libcas098smallfilt_10K"]
const I_START = 1 
const I_END = 20 
const NOTIFY = false

template_ce(seed, data) =
"""
using GrammarExpts,ACASX_CE,RLESUtils,IFTTTUtils
config=configure(ACASX_CE,"$CONFIG")
config[:data]=$data
acasx_ce(;seed=$i, outdir=Pkg.dir("GrammarExpts/results/ACASX_CE_Data/data$data_seed$i"), config...)
$NOTIFY && sendifttt(;value1="ce,$i", value2="$COMP", value3="$DATA")
"""

notifydone() = sendifttt(;value1="$COMP done")

A = JuliaSource[]
append!(A, [JuliaSource(template_ce(i, d)) for i=I_START:I_END, d in DATA])

#include("script.jl")
#fs=julia_process(A, np); notifydone()
