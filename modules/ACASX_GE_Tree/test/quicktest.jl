using GrammarExpts, ACASX_GE_Tree
config=configure(ACASX_GE_Tree, "nvn_dasc", "normal")
config[:maxiterations] = 1
config[:pop_size]=50
acasx_ge_tree(; seed=1, outdir=joinpath(ACASX_GE_Tree.RESULTDIR, "./ACASX_GE_Tree_noflags1"), config...)
