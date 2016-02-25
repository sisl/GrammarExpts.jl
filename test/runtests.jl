using GrammarExpts
using Base.Test
using RLESUtils: FileUtils

const MODULEDIR = joinpath(dirname(@__FILE__), "..", "modules")

pkgs = readdir(MODULEDIR)

for pkg in pkgs
  GrammarExpts.test(pkg)
end
