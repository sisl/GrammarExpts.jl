# GrammarExpts

Author: Ritchie Lee, Carnegie Mellon University Silicon Valley, ritchie.lee@sv.cmu.edu

In the face of big data, gaining insights by manually sifting through data is no longer practical.  Machine learning methods typically rely on statistical models.  Although these may provide good input/output behavior, the results are not conducive to human understanding.  We explore machine learning tasks guided by problem-specific grammar that a user provides.  We learn expressions derived from the provided grammar, making the results intuitive and interpretable to a human.

## Overview

GrammarExpts is a collection of modules for experimenting with grammar-guided expression discovery on various problems.

## Problems

A problem module defines various specifics for a given grammar optimization problem, including:

* Grammar - From which expressions should be derived. Defines domain of search space.  The semantics of the grammar can be defined arbitrarily by the user.  For example, subsets of temporal logic can be used in time-series analysis.
* Fitness function - A function that maps an expression to a real number indicating the quality of the expression.  Lower is better.  For example, for a classification task, this may be misclassification rate.

The following problems are currently implemented:

* ACAS X ("ACASXProblem") - Time-series classification task for encounter data from RLESCAS. Learn an expression for the decision boundary to separate NMACs vs. non-NMACs or discover rules to explain clusterings.
* Symbolic regression ("SymbolicProblem") - Try to reconstruct/rediscover the symbolic form of a mathematical expression from evaluation data only.

## Installation

Julia 0.4 is required.

* Pkg.clone("https://github.com/rcnlee/GrammarExpts.jl.git", "GrammarExpts")
* Pkg.build("GrammarExpts") to automatically install dependencies
* Optional, Pkg.test("GrammarExpts") to test all the modules

### Main Package Dependencies

* ExprSearch.jl - Grammar-guided expression search algorithms
* RLESUtils.jl - Misc tools and utils
* Datasets.jl - Dataset manager
* RLESCAS.jl - Adaptive stress testing for collision avoidance systems.  The conversion scripts are required for processing RLESCAS json files into the DataFrames format used by GrammarExpts.
* TikzQTrees.jl - Plot trees to PDF using the TikzQTrees latex package

### Useful Locations to Know

* PKGDIR/GrammarExpts/modules - Contains all the submodules
* PKGDIR/GrammarExpts/data - Contains input and intermediate-processed data
* PKGDIR/Datasets/data - Contains the input data

#### ACAS X

* PKGDIR/Datsets/data/dasc - Encounter data for DASC dataset
* PKGDIR/Datsets/data/dascfilt - Encounter data for DASC dataset filtered starting 5 seconds before CPA
* PKGDIR/Datasets/data/dasc\_meta - Meta info for DASC dataset. i.e., NMAC labels
* PKGDIR/Datasets/data/dasc\_manuals - SME clustering results for the DASC dataset
* PKGDIR/Datsets/data/libcas098small - Encounter data for Libcas 0.9.8 Small Test dataset
* PKGDIR/Datsets/data/libcas098smallfilt - Encounter data for Libcas 0.9.8 Small Test dataset filtered starting 5 seconds before CPA
* PKGDIR/Datasets/data/libcas098small\_meta - Meta info for libcas098small dataset. i.e., NMAC labels

## Usage

In general, first call ``using GrammarExpts`` to make all the submodules globally visible.

### ACAS X Usage

#### Learn an Expression for ACAS X

```julia
cd(Pkg.dir("GrammarExpts/results"))
addprocs(4)
using GrammarExpts, ACASX\_MC #Monte Carlo
config = configure(ACASX\_MC, "normal", "nvn\_dasc") #load a config. Here, combine two configs
acasx\_mc(; config...) #run. By default will output to current directory
```

```julia
cd(Pkg.dir("GrammarExpts/results"))
addprocs(4)
using GrammarExpts, ACASX\_SA #Simulated Annealing
config = configure(ACASX\_SA, "normal", "nvn\_dasc") #load a config. Here, combine two configs
acasx\_sa(; config...) #run. By default will output to current directory
```

#### Learn a Decision Tree for ACAS X

```julia
cd(Pkg.dir("GrammarExpts/results"))
addprocs(4)
using GrammarExpts, ACASX\_MC\_Tree #Monte Carlo
config = configure(ACASX\_MC, "normal", "nvn\_dasc") #load a config. Here, combine two configs
acasx\_mc\_tree(; config...) #run. By default will output to current directory
```

```julia
cd(Pkg.dir("GrammarExpts/results"))
addprocs(4)
using GrammarExpts, ACASX\_SA\_Tree #Simulated Annealing
config = configure(ACASX\_SA, "normal", "nvn\_dasc") #load a config. Here, combine two configs
acasx\_sa\_tree(; config...) #run. By default will output to current directory
```

#### Data

This is likely not needed for normal operation, but is good info to know.

GrammarExpts uses Datasets.jl to manage its data.  The data is stored one file per encounter in subfolders of ``PKGDIR/Datasets/data``.  To load an entire dataset (collection of encounters), use

```julia
using Datasets
data = dataset("dasc") #dataset name is also folder name
```
which will load a DFSet object (collection of DataFrames)

To load a specific encounter file, use

```julia
using Datasets
D = dataset("dasc", "1") #load dasc dataset encounter 1 into a DataFrame
```

#### Configuration and Configuration Files

Many of the main entry points such as ``ACASX\_MC\_Tree`` use keyword arguments to set configuration parameters.

For convenience, some modules implement a configuration feature.  Config files are stored in a subfolder in the corresponding module.  For example, ``PKGDIR/GrammarExpts/ACASX_MC_Tree/config``

Each config file is a julia file containing a vector of Symbol/Any pairs, for example:
```julia
[
  #tree
  (:maxsteps, 20),

  #MC
  (:n_samples, 125000),
  (:n_threads, 4)
]
```
that specifies parameter/value pairs.  To load the configuration file, use: ``config = configure(ACASX_MC_Tree, filename)``

Sometimes it is convenient to split up configurations into multiple pieces, for example one file for search and another for data.  To load mutiple config files into the same config dict, use: ``config = configure(ACASX_MC_Tree, "normal", "nvn_dasc")`` which loads "normal.jl" and "nvn\_dasc.jl"

At this point you can inspect the parameters in the config object or even overwrite it:
``config[:param] = newvalue``

The configuration can be splatted into the keyword arguments of the function call like:
``acasx_mc_tree(; config...)``

Values not specified by the configuration take on the defaults specified in the function definition.

### Output

Decision tree and visualization:

"acasx\_mc\_tree\_log\_vis\_decisiontree.json" - decision tree json output
"acasx\_mc\_tree\_log\_vis\_decisiontree.pdf" - visualization of the output decision tree. Disable creation of pdf file by setting ``plotpdf=false`` if available in the keyword arguments. The PDF can be later produced from the .json file by calling:

```julia
using TikzQTrees
plottree("acasx_mc_tree_log_vis_decisiontree.json", outfileroot="acasx_mc_tree_log_vis_decisiontree")
```

Output logs are in TaggedDFLogger format, which is a light wrapper around DataFrames.  To load the data into a TaggedDFLogger object, call

```julia
using RLESUtils.Loggers
logs = load_log("acasx_mc_tree_log.txt") #recursively loads the .csv.gz files
keys(logs) #see available logs
logs["parameters"] #Parameters log as a dataframe
```

Alternatively, you can open specific logs as dataframes directly

```julia
using DataFrames
D = readtable("acasx_mc_tree_log_result.csv.gz")
```

You may have noticed that DataFrames just stores its data in CSV format, so another way to access the data is to decompress the .csv.gz file into a .csv file and view it.  This is especially useful when just taking a quick look at a particular log.

### Data Processing

#### Process a new dataset

To extract features from RLESCAS json output to DataFrames format that GrammarExpts uses, use the CASJson2DataFrames module.  In particular, take a look at ``tsfeats1_scripts.jl``.  Entries exist for "dasc" and "libcas098small" datasets, so these can be regenerated from jsons by placing the json files under ``PKGDIR/GrammarExpts/data/dasc/json`` and calling

```julia
using GrammarExpts, CASJson2DataFrames
script_dasc()
```

If the CSVs already exist, then generating them can be skipped by calling ``script_dasc(false)`` instead.

To add a new dataset, one can follow the same format as the DASC dataset by creating a folder and placing the jsons under ``PKGDIR/GrammarExpts/data/[newdataset]/json``, then creating the corresponding entries in ``tsfeats1_scripts.jl`` similar to how the DASC set is done.

Alternatively, ``script_base()`` can be called supplying the paths directly without modifying ``tsfeats1_scripts.jl``.

#### Generate a filtered dataset

A "filtered" dataset is one where occurrences of NMACs are removed from the data, so that the search provides more interesting insights than just rediscovering the NMAC rule.  Encounters are truncated removing 5 seconds prior to CPA to the end of the encounter.  Encounters that have CPAs occur prior to 35 seconds have their truncation points randomly selected between 35 and 50 seconds.  (This is done to prevent the algorithm from keying off the length of the encounter for classification). Use the ``FilterNMACInfo`` module to generate a filtered dataset from an unfiltered one.

Scripts for DASC and libcas098small datasets already exist, so to regenerate these, you can call

```julia
using GrammarExpts, FilterNMACInfo
script_dasc()
```
The naming convention that is used is to append filt to the dataset name, e.g., "dasc" -> "dascfilt".

Alternatively, ``remove_cpa()`` can be called supplying the paths directly without modifying ``FilterNMACInfo.jl``
