# GrammarExpts

Author: Ritchie Lee, Carnegie Mellon University Silicon Valley, ritchie.lee@sv.cmu.edu

In the face of big data, gaining insights by manually sifting through data is no longer practical.  Machine learning methods typically rely on statistical models.  Although these may provide good input/output behavior, the results are not conducive to human understanding.  We explore machine learning tasks guided by problem-specific grammar that a user provides.  We learn expressions derived from the provided grammar, making the results intuitive and interpretable to a human.

## Overview

GrammarExpts is a collection of modules for experimenting with grammar-guided expression discovery on various problems.

## Quick Start

Julia v0.5, 64-bit is required.

* Pkg.clone("https://github.com/sisl/GrammarExpts.jl", "GrammarExpts")
* Pkg.build("GrammarExpts") to automatically install dependencies and generate the processed dasc set
* Recommended, perform the basic tests in the next section

### Initial Tests

Initial tests can be useful in detecting install/dependency problems.

* Pkg.test("RLESUtils") to test all the submodules
* Pkg.test("ExprSearch") to test all the submodules
* Pkg.test("GrammarExpts") to test all the submodules

Optional, more in-depth test that includes data processing.

```julia
using GrammarExpts, PipelineTest
ptest = pipelinetest() #produces data under GrammarExpts/test/PipelineTest
#inspect dataset under Dataset/data/exampledata, exampledata_meta, and exampledatafilt
cleanup(ptest) #remove created artifacts, except results
```

### Usage

In general, first call ``using GrammarExpts`` to make all the submodules globally visible, then call the submodule you want.

### Learn a single expression for ACASXProblem

```julia
cd(Pkg.dir("GrammarExpts/results")) #output directory 
using GrammarExpts, ACASX_CE #Cross-Entropy method, other algorithms available
config = configure(ACASX_CE, "normal", "nvn_dasc") #load a configs. 
acasx_ce(; config...) #run. Outputs to results directory 
```

### Learn a GBDT for ACASXProblem

```julia
cd(Pkg.dir("GrammarExpts/results"))
using GrammarExpts, ACASX_CE_Tree #Cross entropy method, other algorithms available
config = configure(ACASX_CE_Tree, "normal", "nvn_dasc") #load configs. 
acasx_ce_tree(; config...) #run. By default will output to current directory
```

### Preparing a new dataset 



## Package Details
### Main Package Dependencies

These are automatically fetched by the build script:

* ExprSearch.jl - Grammar-guided expression search algorithms
* RLESUtils.jl - Misc tools and utils
* Datasets.jl - Dataset manager
* RLESCAS.jl - Adaptive stress testing for collision avoidance systems.  The conversion scripts are required for processing RLESCAS json files into the DataFrames format used by GrammarExpts.

### Useful Locations to Know

* PKGDIR/GrammarExpts/results - Default output location for results 
* PKGDIR/GrammarExpts/modules - Contains all the submodules
* PKGDIR/GrammarExpts/data - Contains input and intermediate data
* PKGDIR/Datasets/data - Contains the processed input data

### Expression Search Problems

A expression search problem is defined in a problem module that includes: 

* Grammar - From which expressions should be derived. Defines domain of search space.  The semantics of the grammar can be defined arbitrarily by the user.  For example, subsets of temporal logic can be used in time-series analysis.
* Fitness function - A function that maps an expression to a real number indicating the quality of the expression.  Lower is better.  For example, for a classification task, this may be misclassification rate.

The following problems are currently implemented:

* ACAS X ("ACASXProblem") - Time-series classification task for encounter data from RLESCAS. Learn an expression for the decision boundary to separate NMACs vs. non-NMACs or discover rules to explain clusterings.
* Symbolic regression ("SymbolicRegression") - Reconstruct/rediscover the symbolic form of a mathematical expression from evaluation data only. Located under ExprSearch/modules.

### ACAS X Datasets

The build script automatically copies the default datasets into the Datasets/data folder. The source folders for the data are located under PKGDIR/GrammarExpts/data/Datasets. 

* dasc - Encounter data for DASC dataset
* dascfilt - Encounter data for DASC dataset filtered starting 5 seconds before CPA
* libcas098small_10K - 10K encounters from libcas 0.9.8 
* libcas098smallfilt_10K - 10K encounters from libcas 0.9.8 filtered starting 5 seconds before CPA

### Data

This is not needed in normal operation, but is good info to know (e.g., to inspect processed data).

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

### Configuration Files

Many of the main entry points such as ``ACASX_CE_Tree`` use keyword arguments to set configuration parameters.  Default parameters are typically set for a quick test run.

For convenience, some modules implement a configuration feature.  Config files are stored in a subfolder in the corresponding module.  For example, ``PKGDIR/GrammarExpts/ACASX_CE_Tree/config``

Each config file is a julia file containing a vector of Symbol/Any pairs, for example:
```julia
[
  #tree
  (:maxsteps, 20),

  #CE
  (:num_samples, 5000),
  (:iterations, 100),
  (:elite_frac, 0.6),
  (:w_new, 0.4),
  (:w_prior, 0.1),
  (:maxsteps, 40),
  (:default_code, :(eval(false)))
]
```
that specifies parameter/value pairs.  To load the configuration file, use: ``config = configure(ACASX_CE_Tree, filename)``

Sometimes it is convenient to split up configurations into multiple pieces, for example one file for search and another for data.  To load mutiple config files into the same config dict, use: ``config = configure(ACASX_CE_Tree, "normal", "nvn_dasc")`` which loads "normal.jl" and "nvn\_dasc.jl"

At this point you can inspect the parameters in the config object or even overwrite it:
``config[:param] = newvalue``

To use the configuration, splat it into the keyword arguments of the function call:
``acasx_ce_tree(; config...)``

Values not specified by the configuration take on the defaults specified in the function definition.

### Output

Decision tree and visualization:

* "acasx\_ce\_tree\_log\_vis\_decisiontree.json" - decision tree json output
* "acasx\_ce\_tree\_log\_vis\_decisiontree.pdf" - visualization of the output decision tree. Disable creation of pdf file by setting ``plotpdf=false`` if available in the keyword arguments. The PDF can be later produced from the .json file by calling:

```julia
using TikzQTrees
plottree("acasx_ce_tree_log_vis_decisiontree.json", outfileroot="acasx_ce_tree_log_vis_decisiontree")
```

Output logs are in TaggedDFLogger (RLESUtils.Loggers) format, which is a light wrapper around DataFrames.  To load the data, call:

```julia
using RLESUtils.Loggers
logs = load_log("acasx_ce_tree_log.txt") #recursively loads the .csv.gz files
keys(logs) #see available logs
logs["parameters"] #Parameters log as a dataframe
```

Alternatively, you can open specific logs as dataframes directly

```julia
using DataFrames
D = readtable("acasx_ce_tree_log_result.csv.gz")
```

You may have noticed that DataFrames just stores its data in CSV format, so another way to access the data is to decompress the .csv.gz file into a .csv file and view it.  This is especially useful when just taking a quick look at a particular log.

