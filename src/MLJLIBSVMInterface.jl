module MLJLIBSVMInterface

export LinearSVC, SVC
export NuSVC, NuSVR
export EpsilonSVR
export OneClassSVM

import MLJModelInterface
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass
import CategoricalArrays
import LIBSVM
using Statistics

const MMI = MLJModelInterface
const PKG = "MLJLIBSVMInterface"


# # MODEL TYPES

mutable struct LinearSVC <: MMI.Deterministic
    solver::LIBSVM.Linearsolver.LINEARSOLVER
    tolerance::Float64
    cost::Float64
    bias::Float64
end

function LinearSVC(
    ;solver::LIBSVM.Linearsolver.LINEARSOLVER = LIBSVM.Linearsolver.L2R_L2LOSS_SVC_DUAL
    ,tolerance::Float64 = Inf
    ,cost::Float64 = 1.0
    ,bias::Float64= -1.0)

    model = LinearSVC(
        solver
        ,tolerance
        ,cost
        ,bias
    )

    message = MMI.clean!(model)
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct SVC <: MMI.Deterministic
    kernel
    gamma::Float64
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    probability::Bool
end

function SVC(
    ;kernel = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,cost::Float64 = 1.0
    ,cachesize::Float64=200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.0
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true
    ,probability::Bool = false)

    model = SVC(
        kernel
        ,gamma
        ,cost
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
        ,probability
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct NuSVC <: MMI.Deterministic
    kernel
    gamma::Float64
    nu::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function NuSVC(
    ;kernel = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,nu::Float64 = 0.5
    ,cachesize::Float64 = 200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = NuSVC(
        kernel
        ,gamma
        ,nu
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct OneClassSVM <: MMI.UnsupervisedDetector
    kernel
    gamma::Float64
    nu::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function OneClassSVM(
    ;kernel = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,nu::Float64 = 0.1
    ,cachesize::Float64 = 200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.0
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = OneClassSVM(
        kernel
        ,gamma
        ,nu
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct NuSVR <: MMI.Deterministic
    kernel
    gamma::Float64
    nu::Float64
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function NuSVR(
    ;kernel = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,nu::Float64 = 0.5
    ,cost::Float64 = 1.0
    ,cachesize::Float64 = 200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = NuSVR(
        kernel
        ,gamma
        ,nu
        ,cost
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct EpsilonSVR <: MMI.Deterministic
    kernel
    gamma::Float64
    epsilon::Float64
    cost::Float64
    cachesize::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
end

function EpsilonSVR(
    ;kernel = LIBSVM.Kernel.RadialBasis
    ,gamma::Float64 = 0.0
    ,epsilon::Float64 = 0.1
    ,cost::Float64 = 1.0
    ,cachesize::Float64 = 200.0
    ,degree::Int32 = Int32(3)
    ,coef0::Float64 = 0.
    ,tolerance::Float64 = .001
    ,shrinking::Bool = true)

    model = EpsilonSVR(
        kernel
        ,gamma
        ,epsilon
        ,cost
        ,cachesize
        ,degree
        ,coef0
        ,tolerance
        ,shrinking
    )

    message = MMI.clean!(model)   #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

# all SVM models defined here:
const SVM = Union{LinearSVC, SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM}


# # CLEAN METHOD

const ERR_PRECOMPUTED_KERNEL = ArgumentError(
    "Pre-computed kernels are not supported by installed version of "*
    "MLJLIBSVMInterface.jl. Alternatively, specify `kernel=k` for some "*
    "function or callable `k(v1::AbstractVector, v2::AbstractVector)`. "
)

function MMI.clean!(model::SVM)
    message = ""
    !(model isa LinearSVC) &&
        model.kernel == LIBSVM.Kernel.Precomputed &&
        throw(ERR_PRECOMPUTED_KERNEL)
    return message
end


# # HELPERS

function err_bad_weights(keys)
    keys_str = join(keys, ", ")
    ArgumentError(
    "Class weights must be a dictionary with these keys: $keys_str. "
    )
end

"""
    map_model_type(model::SVM)

Private method.

Helper function to map the model to the correct LIBSVM model type
needed for function dispatch.

"""
function map_model_type(model::SVM)
    if isa(model, LinearSVC)
        return LIBSVM.LinearSVC
    elseif isa(model, SVC)
        return LIBSVM.SVC
    elseif isa(model, NuSVC)
        return LIBSVM.NuSVC
    elseif isa(model, NuSVR)
        return LIBSVM.NuSVR
    elseif isa(model, EpsilonSVR)
        return LIBSVM.EpsilonSVR
    elseif isa(model, OneClassSVM)
        return LIBSVM.OneClassSVM
    else
        error("Got unsupported model type: $(typeof(model))")
    end
end

"""
    get_svm_parameters(model::Union{SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM})

Private method.

Helper function to get the parameters from the SVM model struct.
"""
function get_svm_parameters(model::Union{SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM})
    #Build arguments for calling svmtrain
    params = Tuple{Symbol, Any}[]
    push!(params, (:svmtype, map_model_type(model))) # get LIBSVM model type
    for fn in fieldnames(typeof(model))
        push!(params, (fn, getfield(model, fn)))
    end

    return params
end

# convert raw value `x` to a `CategoricalValue` using the pool of `v`:
function categorical_value(x, v)
    pool = CategoricalArrays.pool(v)
    return pool[get(pool, x)]
end

# to ensure the keys of user-provided weights are `CategoricalValue`s,
# and the values are floats:
fix_keys(weights::Dict{<:CategoricalArrays.CategoricalValue}, y) =
    Dict(k => float(weights[k]) for k in keys(weights))
fix_keys(weights, y) =
    Dict(categorical_value(x, y) => float(weights[x]) for x in keys(weights))

"""
    encode(weights::Dict, y)

Private method.

Check that `weights` is a valid dictionary, based on the pool of `y`,
and return a new dictionary whose keys are restricted to those
appearing as elements of `y` (and not just appearing in the pool of
`y`) and which are additionally replaced by their integer representations
(the categorical reference integers).

"""
function encode(weights::Dict, y)
    kys = CategoricalArrays.levels(y)
    Set(keys(weights)) == Set(kys) || throw(err_bad_weights(kys))
    _weights = fix_keys(weights, y)
    levels_seen = unique(y) # not `CategoricalValue`s !
    cvs = [categorical_value(x, y) for x in levels_seen]
    return Dict(MMI.int(cv) => _weights[cv] for cv in cvs)
end

function get_encoding(decoder)
    refs = MMI.int.(decoder.classes)
    return Dict(i => decoder(i) for i in refs)
end


"""
    orientation(scores)

Private method.

Return `1` if the majority of elements of `scores` are less than the
midpoint between the minimum and maximum values. For outlier detection
scores, the implication in that case is that scores increase with
increasing likelihood of outlierness.

Otherwise return `-1` (scores decrease with increasing likelihood of
outlierness).

"""
function orientation(scores)
    middle = (maximum(scores) + minimum(scores))/2
    if quantile(scores, 0.5) < middle
        return 1
    else
        return -1
    end
end


# # FIT METHOD

function MMI.fit(model::LinearSVC, verbosity::Int, X, y, weights=nothing)

    Xmatrix = MMI.matrix(X)' # notice the transpose
    y_plain = MMI.int(y)
    decode  = MMI.decoder(y[1]) # for predict method

    _weights = if weights == nothing
        nothing
    else
        encode(weights, y)
    end

    result = LIBSVM.LIBLINEAR.linear_train(y_plain, Xmatrix,
        weights = _weights, solver_type = Int32(model.solver),
        C = model.cost, bias = model.bias,
        eps = model.tolerance, verbose = ifelse(verbosity > 1, true, false)
    )

    fitresult = (result, decode)
    cache = nothing
    report = NamedTuple()

    return fitresult, cache, report
end

MMI.fitted_params(::LinearSVC, fitresult) =
    (libsvm_model=fitresult[1], encoding=get_encoding(fitresult[2]))

function MMI.fit(model::Union{SVC, NuSVC}, verbosity::Int, X, y, weights=nothing)

    Xmatrix = MMI.matrix(X)' # notice the transpose
    y_plain = MMI.int(y)
    decode  = MMI.decoder(y[1]) # for predict method

    _weights = if weights == nothing
        nothing
    else
        model isa NuSVC && error("`NuSVC` does not support class weights. ")
        encode(weights, y)
    end

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    model.gamma == 0.0 && (model.gamma = 1.0/(var(Xmatrix) * size(Xmatrix, 1)) )
    result = LIBSVM.svmtrain(Xmatrix, y_plain;
                             get_svm_parameters(model)..., weights=_weights,
                             verbose = ifelse(verbosity > 1, true, false)
                             )

    fitresult = (result, decode)
    cache = nothing
    report = (gamma=model.gamma,)

    return fitresult, cache, report
end

MMI.fitted_params(::Union{SVC, NuSVC}, fitresult) =
    (libsvm_model=fitresult[1], encoding=get_encoding(fitresult[2]))


function MMI.fit(model::Union{NuSVR, EpsilonSVR}, verbosity::Int, X, y)

    Xmatrix = MMI.matrix(X)' # notice the transpose

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    model.gamma == 0.0 && (model.gamma = 1.0/(var(Xmatrix) * size(Xmatrix, 1)) )
    fitresult = LIBSVM.svmtrain(Xmatrix, y;
        get_svm_parameters(model)...,
        verbose = ifelse(verbosity > 1, true, false)
    )

    report = (gamma=model.gamma,)

    return fitresult, cache, report
end

MMI.fitted_params(::Union{NuSVR, EpsilonSVR}, fitresult) =
    (libsvm_model=fitresult,)

function MMI.fit(model::OneClassSVM, verbosity::Int, X)

    Xmatrix = MMI.matrix(X)' # notice the transpose

    cache = nothing

    model = deepcopy(model)
    model.gamma == -1.0 && (model.gamma = 1.0/size(Xmatrix, 1))
    model.gamma == 0.0 && (model.gamma = 1.0/(var(Xmatrix) * size(Xmatrix, 1)) )
    libsvm_model = LIBSVM.svmtrain(Xmatrix;
                                   get_svm_parameters(model)...,
                                   verbose = ifelse(verbosity > 1, true, false)
                                   )

    # get orientation and training scores:
    p, decision_matrix = LIBSVM.svmpredict(libsvm_model, Xmatrix)
    decision_scores = view(decision_matrix, 1, :)
    orientation = MLJLIBSVMInterface.orientation(decision_scores)
    scores = orientation*decision_scores
    @show p

    fitresult = (libsvm_model, orientation)
    report = (gamma=model.gamma, scores=scores)

    return fitresult, cache, report
end

MMI.fitted_params(::OneClassSVM, fitresult) =
    (libsvm_model=fitresult[1], orientation=fitresult[2])


# # PREDICT AND TRANSFORM

function MMI.predict(model::LinearSVC, fitresult, Xnew)
    result, decode = fitresult
    p, _ = LIBSVM.LIBLINEAR.linear_predict(result, MMI.matrix(Xnew)')
    return decode(p)
end

function MMI.predict(model::Union{SVC, NuSVC}, fitresult, Xnew)
    result, decode = fitresult
    p, _ = LIBSVM.svmpredict(result, MMI.matrix(Xnew)')
    return decode(p)
end

function MMI.predict(model::Union{NuSVR, EpsilonSVR}, fitresult, Xnew)
    (p,d) = LIBSVM.svmpredict(fitresult, MMI.matrix(Xnew)')
    return p
end

function MMI.transform(model::OneClassSVM, fitresult, Xnew)
    libsvm_model, orientation = fitresult
    _, decision_matrix = LIBSVM.svmpredict(libsvm_model, MMI.matrix(Xnew)')
    decision_scores = view(decision_matrix, 1, :)
    return orientation*decision_scores
end


# metadata
MMI.load_path(::Type{<:LinearSVC}) = "$PKG.LinearSVC"
MMI.load_path(::Type{<:SVC}) = "$PKG.SVC"
MMI.load_path(::Type{<:NuSVC}) = "$PKG.NuSVC"
MMI.load_path(::Type{<:NuSVR}) = "$PKG.NuSVR"
MMI.load_path(::Type{<:EpsilonSVR}) = "$PKG.EpsilonSVR"
MMI.load_path(::Type{<:OneClassSVM}) = "$PKG.OneClassSVM"

MMI.supports_class_weights(::Type{<:LinearSVC}) = true
MMI.supports_class_weights(::Type{<:SVC}) = true

MMI.human_name(::Type{<:LinearSVC}) = "linear support vector classifier"
MMI.human_name(::Type{<:SVC}) = "C-support vector classifier"
MMI.human_name(::Type{<:NuSVC}) = "nu-support vector classifier"
MMI.human_name(::Type{<:NuSVR}) = "nu-support vector regressor"
MMI.human_name(::Type{<:EpsilonSVR}) = "epsilon-support vector regressor"
MMI.human_name(::Type{<:OneClassSVM}) = "$one-class support vector machine"

MMI.package_name(::Type{<:SVM}) = "LIBSVM"
MMI.package_uuid(::Type{<:SVM}) = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
MMI.is_pure_julia(::Type{<:SVM}) = false
MMI.package_url(::Type{<:SVM}) = "https://github.com/mpastell/LIBSVM.jl"
MMI.input_scitype(::Type{<:SVM}) = Table(Continuous)
MMI.target_scitype(::Type{<:Union{LinearSVC, SVC, NuSVC}}) =
    AbstractVector{<:Finite}
MMI.target_scitype(::Type{<:Union{NuSVR, EpsilonSVR}}) =
    AbstractVector{Continuous}
MMI.output_scitype(::Type{<:OneClassSVM}) =
    AbstractVector{<:Finite{2}} # Bool (true means inlier)


# # DOCUMENT STRINGS

const DOC_REFERENCE = "C.-C. Chang and C.-J. Lin (2011): \"LIBSVM: a library for "*
    "support vector machines.\" *ACM Transactions on Intelligent Systems and "*
    "Technology*, 2(3):27:1–27:27. Updated at "*
    "[https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf]"*
    "(https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)"

const DOC_ALGORITHM = "Reference for algorithm and core C-library: $DOC_REFERENCE. "

const DOC_REFERENCE_LINEAR = "Rong-En Fan et al (2008): \"LIBLINEAR: A Library for "*
    "Large Linear Classification.\" *Journal of Machine Learning Research* 9 1871-1874. "*
    "Available at [https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf]"*
    "(https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf)"

const DOC_ALGORITHM_LINEAR = "Reference for algorithm and core C-library: "*
    "$DOC_REFERENCE_LINEAR. "

const DOC_KERNEL = """
- `kernel=LIBSVM.Kernel.RadialBasis`: either an object that can be
  called, as in `kernel(x1, x2)` (where `x1` and `x2` are vectors
  whose length matches the number of columns of the training data `X`,
  see examples below) or one of the following built-in kernels from
  the LIBSVM package:

  - `LIBSVM.Kernel.Linear`: `(x1, x2) -> x1'*x2`

  - `LIBSVM.Kernel.Polynomial`: `(x1, x2) -> gamma*x1'*x2 + coef0)^degree`

  - `LIBSVM.Kernel.RadialBasis`: `(x1, x2) -> (exp(-gamma*norm(x1, x2)^2)`

  - `LIBSVM.Kernel.Sigmoid`: `(x1, x2) - > tanh(gamma*x1'*x2 + coef0)`

  where `gamma`, `coef0`, `degree` are other hyper-parameters.

- `gamma = 0.0`: kernel parameter (see above); if `gamma==-1.0` then
  `gamma = 1/nfeatures` is used in training, where `nfeatures` is the
  number of features (columns of `X`).  If `gamma==0.0` then `gamma =
  1/(var(Tables.matrix(X))*nfeatures)` is used. Actual value used
  appears in the report (see below).

- `coef0 = 0.0`: kernel parameter (see above)

- `degree::Int32 = Int32(3)`: degree in polynomial kernel (see above)

"""


# ## LinearSVC

"""
$(MMI.doc_header(LinearSVC))

$DOC_ALGORITHM_LINEAR

This model type is similar to `SVC` from the same package with the setting
`kernel=LIBSVM.Kernel.KERNEL.Linear`, but is optimized for the linear
case.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with one of:

    mach = machine(model, X, y)
    mach = machine(model, X, y, w)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have `Continuous` element scitype; check column scitypes with
  `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

- `w`: a dictionary of class weights, keyed on `levels(y)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `solver=LIBSVM.Linearsolver.L2R_L2LOSS_SVC_DUAL`: linear solver,
  which must be one of the following from the LIBSVM.jl package:

    - `LIBSVM.Linearsolver.L2R_LR`: L2-regularized logistic regression (primal))

    - `LIBSVM.Linearsolver.L2R_L2LOSS_SVC_DUAL`: L2-regularized
      L2-loss support vector classification (dual)

    - `LIBSVM.Linearsolver.L2R_L2LOSS_SVC`: L2-regularized L2-loss
      support vector classification (primal)

    - `LIBSVM.Linearsolver.L2R_L1LOSS_SVC_DUAL`: L2-regularized
      L1-loss support vector classification (dual)

    - `LIBSVM.Linearsolver.MCSVM_CS`: support vector classification by
      Crammer and Singer) `LIBSVM.Linearsolver.L1R_L2LOSS_SVC`:
      L1-regularized L2-loss support vector classification)

    - `LIBSVM.Linearsolver.L1R_LR`:  L1-regularized logistic regression

    - `LIBSVM.Linearsolver.L2R_LR_DUAL`: L2-regularized logistic regression (dual)

- `tolerance::Float64=Inf`: tolerance for the stopping criterion;

- `cost=1.0` (range (0, `Inf`)): the parameter denoted ``C`` in the
  cited reference; for greater regularization, decrease `cost`

- `bias= -1.0`: if `bias >= 0`, instance `x` becomes `[x; bias]`; if
  `bias < 0`, no bias term added (default -1)


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `libsvm_model`: the trained model object created by the LIBSVM.jl package

- `encoding`: class encoding used internally by `libsvm_model` - a
  dictionary of class labels keyed on the internal integer representation


# Examples

```
using MLJ
import LIBSVM

LinearSVC = @load LinearSVC pkg=LIBSVM               # model type
model = LinearSVC(solver=LIBSVM.Linearsolver.L2R_LR) # instance

X, y = @load_iris # table, vector
mach = machine(model, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "versicolor"
 "virginica"
```

## Incorporating class weights

```julia
weights = Dict("virginica" => 1, "versicolor" => 20, "setosa" => 1)
mach = machine(model, X, y, weights) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "versicolor"
 "versicolor"
 "versicolor"
```


See also the [`SVC`](@ref) and [`NuSVC`] classifiers, and
[LIVSVM.jl](https://github.com/JuliaML/LIBSVM.jl) and the
[documentation](https://github.com/cjlin1/liblinear/blob/master/README)
for the original C implementation.

"""
LinearSVC


# ## SVC

"""
$(MMI.doc_header(SVC))

$DOC_ALGORITHM


# Training data

In MLJ or MLJBase, bind an instance `model` to data with one of:

    mach = machine(model, X, y)
    mach = machine(model, X, y, w)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have `Continuous` element scitype; check column scitypes with
  `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

- `w`: a dictionary of class weights, keyed on `levels(y)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

$DOC_KERNEL

- `cost=1.0` (range (0, `Inf`)): the parameter denoted ``C`` in the
  cited reference; for greater regularization, decrease `cost`

- `cachesize=200.0` cache memory size in MB

- `tolerance=0.001`: tolerance for the stopping criterion

- `shrinking=true`: whether to use shrinking heuristics

- `probability=false`: whether to base classification on calibrated
  probabilities (expensive) or to use the raw decision function
  (recommended). Note that in either case `predict` returns point
  predictions and not probabilities, so that this option has little
  utility in the current re-implementation.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `libsvm_model`: the trained model object created by the LIBSVM.jl package

- `encoding`: class encoding used internally by `libsvm_model` - a
  dictionary of class labels keyed on the internal integer representation


# Report

The fields of `report(mach)` are:

- `gamma`: actual value of the kernel parameter `gamma` used in training


# Examples

## Using a built-in kernel

```
using MLJ
import LIBSVM

SVC = @load SVC pkg=LIBSVM               # model type
model = SVC(kernel=LIBSVM.Kernel.Polynomial) # instance

X, y = @load_iris # table, vector
mach = machine(model, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "virginica"
 "virginica"
```

## Using a user-defined kernel

```
k(x1, x2) = x1'*x2 # equivalent to `LIBSVM.Kernel.Linear`
model = SVC(kernel=k)
mach = machine(model, X, y) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "virginica"
 "virginica"
```

## Incorporating class weights

In either scenario above, we can do:

```julia
weights = Dict("virginica" => 1, "versicolor" => 20, "setosa" => 1)
mach = machine(model, X, y, weights) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "versicolor"
 "versicolor"
 "versicolor"
```

See also the classifiers [`NuSVC`](@ref) and [`LinearSVC`](@ref), and
[LIVSVM.jl](https://github.com/JuliaML/LIBSVM.jl) and the
[documentation](https://github.com/cjlin1/libsvm/blob/master/README)
for the original C implementation.

"""
SVC


# ## NuSVC

"""
$(MMI.doc_header(NuSVC))

$DOC_ALGORITHM

This model is simply a reparameterization of the [`SVC`](@ref)
classifier, where `nu` replaces `cost`, and is therefore mathematically
equivalent to it.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have `Continuous` element scitype; check column scitypes with
  `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

$DOC_KERNEL

- `nu=0.5` (range (0, 1]): An upper bound on the fraction of margin
  errors and a lower bound of the fraction of support vectors. Denoted
  `ν` in the cited paper.

- `cachesize=200.0` cache memory size in MB

- `tolerance=0.001`: tolerance for the stopping criterion

- `shrinking=true`: whether to use shrinking heuristics


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `libsvm_model`: the trained model object created by the LIBSVM.jl package

- `encoding`: class encoding used internally by `libsvm_model` - a
  dictionary of class labels keyed on the internal integer representation


# Report

The fields of `report(mach)` are:

- `gamma`: actual value of the kernel parameter `gamma` used in training


# Examples

## Using a built-in kernel

```
using MLJ
import LIBSVM

NuSVC = @load NuSVC pkg=LIBSVM               # model type
model = NuSVC(kernel=LIBSVM.Kernel.Polynomial) # instance

X, y = @load_iris # table, vector
mach = machine(model, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "virginica"
 "virginica"
```

## Using a user-defined kernel

```
k(x1, x2) = x1'*x2 # equivalent to `LIBSVM.Kernel.Linear`
model = NuSVC(kernel=k)
mach = machine(model, X, y) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "virginica"
 "virginica"
```

## Incorporating class weights

In either scenario above, we can do:

```julia
weights = Dict("virginica" => 1, "versicolor" => 20, "setosa" => 1)
mach = machine(model, X, y, weights) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "versicolor"
 "versicolor"
 "versicolor"
```

See also the classifiers [`SVC`](@ref) and [`LinearSVC`](@ref),
[LIVSVM.jl](https://github.com/JuliaML/LIBSVM.jl) and the
[documentation](https://github.com/cjlin1/libsvm/blob/master/README)
for the original C implementation.

"""
NuSVC


# ## EpsilonSVR

"""
$(MMI.doc_header(EpsilonSVR))

$DOC_ALGORITHM

This model is an adaptation of the classifier `SVC` to regression, but
has an additional parameter `epsilon` (denoted `ϵ` in the cited
reference).


# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have `Continuous` element scitype; check column scitypes with
  `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

$DOC_KERNEL

- `cost=1.0` (range (0, `Inf`)): the parameter denoted ``C`` in the
  cited reference; for greater regularization, decrease `cost`

- `epsilon=0.1` (range (0, `Inf`)): the parameter denoted ``ϵ`` in the
  cited reference; `epsilon` is the thickness of the "penalty-free"
  neighborhood of the decision surface.

- `cachesize=200.0` cache memory size in MB

- `tolerance=0.001`: tolerance for the stopping criterion

- `shrinking=true`: whether to use shrinking heuristics


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `libsvm_model`: the trained model object created by the LIBSVM.jl package


# Report

The fields of `report(mach)` are:

- `gamma`: actual value of the kernel parameter `gamma` used in training


# Examples

## Using a built-in kernel

```
using MLJ
import LIBSVM

EpsilonSVR = @load EpsilonSVR pkg=LIBSVM            # model type
model = EpsilonSVR(kernel=LIBSVM.Kernel.Polynomial) # instance

X, y = make_regression(rng=123) # table, vector
mach = machine(model, X, y) |> fit!

Xnew, _ = make_regression(3, rng=123)

julia> yhat = predict(mach, Xnew)
3-element Vector{Float64}:
  0.2512132502584155
  0.007340201523624579
 -0.2482949812264707
```

## Using a user-defined kernel

```
k(x1, x2) = x1'*x2 # equivalent to `LIBSVM.Kernel.Linear`
model = EpsilonSVR(kernel=k)
mach = machine(model, X, y) |> fit!

julia> yhat = predict(mach, Xnew)
3-element Vector{Float64}:
  1.1121225361666656
  0.04667702229741916
 -0.6958148424680672
```

See also [`NuSVR`](@ref),
[LIVSVM.jl](https://github.com/JuliaML/LIBSVM.jl) and the
[documentation](https://github.com/cjlin1/libsvm/blob/master/README)
for the original C implementation.

"""
EpsilonSVR


# ## NuSVR

"""
$(MMI.doc_header(NuSVR))

$DOC_ALGORITHM

This model is a reparameterization of `EpsilonSCV` in which the
`epsilon` hyper-parameter is replaced with a new parameter `nu`
(denoted ``ν`` in the cited reference) which attempts to control the
number of support vectors directly.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have `Continuous` element scitype; check column scitypes with
  `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- $DOC_KERNEL

- `cost=1.0` (range (0, `Inf`)): the parameter denoted ``C`` in the
  cited reference; for greater regularization, decrease `cost`

- `nu=0.5` (range (0, 1]): An upper bound on the fraction of margin
  errors and a lower bound of the fraction of support vectors. Denoted
  ``ν`` in the cited paper.

- `cachesize=200.0` cache memory size in MB

- `tolerance=0.001`: tolerance for the stopping criterion

- `shrinking=true`: whether to use shrinking heuristics


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `libsvm_model`: the trained model object created by the LIBSVM.jl package


# Report

The fields of `report(mach)` are:

- `gamma`: actual value of the kernel parameter `gamma` used in training


# Examples

## Using a built-in kernel

```
using MLJ
import LIBSVM

NuSVR = @load NuSVR pkg=LIBSVM            # model type
model = NuSVR(kernel=LIBSVM.Kernel.Polynomial) # instance

X, y = make_regression(rng=123) # table, vector
mach = machine(model, X, y) |> fit!

Xnew, _ = make_regression(3, rng=123)

julia> yhat = predict(mach, Xnew)
3-element Vector{Float64}:
  0.2008156459920009
  0.1131520519131709
 -0.2076156254934889
```

## Using a user-defined kernel

```
k(x1, x2) = x1'*x2 # equivalent to `LIBSVM.Kernel.Linear`
model = NuSVR(kernel=k)
mach = machine(model, X, y) |> fit!

julia> yhat = predict(mach, Xnew)
3-element Vector{Float64}:
  1.1211558175964662
  0.06677125944808422
 -0.6817578942749346
```

See also [`EpsilonSVR`](@ref),
[LIVSVM.jl](https://github.com/JuliaML/LIBSVM.jl) and the
[documentation](https://github.com/cjlin1/libsvm/blob/master/README)
for the original C implementation.

"""
NuSVR


# ## OneClassSVM

"""
$(MMI.doc_header(OneClassSVM))

$DOC_ALGORITHM

This model is an outlier detection model delivering raw scores based
on the decision function of a support vector machine. Like the
[`NuSVC`](@ref) classifier, it uses the `nu` reparameterization of the
`cost` parameter appearing in standard support vector classification
[`SVC`](@ref).

To extract
normalized scores ("probabilities") wrap the model using
`ProbabilisticDetector` from
[OutlierDection.jl](https://github.com/OutlierDetectionJL/OutlierDetection.jl). For
threshold-based classification, wrap the probabilistic model using
MLJ's `BinaryThresholdPredictor`. Examples of wrapping appear below.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have `Continuous` element scitype; check column scitypes with
  `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

$DOC_KERNEL

- `nu=0.5` (range (0, 1]): An upper bound on the fraction of margin
  errors and a lower bound of the fraction of support vectors. Denoted
  `ν` in the cited paper.

- `cachesize=200.0` cache memory size in MB

- `tolerance=0.001`: tolerance for the stopping criterion

- `shrinking=true`: whether to use shrinking heuristics


# Operations

- `transform(mach, Xnew)`: return predictions of the target, given
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `libsvm_model`: the trained model object created by the LIBSVM.jl package

- `orientation`: this equals `1` if the decision function for
  `libsvm_model` is increasing with increasing outlierness, and `-1`
  if it is decreasing instead. Correspondingly, the `libsvm_model` attaches


# Report

The fields of `report(mach)` are:

- `gamma`: actual value of the kernel parameter `gamma` used in training


# Examples

## Generating raw scores for outlierness


```
using MLJ
import LIBSVM
import StableRNGs.StableRNG

OneClassSVM = @load OneClassSVM pkg=LIBSVM           # model type
model = OneClassSVM(kernel=LIBSVM.Kernel.Polynomial) # instance

rng = StableRNG(123)
Xmatrix = randn(rng, 5, 3)
Xmatrix[1, 1] = 100.0
X = MLJ.table(Xmatrix)

mach = machine(model, X) |> fit!

# training scores (outliers have larger scores):
julia> report(mach).scores
5-element Vector{Float64}:
  6.711689156091755e-7
 -6.740101976655081e-7
 -6.711632439648446e-7
 -6.743015858874887e-7
 -6.745393717880104e-7

# scores for new data:
Xnew = MLJ.table(rand(rng, 2, 3))

julia> transform(mach, rand(rng, 2, 3))
2-element Vector{Float64}:
 -6.746293022511047e-7
 -6.744289265348623e-7
```

## Generating probabilistic predictions of outlierness

Continuing the previous example:

```
using OutlierDetection
pmodel = ProbabilisticDetector(model)
pmach = machine(pmodel, X) |> fit!

# probabilistic predictions on new data:

julia> y_prob = predict(pmach, Xnew)
2-element UnivariateFiniteVector{OrderedFactor{2}, String, UInt8, Float64}:
 UnivariateFinite{OrderedFactor{2}}(normal=>1.0, outlier=>9.57e-5)
 UnivariateFinite{OrderedFactor{2}}(normal=>1.0, outlier=>0.0)

# probabilities for outlierness:

julia> pdf.(y_prob, "outlier")
2-element Vector{Float64}:
 9.572583265925801e-5
 0.0


## Outlier classification using a probability threshold:

Continuing the previous example:

```
dmodel = BinaryThresholdPredictor(pmodel, threshold=0.9)
dmach = machine(dmodel, X) |> fit!

## Using a user-defined kernel


```
k(x1, x2) = x1'*x2 # equivalent to `LIBSVM.Kernel.Linear`
model = OneClassSVM(kernel=k)
mach = machine(model, X, y) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "virginica"
 "virginica"
```

## Incorporating class weights

In either scenario above, we can do:

```julia
weights = Dict("virginica" => 1, "versicolor" => 20, "setosa" => 1)
mach = machine(model, X, y, weights) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "versicolor"
 "versicolor"
 "versicolor"
```

See also the classifiers [`SVC`](@ref) and [`LinearSVC`](@ref),
[LIVSVM.jl](https://github.com/JuliaML/LIBSVM.jl) and the
[documentation](https://github.com/cjlin1/libsvm/blob/master/README)
for the original C implementation.

"""
OneClassSVM

end # module
