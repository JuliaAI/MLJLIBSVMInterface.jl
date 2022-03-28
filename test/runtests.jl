using MLJBase
using Test
using LinearAlgebra
using CategoricalArrays

using MLJLIBSVMInterface
import StableRNGs
import LIBSVM


## HELPERS

@testset "`fix_keys` and `encode` for weight dicts" begin
    v = categorical(['a', 'b', 'b', 'c'])
    weights = Dict('a' => 1.0, 'b' => 2.0, 'c' => 3.0)
    vfixed = MLJLIBSVMInterface.fix_keys(weights, v)
    @test vfixed[v[1]] == 1.0
    @test vfixed[v[2]] == 2.0
    @test vfixed[v[4]] == 3.0
    @test length(keys(vfixed)) == 3
    @test MLJLIBSVMInterface.fix_keys(vfixed, v) == vfixed

    refs = int.(v)
    weights_encoded = MLJLIBSVMInterface.encode(weights, v[1:end-1]) # exludes `c`
    @test weights_encoded[refs[1]] == 1.0
    @test weights_encoded[refs[2]] == 2.0
    @test length(keys(weights_encoded)) ==  2

    @test_throws(MLJLIBSVMInterface.err_bad_weights(levels(v)),
                 MLJLIBSVMInterface.encode(Dict('d'=> 1.0), v))
end

@testset "orientation of scores" begin
    scores = [1, 1, 1, 1, 0]
    @test MLJLIBSVMInterface.orientation(scores) == -1
    @test MLJLIBSVMInterface.orientation(-scores) == 1
    @test MLJLIBSVMInterface.orientation(scores .+ 100) == -1
    @test MLJLIBSVMInterface.orientation(-scores .+ 100) == 1
end


## CLASSIFIERS

plain_classifier = SVC()
nu_classifier = NuSVC()
linear_classifier = LinearSVC()

# test preservation of categorical levels:
X, y = @load_iris

train, test = partition(eachindex(y), 0.6); # levels of y are split across split

fitresultC, cacheC, reportC = MLJBase.fit(plain_classifier, 1,
                                          selectrows(X, train), y[train]);
fitresultCnu, cacheCnu, reportCnu = MLJBase.fit(nu_classifier, 1,
                                          selectrows(X, train), y[train]);
fitresultCL, cacheCL, reportCL = MLJBase.fit(linear_classifier, 1,
                                          selectrows(X, train), y[train]);
pcpred = MLJBase.predict(plain_classifier, fitresultC, selectrows(X, test));
nucpred = MLJBase.predict(nu_classifier, fitresultCnu, selectrows(X, test));
lcpred = MLJBase.predict(linear_classifier, fitresultCL, selectrows(X, test));

@test Set(classes(pcpred[1])) == Set(classes(y[1]))
@test Set(classes(nucpred[1])) == Set(classes(y[1]))
@test Set(classes(lcpred[1])) == Set(classes(y[1]))

fpC = MLJBase.fitted_params(plain_classifier, fitresultC)
fpCnu = MLJBase.fitted_params(nu_classifier, fitresultCnu)
fpCL = MLJBase.fitted_params(linear_classifier, fitresultCL)

for fp in [fpC, fpCnu, fpCL]
    @test keys(fp) == (:libsvm_model, :encoding)
    @test fp.encoding[int(MLJBase.classes(y)[1])] == classes(y)[1]
end

rng = StableRNGs.StableRNG(123)

# test with linear data:
x1 = randn(rng, 3000);
x2 = randn(rng, 3000);
x3 = randn(rng, 3000);
X = (x1=x1, x2=x2, x3=x3);
y = x1 - x2 -2x3;
ycat = map(y) do η
    η > 0 ? "go" : "stop"
end |> categorical;
train, test = partition(eachindex(ycat), 0.8);
fitresultC, cacheC, reportC = MLJBase.fit(plain_classifier, 1,
                                          selectrows(X, train), ycat[train]);
fitresultCnu, cacheCnu, reportCnu = MLJBase.fit(nu_classifier, 1,
                                          selectrows(X, train), ycat[train]);
fitresultCL, cacheCL, reportCL = MLJBase.fit(linear_classifier, 1,
                                          selectrows(X, train), ycat[train]);
pcpred = MLJBase.predict(plain_classifier, fitresultC, selectrows(X, test));
nucpred = MLJBase.predict(nu_classifier, fitresultCnu, selectrows(X, test));
lcpred = MLJBase.predict(linear_classifier, fitresultCL, selectrows(X, test));
@test sum(pcpred .!= ycat[test])/length(ycat) < 0.05
@test sum(nucpred .!= ycat[test])/length(ycat) < 0.05
@test sum(lcpred .!= ycat[test])/length(ycat) < 0.05


## REGRESSORS

plain_regressor = EpsilonSVR()
nu_regressor = NuSVR()

# test with linear data:
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 1,
                                          selectrows(X, train), y[train]);
fitresultRnu, cacheRnu, reportRnu = MLJBase.fit(nu_regressor, 1,
                                                selectrows(X, train), y[train]);

fpR = MLJBase.fitted_params(plain_regressor, fitresultR)
fpRnu = MLJBase.fitted_params(nu_regressor, fitresultRnu)

for fp in [fpR, fpRnu]
    @test fp.libsvm_model isa LIBSVM.SVM
end

rpred = MLJBase.predict(plain_regressor, fitresultR, selectrows(X, test));
nurpred = MLJBase.predict(nu_regressor, fitresultRnu, selectrows(X, test));

@test norm(rpred - y[test])/sqrt(length(y)) < 0.2
@test norm(nurpred - y[test])/sqrt(length(y)) < 0.2


## ANOMALY DETECTION

N = 50
rng = StableRNGs.StableRNG(123)
Xmatrix = randn(rng, 2N, 3)

# insert outliers at observation 1 and N:
Xmatrix[1, 1] = 100.0
Xmatrix[N, 3] = 200.0

X = MLJBase.table(Xmatrix)

oneclasssvm = OneClassSVM()

fitresultoc, cacheoc, reportoc = MLJBase.fit(oneclasssvm, 1, X)

fp = MLJBase.fitted_params(oneclasssvm, fitresultoc)
@test fp.libsvm_model isa LIBSVM.SVM

training_scores = reportoc.scores
scores = MLJBase.transform(oneclasssvm, fitresultoc, X)

@test scores == training_scores

# crude extraction of outliers from scores:
midpoint = mean([minimum(scores), maximum(scores)])
outlier_indices = filter(eachindex(scores)) do i
    scores[i] .> midpoint
end

@test outlier_indices == [1, N]


## CONSTRUCTOR FAILS

@test_throws(MLJLIBSVMInterface.ERR_PRECOMPUTED_KERNEL,
             SVC(kernel=LIBSVM.Kernel.Precomputed))


## CALLABLE KERNEL

X, y = make_blobs()

kernel(x1, x2) = x1' * x2

model  = SVC(kernel=kernel)
model₂ = SVC(kernel=LIBSVM.Kernel.Linear)

fitresult, cache, report = MLJBase.fit(model, 0, X, y);
fitresult₂, cache₂, report₂ = MLJBase.fit(model₂, 0, X, y);

@test fitresult[1].rho ≈ fitresult₂[1].rho
@test fitresult[1].coefs ≈ fitresult₂[1].coefs
@test fitresult[1].SVs.indices ≈ fitresult₂[1].SVs.indices

yhat = MLJBase.predict(model, fitresult, X);
yhat₂ = MLJBase.predict(model₂, fitresult₂, X);

@test yhat == yhat₂

@test accuracy(yhat, y) > 0.75

model = @test_throws(MLJLIBSVMInterface.ERR_PRECOMPUTED_KERNEL,
                   SVC(kernel=LIBSVM.Kernel.Precomputed))


## WEIGHTS

rng = StableRNGs.StableRNG(123)
centers = [0 0;
           0.1 0;
           0.2 0]
X, y = make_blobs(100, rng=rng, centers=centers,) # blobs close together

train = eachindex(y)[y .!= 2]
Xtrain = selectrows(X, train)
ytrain = y[train] # the `2` class is not in here

weights_uniform     = Dict(1=> 1.0, 2=> 1.0, 3=> 1.0)
weights_favouring_3 = Dict(1=> 1.0, 2=> 1.0, 3=> 100.0)

for model in [SVC(), LinearSVC()]

    # without weights:
    Θ, _, _ = MLJBase.fit(model, 0, Xtrain, ytrain)
    ŷ = predict(model, Θ, X);
    @test levels(ŷ) == levels(y) # the `2` class persists as a level

    # with uniform weights:
    Θ_uniform, _, _ = MLJBase.fit(model, 0, Xtrain, ytrain, weights_uniform)
    ŷ_uniform = predict(model, Θ_uniform, X);
    @test levels(ŷ_uniform) == levels(y)

    # with weights favouring class `3`:
    Θ_favouring_3, _, _ = MLJBase.fit(model, 0, Xtrain, ytrain, weights_favouring_3)
    ŷ_favouring_3 = predict(model, Θ_favouring_3, X);
    @test levels(ŷ_favouring_3) == levels(y)

    # comparisons:
    if !(model isa LinearSVC) # linear solver is not deterministic
        @test ŷ_uniform == ŷ
    end
    d = sum(ŷ_favouring_3 .== 3) - sum(ŷ .== 3)
    if d <= 0
        @show model
        @show d
    end

end
