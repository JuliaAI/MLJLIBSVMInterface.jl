using MLJBase
using Test
using LinearAlgebra

using MLJLIBSVMInterface
import StableRNGs


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

rpred = MLJBase.predict(plain_regressor, fitresultR, selectrows(X, test));
nurpred = MLJBase.predict(nu_regressor, fitresultRnu, selectrows(X, test));

@test norm(rpred - y[test])/sqrt(length(y)) < 0.2
@test norm(nurpred - y[test])/sqrt(length(y)) < 0.2


## ANOMALY DETECTION

oneclasssvm = OneClassSVM()

fitresultoc, cacheoc, reportoc = MLJBase.fit(oneclasssvm, 1,
                                          selectrows(X, train));
# output is CategoricalArray{Bool}
ocpred = MLJBase.transform(oneclasssvm,
                           fitresultoc,
                           selectrows(X, test));

# test whether the proprotion of outliers corresponds to the `nu` parameter
@test isapprox((length(train) - sum(MLJBase.transform(oneclasssvm, fitresultoc, selectrows(X, train)) .== true)) / length(train), oneclasssvm.nu, atol=0.005)
@test isapprox((length(test) - sum(ocpred .== true))  / length(test), oneclasssvm.nu, atol=0.05)

