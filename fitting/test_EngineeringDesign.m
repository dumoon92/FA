global ModelInfo

k = 2;
n = 20;

ModelInfo.X = bestlh(n, k, 50,20);

for i = 1:n
    ModelInfo.y(i, 1)=branin(ModelInfo.X(i, :));
end 

UpperTheta = ones(1, k).*2;
LowerTheta=ones(1, k).*-3;

[ModelInfo.Theta, MinNegLnLikelihood] = ...
    ga(@likelihood, k, [], [], [], [], LowerTheta, UpperTheta);

[NegLnLike, ModelInfo.Psi, ModelInfo.U] = likelihood(ModelInfo.Theta);

Xplot = 0: 1/20: 1;
for i = 1:21
    for j = 1:21
        BraninPred(j, i) = pred([Xplot(i) Xplot(j)]);
        BraninTrue(j, i) = branin([Xplot(i) Xplot(j)]);
    end
end