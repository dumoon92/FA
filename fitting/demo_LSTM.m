close all; clear
num_trn = 2000;
num_tst = 300;
load('088IRWaSS7_Wi1d89_C4d3_wave.mat', 'WG5_DHI');
data_set = WG5_DHI;
Y = data_set.Data(1:num_trn)';
X = data_set.Time(1:num_trn)';
Y_tst = data_set.Data(num_trn+1:num_trn+num_tst)';
X_tst = data_set.Time(num_trn+1:num_trn+num_tst)';

X = my_std(X); Y = my_std(Y); X_tst = my_std(X_tst); Y_tst = my_std(Y_tst);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(X(1:end-1),Y(2:end),layers,options);

net = predictAndUpdateState(net,X);
[net,YPred] = predictAndUpdateState(net,Y(end));

numTimeStepsTest = numel(X_tst);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = std(Y)*YPred + mean(Y);

figure
plot(X(1:end-1), Y(1:end-1))
hold on
idx = num_trn:(num_trn+num_tst);
plot(X_tst, [Y_tst, YPred], '.-');
% plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])