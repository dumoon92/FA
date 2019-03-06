function [tic_toc, predict_data] = my_fit_wave_data(data_set, test_x, task, kernel, normalization)
[m,n] = size(test_x);
if m < n
    test_x = test_x';
end
% task = 'krig';  % svm, krig, neural networks
% normalization = false;  % true for normalize
% svm_kernel = 'rbf';
% krig_kernel = 'squaredexponential';
% 
run_time = datetime('now')
task
if normalization
    data = my_row_normalize(data_set.Data);
else
    data = data_set.Data;
end
time = data_set.Time;

% train_num = int64(0.7*length(data));
train_num = 2000;  % only for fast testing small number of data 
train_data = data(1:train_num);
test_data = data(train_num+1: train_num+int64(train_num/4));
train_time = time(1: train_num);
test_time = time(train_num+1: train_num+int64(train_num/4));

% train_time = 1:train_num;
% train_time = double(train_time');
% test_time = train_num+1: train_num+int64(train_num/4);
% test_time = double(test_time');

%% Training
tic
switch task
    case 'svm'
        [predict_train_data, svmMdl] = my_fitrsvm(train_time, train_data, kernel);
    case 'krig'
        [predict_train_data, krigMdl] = my_fitrkrig(train_time, train_data, kernel);  % training model,  krigMdl is model
    case 'LSTM'
        % https://de.mathworks.com/help/deeplearning/examples/time-series-forecasting-using-deep-learning.html
        numFeatures = 1;
        numResponses = 1;
        numHiddenUnits = 200;
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        options = trainingOptions('adam', ...
            'MaxEpochs',50, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',0, ...
            'Plots','training-progress');

        net = trainNetwork(train_time(1:end-1)',train_data(2:end)',layers,options);

end
tic_toc = toc

%% Prediction
switch task
    case 'svm'
        predict_data = predict(svmMdl, test_time);  % using model generate fitted data
    case 'krig'
        predict_data = predict(krigMdl, test_time);  % using model generate fitted data
    case 'LSTM'
        net = predictAndUpdateState(net,train_data(1:end-1)');
        [net,predict_data] = predictAndUpdateState(net,train_data(end));

        numTimeStepsTest = numel(test_time);
        for i = 2:numTimeStepsTest
            [net,predict_data(:,i)] = predictAndUpdateState(net,predict_data(:,i-1),'ExecutionEnvironment','cpu');
        end
        predict_data = double(predict_data)';

        [net,predict_train_data] = predictAndUpdateState(net,train_data(end));
        size(predict_train_data)
        numTimeStepsTest = numel(train_time);
        for i = 2:numTimeStepsTest
            [net,predict_train_data(:,i)] = predictAndUpdateState(net,predict_train_data(:,i-1),'ExecutionEnvironment','cpu');
        end
        predict_train_data = double(predict_train_data)';
end
rmse = immse(predict_data, test_data)/length(test_data)/mean(test_data)
plt_num = numel(test_time);
plot(train_time, train_data, train_time, predict_train_data, test_time(1:plt_num), test_data(1:plt_num), test_time(1:plt_num), predict_data(1:plt_num))
legend('real data', 'prediction data')