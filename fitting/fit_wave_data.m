clear; close all;

task = 'svm';  % svm, krig, neural networks
normalization = true;  % true for normalize
svm_kernel = 'gaussian';
krig_kernel = 'squaredexponential';

load('DataWave/088IRWaSS7_Wi1d89_C4d3_wave.mat', 'WG10_DHI');
data_set = WG10_DHI;
if normalization
    data = my_row_normalize(data_set.Data);
else
    data = data_set.Data;
end
time = data_set.Time;

train_num = int64(0.7*length(data));
train_num = 2000;  % only for fast testing small number of data 
train_data = data(1:train_num);
test_data = data(train_num+1: end);
train_time = time(1: train_num);
test_time = time(train_num+1: end);


%% Training
tic
switch task
    case 'svm'
        [~, svmMdl] = my_fitrsvm(train_time, train_data, svm_kernel);
    case 'krig'
        [~, krigMdl] = my_fitrkrig(train_time, train_data, krig_kernel);  % training model,  krigMdl is model
end
time = toc

%% Prediction
switch task
    case 'svm'
        predict_data = predict(svmMdl, test_time);  % using model generate fitted data
    case 'krig'
        predict_data = predict(krigMdl, test_time);  % using model generate fitted data
end
rmse = immse(predict_data, test_data)/length(test_data)/mean(test_data)
plt_num = 200;
plot(test_time(1:plt_num), test_data(1:plt_num), test_time(1:plt_num), predict_data(1:plt_num))
legend('real data', 'prediction data')