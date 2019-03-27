clear
close  all
clc
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;
y_raw=data.Data;

task = 'svm';

data_set_num = 500;
train_len = 50; 

parameter_str = strcat('-',num2str(data_set_num(end)), '-', num2str(train_len(end)),'_');

predict_len = 300; 
start_train_index = 111; 
start_predict_index = 3e4;

svm_kernel = 'gaussian';
krig_kernel = 'matern52';
if strcmp(task, 'svm')
    [test_y, predict_y, error, rmse] = my_new_svm(y_raw, data_set_num, train_len, predict_len, start_train_index, start_predict_index, svm_kernel);
elseif strcmp(task, 'krig')
    [test_y, predict_y, error, rmse] = my_new_krig(y_raw, data_set_num, train_len, predict_len, start_train_index, start_predict_index, krig_kernel);
end