close all
clear;
clc;
format compact;
%%
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;y_raw=data.Data;
y = my_row_normalize(y_raw)*1e2;

data_set_num = 2e3;
train_num = 5e2;
label_num = 10;

start_index = randi([1 4e4],1,1);

train_input = zeros(data_set_num, train_num);
train_label = zeros(data_set_num, label_num);
for i = 1:data_set_num
    train_input(i, :) = y(start_index: start_index+train_num-1);
    train_label(i, :) = y(start_index+train_num: start_index+train_num+label_num-1);
end

svmMdl = fitrsvm(train_input, train_label);
Y_svm = predict(svmMdl, train_input);
plot(Y_svm(:,1))






