close all
clear;
clc;
format compact;
%%
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;y_raw=data.Data;
y = my_row_normalize(y_raw)*1e2;
y = sin(2*pi*linspace(0, 5*pi, 1e5))';  % use sinus function as test
data_set_num = 2e2;
train_len = 1e3;
predict_len = 2e4;

start_train_index = 1;
start_predict_index = randi([4e4 8e4],1,1)

train_input = zeros(data_set_num, train_len);
train_label = zeros(data_set_num, predict_len);
for i = 1:data_set_num
    train_input(i, :) = y(start_train_index+i-1: start_train_index+i-1+train_len-1)';
    train_label(i, :) = y(start_train_index+train_len+i-1: start_train_index+train_len+i-1+predict_len-1)';
end

test_y = y(start_predict_index: start_predict_index+predict_len-1);
predict_y = zeros(predict_len, predict_len);
predict_y_input = y(start_predict_index-train_len: start_predict_index-1)';  
% the used data for prediction is only from 4e4-2e3 to 4e4-1
for label_index = 1:predict_len
    svmMdl = fitrsvm(train_input, train_label(:, label_index));
    predict_y(:, label_index) = predict(svmMdl, predict_y_input);
%     size(predict(svmMdl, y(start_predict_index-predict_len+label_index:start_predict_index-1+label_index)))
end
plot(predict_y(1, :), 'r--')
hold on 
plot(test_y, 'b')
saveas(gcf, strcat('untitled_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));






