close all
clear;
clc;
format compact;
%%
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;
y_raw=data.Data;
y = my_row_normalize(y_raw);
% y = sin(2*pi*linspace(0, 5*pi, 1e3))';  % use sinus function as test
data_set_num = 3e2;
train_len = 1e2;
predict_len = 1000;

start_train_index = 1;
start_predict_index = 30000;
% start_predict_index = randi([4e4 8e4],1,1)

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
    label_index
    svmMdl = fitrsvm(train_input, train_label(:, label_index));
    predict_y(:, label_index) = predict(svmMdl, predict_y_input);
%     size(predict(svmMdl, y(start_predict_index-predict_len+label_index:start_predict_index-1+label_index)))
end
subplot(2, 1, 1);
plot(predict_y(1, :), 'r--')
hold on 
plot(test_y, 'b')
title({['data set num = ', num2str(data_set_num)]; ['train len = ', num2str(train_len)]; ['predict len = ', num2str(predict_len)]; ...
        ['start train index = ', num2str(start_train_index)]; ['start predict index = ', num2str(start_predict_index)]}');
subplot(2, 1, 2);
plot(abs(test_y - predict_y(1, :)')./test_y);
hold on 
title('relative error in %')
saveas(gcf, strcat('untitled_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));





