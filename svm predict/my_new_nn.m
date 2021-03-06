function [test_y, predict_y, error] = my_new_nn(y_raw, data_set_num, train_len, predict_len, start_train_index, start_predict_index)
%% use front y as x, following y as y, no relation of time
y = my_row_normalize(y_raw);
% y = sin(2*pi*linspace(0, 5*pi, 1e3))';  % use sinus function as test

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
    
    NN_Mdl = fitnet(20, 'trainlm');
    surrogate = train(NN_Mdl,train_input, train_label(:, label_index), 'useGPU','no');
    predict_y(:, label_index) = surrogate(predict_y_input);
end
figure
subplot(2, 1, 1);
plot(predict_y(1, :), 'r--')
hold on 
plot(test_y, 'b')
title({['data set num = ', num2str(data_set_num)]; ['train len = ', num2str(train_len)]; ['predict len = ', num2str(predict_len)]; ...
        ['start train index = ', num2str(start_train_index)]; ['start predict index = ', num2str(start_predict_index)]}');
subplot(2, 1, 2);
error = sum(abs(test_y - predict_y(1, :)')./test_y)/numel(test_y);
plot(abs(test_y - predict_y(1, :)')./test_y);
hold on 
title('relative error in %')
saveas(gcf, strcat('new_krig_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));





