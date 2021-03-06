function [test_y, predict_y, error, rmse] = my_new_svm(y_raw, data_set_num, train_len, predict_len, start_train_index, start_predict_index, kernel)
%% m3, use front y as x, following y as y, no relation of time
kernel
kernel = char(kernel);
y = my_row_normalize(y_raw);
% y = sin(2*pi*linspace(0, 5*pi, 1e3))';  % use sinus function as test
parameter_str = strcat('-', num2str(data_set_num),...
                       '-', num2str(train_len),...
                       '-', num2str(predict_len),...
                       '-', num2str(start_train_index),...
                       '-', num2str(start_predict_index),...
                       '_');
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
    if mod(label_index, 25) == 0
        label_index
    end
    svmMdl = fitrsvm(train_input, train_label(:, label_index), 'KernelFunction', kernel);
    predict_y(:, label_index) = predict(svmMdl, predict_y_input);
%     size(predict(svmMdl, y(start_predict_index-predict_len+label_index:start_predict_index-1+label_index)))
end
figure('units','normalized','outerposition',[0 0 0.35 .45])  % output graph as full screen
plot(predict_y(1, :), 'r--')
hold on 
grid on
plot(test_y, 'b')
xlabel('Data point index');
ylabel('Wave elevation');
legend('Predict', 'Real');
% title({['kernel = ', kernel, '    data set number = ', num2str(data_set_num), '    train length = ', num2str(train_len)]; ['predict length = ', num2str(predict_len), ...
%         '    start train index = ', num2str(start_train_index), '    start predict index = ', num2str(start_predict_index)]}');
title('Wave elevation of SVM under different inputs')
saveas(gcf, strcat('m2_svm_single_', regexprep(datestr(now,'dd-mm-yyyy HH:MM:SS FFF'), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
close
figure('units','normalized','outerposition',[0 0 0.35 .45])  % output graph as full screen
error = sum(abs(test_y - predict_y(1, :)')./test_y)/numel(test_y);
rmse = abs(immse(test_y, predict_y(1, :)'))^0.5;
plot(abs(test_y - predict_y(1, :)')./test_y);
hold on 
xlabel('Data point index');
ylabel('Average ralative error');
title('Average relative error of SVM under different inputs')
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m2_svm_single_error_', regexprep(datestr(now,'dd-mm-yyyy HH:MM:SS FFF'), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
close





