close all
clear;
clc;
format compact;
%%
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;y_raw=data.Data;
x = my_row_normalize(x_raw); y = my_row_normalize(y_raw);
n=size(x,1);
%% parameters
kernel = 'rbf';
data_set_num = 300;
train_len = 100; 
predict_len = 300;
start_train = 666;
start_predict = 30000;
parameter_str = strcat('-', num2str(data_set_num),...
                       '-', num2str(train_len),...
                       '-', num2str(predict_len),...
                       '-', num2str(start_train),...
                       '-', num2str(start_predict),...
                       '_');

rmse_matrix = [];
% start_predict = start_train+train_len;
x_y_predict_y = zeros(predict_len, 3);

%% train, get 1 value, repredict
x_train = x(start_train: start_train+train_len-1, :);
y_train = y(start_train: start_train+train_len-1, :);
model = fitrsvm(x_train, y_train, 'KernelFunction', kernel);
relative_error = zeros(predict_len, 1);
for predict_index = 1: predict_len
    if mod(predict_index, 200) == 0
        predict_index
    end

    x_y_predict_y(predict_index, :) = [x(start_predict+predict_index-1), ...
        y(start_predict+predict_index-1), predict(model, x(start_predict+predict_index-1))];
    x_train = x(start_predict+predict_index: start_predict+train_len-1+predict_index, :);
    y_train = y(start_predict+predict_index: start_predict+train_len-1+predict_index, :);
    relative_error(predict_index, 1) = sum(abs(x_y_predict_y(1:predict_index, 2) - x_y_predict_y(1:predict_index, 3))...
        ./x_y_predict_y(1:predict_index, 2))/predict_index;
    model = fitrsvm(x_train, y_train, 'KernelFunction', kernel);
end
rmse_matrix = [rmse_matrix, sum(abs(x_y_predict_y(:, 2) - x_y_predict_y(:, 3))./x_y_predict_y(:, 2))/size(x_y_predict_y, 1)];
%% save mat
save(strcat('m2_SVM_krig_single', '_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.mat'))
%% plot predictions
figure('units','normalized','outerposition',[0 0 .35 .45])  % output graph as full screen
plot(1: numel(x_y_predict_y(:,1)), x_y_predict_y(:,2), ...
     1: numel(x_y_predict_y(:,1)), x_y_predict_y(:,3), '--');
title(strcat('SVM prediction vs real with train length=', num2str(train_len)));
xlabel('Data Point Index');
ylabel('Wave elevation');
legend('Real data', 'Predict data');
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m2_svm_single_',parameter_str, num2str(train_len),'_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close

%% plot relative error
figure('units','normalized','outerposition',[0 0 0.35 .45])  % output graph as full screen
plot(relative_error);
hold on 
grid minor
xlabel('Data point index');
ylabel('Average ralative error');
% title('relative error')

set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m2_svm_single_error_', parameter_str, regexprep(datestr(now,'dd-mm-yyyy HH:MM:SS FFF'), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
close

