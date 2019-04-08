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
train_len_set=[2, 4, 7, 10, 30, 50, 70, 100, 130, 170, 200,230, 270, 300];
% train_len_set = [9, 20, 200];
rmse_matrix = [];

for train_len = train_len_set
    train_len
    predict_len = 1e3;
    start_train = 1;
    start_predict = start_train+train_len;
    x_y_predict_y = zeros(predict_len, 3);

    %% train, get 1 value, repredict
    x_train = x(start_train: start_train+train_len-1, :);
    y_train = y(start_train: start_train+train_len-1, :);
    model = fitrsvm(x_train, y_train);
    for predict_index = 1: predict_len
        if mod(predict_index, 200) == 0
            predict_index
        end

        x_y_predict_y(predict_index, :) = [x(start_predict+predict_index-1), ...
            y(start_predict+predict_index-1), predict(model, x(start_predict+predict_index-1))];
        x_train = x(start_predict+predict_index: start_predict+train_len-1+predict_index, :);
        y_train = y(start_predict+predict_index: start_predict+train_len-1+predict_index, :);
        model = fitrsvm(x_train, y_train);
    end
    rmse_matrix = [rmse_matrix, sum(abs(x_y_predict_y(:, 2) - x_y_predict_y(:, 3))./x_y_predict_y(:, 2))/size(x_y_predict_y, 1)];
    %% plot
    figure('units','normalized','outerposition',[0 0 .35 .45])  % output graph as full screen
    plot(1: numel(x_y_predict_y(:,1)), x_y_predict_y(:,2), ...
         1: numel(x_y_predict_y(:,1)), x_y_predict_y(:,3), '--');
    title(strcat('SVM prediction vs real with train length=', num2str(train_len)));
    xlabel('Data Point Index');
    ylabel('Wave hight(normalized)');
    legend('Real data', 'Predict data');
    set(gcf, 'Units', 'inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    saveas(gcf, strcat('m2_SVM_predict_with_update-',num2str(train_len),'_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
    close
end
save(strcat('m2_SVM_data', '_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.mat'))
%% plot rmse-train_num
figure('units','normalized','outerposition',[0 0 .35 .45])  % output graph as full screen
plot(train_len_set, rmse_matrix, '-*');
title(strcat('SVM Relative Error with different train length', num2str(train_len)));
xlabel('Train length');
ylabel('Relative error');
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m2_SVM_Relative_Error_train_len', '_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close

