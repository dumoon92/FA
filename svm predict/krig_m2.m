close all
clear;
clc;
format compact;
%%
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;y_raw=data.Data;
x = x_raw; 
y = my_row_normalize(y_raw);
n=size(x,1);
train_len_set=[2, 4, 7, 10, 30, 50, 70, 100, 130, 170, 200,230, 270, 300];
% train_len_set = [30, 300];
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
    model = fitrgp(x_train, y_train);
    for predict_index = 1: predict_len
        if mod(predict_index, 200) == 0
            predict_index
        end
        
        new_x = x(start_predict+predict_index-1);
        new_y = y(start_predict+predict_index-1);
        x_y_predict_y(predict_index, :) = [new_x, new_y, predict(model, new_x)];
%         x_train = [x_train(2:end); new_x];
%         y_train = [y_train(2:end); new_y];
        x_train = x(start_predict+predict_index: start_predict+train_len-1+predict_index, :);
        y_train = y(start_predict+predict_index: start_predict+train_len-1+predict_index, :);
        model = fitrgp(x_train, y_train);
    end
%     x_y_predict_y = x_y_predict_y(10:end, :);
    rmse_matrix = [rmse_matrix, sum(abs(x_y_predict_y(:, 2) - x_y_predict_y(:, 3))./x_y_predict_y(:, 2))/size(x_y_predict_y, 1)];
    %% plot
    figure
    plot(1: length(x_y_predict_y(:,1)), x_y_predict_y(:,2), ...
         1: length(x_y_predict_y(:,1)), x_y_predict_y(:,3), '--');
    title(strcat('Kriging prediction vs real with train length=', num2str(train_len)));
    xlabel('Data point index');
    ylabel('Wave elevations');
    legend('Real data', 'Predict data');
    set(gcf, 'Units', 'inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    saveas(gcf, strcat('m2_Krig_',num2str(train_len),'_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
    close
end
save(strcat('m2_krig_data', '_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.mat'))
%% plot rmse-train_num
figure
plot(train_len_set, rmse_matrix, '-*');
% title(strcat('Krig Relative Error with different train length', num2str(train_len)));
xlabel('Train length');
ylabel('Relative error');
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('Krig', '_Relative_Error_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close

