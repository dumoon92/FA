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
train_num=1000;
predict_interval = 1;
predict_len = 1e3;

predict_x_y_raw_y = zeros(predict_len, 3);

%% train, then predict
train_len = 3e3;
start_train = 1;
start_predict = start_train+train_len;

x_train = x(start_train: start_train+train_len-1, :);
y_train = y(start_train: start_train+train_len-1, :);
model = fitrgp(x_train,y_train);
y_train_pre = predict(model,x_train);

x_test = x(start_predict: start_predict+predict_len-1,:);
y_test = y(start_predict: start_predict+predict_len-1,:);
py = predict(model, x_test);
predict_x_y_raw_y(start_train:start_train+predict_len-1,:) = [x_test, y_test, py];

figure
plot(predict_x_y_raw_y(:,1), predict_x_y_raw_y(:,2), ...
     predict_x_y_raw_y(:,1), predict_x_y_raw_y(:,3), '--');
title('Krig prediction vs real');
xlabel('data points');
ylabel('wave hight(normalized)');
legend('real data', 'predict data');
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('old_krig_predict_no_update_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close


%% use the new recieved point to make next prediction
for i = 1:predict_interval:predict_len
    x_train = x(i: i+train_num-1, :);
    y_train = y(i+train_num: i+train_num, :);
    model = libsvmtrain(y_train,x_train,'-s 4 -t 2 -c 2.2 -g 2.8 -h 0');
    
    x_test = x(i+train_num: i+train_num+predict_interval-1,:);
    y_test = y(i+train_num: i+train_num+predict_interval-1,:);
    [py,mse,devalue] = libsvmpredict(y_test,x_test,model);
    predict_x_y_raw_y(i:i+predict_interval-1,:) = [x_test, y_test, py];
end

for i = 1: predict_len:
    x_train = 

% predict_x_y_raw_y = predict_x_y_raw_y(1:5e2, :);
figure
plot(predict_x_y_raw_y(:,1), predict_x_y_raw_y(:,2), predict_x_y_raw_y(:,1), predict_x_y_raw_y(:,3), '--')
title('Krig prediction vs real');
xlabel('data points');
ylabel('wave hight(normalized)');
legend('real data', 'predict data');
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('old_svm_update_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close





