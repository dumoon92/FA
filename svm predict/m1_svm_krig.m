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
predict_len = 300;

%% kriging, train, then predict
train_len = 7e2;
start_train = 1;
start_predict = start_train+train_len;

x_train = x(start_train: start_train+train_len-1, :);
y_train = y(start_train: start_train+train_len-1, :);
model = fitrgp(x_train,y_train);
y_train_pre = predict(model,x_train);

x_test = x(start_predict: start_predict+predict_len-1,:);
y_test = y(start_predict: start_predict+predict_len-1,:);
py = predict(model, x_test);
predict_x_y_pre_y = [x_test, y_test, py];

figure('units','normalized','outerposition',[0 0 0.3 0.45])  % output graph as full screen
plot(1:length(x_train), y_train, ...
     length(x_train):length(x_train)+length(predict_x_y_pre_y(:,1))-1, predict_x_y_pre_y(:,2), ...
     length(x_train):length(x_train)+length(predict_x_y_pre_y(:,1))-1, predict_x_y_pre_y(:,3), '--');
hold on
plot(1:length(x_train), y_train_pre, '--');
% set a colored square block
area(1:length(x_train), ones(length(x_train), 1),'FaceColor','b','FaceAlpha',.15,'EdgeAlpha',.3)
area(length(x_train):length(x_train)+length(predict_x_y_pre_y(:,1))-1, ones(length(predict_x_y_pre_y(:,1)), 1),'FaceColor','r','FaceAlpha',.15,'EdgeAlpha',.3)

title('Wave prediction vs real in kriging');
xlabel('Data point index');
ylabel('Wave elevation');
legend('real train data', 'predict train data', 'real test data', 'predict test data');
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m1_krig_predict_no_update_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close

%% svm, train, then predict
train_len = 700;
start_train = 1;
start_predict = start_train+train_len;

x_train = x(start_train: start_train+train_len-1, :)*1e3;  % amplify to make svm work better
y_train = y(start_train: start_train+train_len-1, :);
% y_train = sin(x_train);  % use sin function to test
model = fitrsvm(x_train,y_train, 'KernelFunction','polynomial');
y_train_pre = predict(model,x_train);

x_test = x(start_predict: start_predict+predict_len-1,:)*1e3;
y_test = y(start_predict: start_predict+predict_len-1,:);
py = predict(model, x_test);
predict_x_y_pre_y = [x_test, y_test, py];

figure('units','normalized','outerposition',[0 0 0.3 0.45])  % output graph as full screen
plot(1:length(x_train), y_train, ...
     length(x_train):length(x_train)+length(predict_x_y_pre_y(:,1))-1, predict_x_y_pre_y(:,2), ...
     length(x_train):length(x_train)+length(predict_x_y_pre_y(:,1))-1, predict_x_y_pre_y(:,3), '--');
hold on
plot(1:length(x_train), y_train_pre, '--');
% set a colored square block
area(1:length(x_train), ones(length(x_train), 1),'FaceColor','b','FaceAlpha',.15,'EdgeAlpha',.3)
area(length(x_train):length(x_train)+length(predict_x_y_pre_y(:,1))-1, ones(length(predict_x_y_pre_y(:,1)), 1),'FaceColor','r','FaceAlpha',.15,'EdgeAlpha',.3)

title('Wave prediction vs real in SVM');
xlabel('Data point index');
ylabel('Wave elevation');
legend('real train data', 'predict train data', 'real test data', 'predict test data');
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m1_svm_predict_no_update_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close








