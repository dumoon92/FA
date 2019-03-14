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
train_num=3e4;
next_num = 5;
predict_num = 1e3;

predict_x_y_raw_y = zeros(predict_num, 3);

x_train = x(1: train_num, :);
y_train = y(1: i+train_num, :);
model = libsvmtrain(y_train,x_train,'-s 4 -t 2 -c 2.2 -g 1 -h 0');
x_test = x(1+train_num,:);
y_test=y(1+train_num,:);
[py,mse,devalue] = libsvmpredict(y,x,model);

% for i = 1:next_num:predict_num
% %     x_train = x(i: i+train_num-1, :);
% %     y(i+train_num-1, :) = py;
% %     y_train = y(i: i+train_num-1, :);
% %     model = libsvmtrain(y_train,x_train,'-s 4 -t 2 -c 2.2 -g 2.8 -h 0');
%     
%     x_test = x(i+train_num: i+train_num+next_num-1,:);
%     y_test=y(i+train_num: i+train_num+next_num-1,:);
%     [py,mse,devalue] = libsvmpredict(y_test,x_test,model);
%     predict_x_y_raw_y(i:i+next_num-1,:) = [x_test, y_test, py];
% end
% % predict_x_y_raw_y = predict_x_y_raw_y(1:5e2, :);
% plot(predict_x_y_raw_y(:,1), predict_x_y_raw_y(:,2), '-o', predict_x_y_raw_y(:,1), predict_x_y_raw_y(:,3))
% for i=1:10000
%     x_train = x(i:i+train_num-1,:);
%     y_train=y(i:i+train_num-1,:);
%     model = libsvmtrain(y_train,x_train,'-s 4 -t 2 -c 2.2 -g 2.8 -h 0');
%     
%     x_test = x(i+train_num,:);
%     y_test=y(i+train_num,:);
%     [py,mse,devalue] = libsvmpredict(y_test,x_test,model);
%     
%     y_predict = [y_predict; py];
%     plot(x_test,y_test,'b');
%     hold on;
%     plot(x_test,py,'r');
%     hold on
% end





