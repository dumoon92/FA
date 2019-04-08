clear
close all 

N = 20;
N_train = int64(linspace(1000, 1e4, N));  % training number of points from xxx to xxx, devided by N parts
N_test = int64(linspace(100, 890, N));  % testing number of points from xxx to xxx, devided by N parts
considered_variable = {'N_z', 'A', 'tC', 'S_w', 'W_dg', 'W_p', 'lambda',  'Lambda', 'q', 'W_fw'};





NN_time_record = zeros(1, N);
NN_rmse_record = zeros(1, N);
for i = 1:N
    [X, Y] = my_generate_10_variable_function(N_train(i), considered_variable);
    [X_test, Y_test] = my_generate_10_variable_function(N_test(i), considered_variable);
    tic
    NN_Mdl = fitnet(20, 'trainlm');
    surrogate = train(NN_Mdl, X', Y', 'useGPU','yes');
    Y_NN_predict = (surrogate(X_test'))';
    NN_time_record(i) = toc;
    NN_rmse_record(i) = sqrt(immse(Y_test, Y_NN_predict))/double(N_test(i))/mean(Y_test);   % compare generated fitted data and ideal data, averaged per point
end
save('NN_record.mat', 'NN_time_record', 'NN_rmse_record');

%%
figure('units','normalized','outerposition',[0 0 0.3 0.45])  % output graph as full screen
load('10d_rmse_time_record_N20.mat', 'krig_rmse_record', 'krig_time_record', 'krig_kernel', 'svm_rmse_record', 'svm_time_record', 'svm_kernel');
load('NN_record.mat')
semilogy(linspace(1000, 1e4, 20), [NN_time_record; krig_time_record(4, :); svm_time_record(4, :)]);
hold on
grid on
legend('NN', 'krig ardsquaredexponential', 'svm polynomial');
title('Training time of three different models in different data numbers')
xlabel('Number of data')
ylabel('Training time (Seconds)')

set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, 'NNvsKrigvsSVM_time_record.pdf')
% legend(['NN', krig_kernel, svm_kernel])

%%
figure('units','normalized','outerposition',[0 0 0.3 0.45])  % output graph as full screen
semilogy(linspace(1000, 1e4, 20), [NN_rmse_record; krig_rmse_record(4, :); svm_rmse_record(4, :)]);
hold on
grid on
legend('NN', 'krig ardsquaredexponential', 'svm polynomial');
title('RMSE of three different models in different data numbers')
xlabel('Number of data')
ylabel('RMSE')

set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, 'NNvsKrigvsSVM_rmse_record.pdf')

