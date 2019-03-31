clear
close all 

N_train = int64(1e4);
N_test = int64(900);

considered_variable = {'N_z', 'A', 'tC', 'S_w', 'W_dg', 'W_p', 'lambda',  'Lambda', 'q', 'W_fw'};
N = numel(considered_variable);

NN_time_record = zeros(1, N);
NN_rmse_record = zeros(1, N);

for i = 1:numel(considered_variable)
    considered_variable(1: i)
    
    [X, Y] = my_generate_10_variable_function(N_train, considered_variable(1: i));
    [X_test, Y_test] = my_generate_10_variable_function(N_test, considered_variable(1: i));
    tic
    disp(tic)
    NN_Mdl = fitnet(20, 'trainlm');
    surrogate = train(NN_Mdl, X', Y', 'useGPU','yes');
    Y_NN_predict = (surrogate(X_test'))';
    disp(toc)
    NN_time_record(i) = toc;
    NN_rmse_record(i) = sqrt(immse(Y_test, Y_NN_predict))/double(N_test)/mean(Y_test);   % compare generated fitted data and ideal data, averaged per point
end




%%
time = regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''});
save(strcat('NN_record', time, '.mat'), 'NN_time_record', 'NN_rmse_record');

figure('units','normalized','outerposition',[0 0 1 1])  % output graph as full screen
load('10d_rmse_time_record_dimension-kernel_22-Mar-2019_16_15_33', 'krig_rmse_record', 'krig_time_record', 'krig_kernel', 'svm_rmse_record', 'svm_time_record', 'svm_kernel');
semilogy(1:10, [NN_time_record; krig_time_record(4, :); svm_time_record(4, :)]);
legend('NN', 'krig ardsquaredexponential', 'svm polynomial');
title('Training Time of different dimensions')
xlabel('Dimension of Data')
ylabel('Training Time (Seconds)')
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('nn_dimension_time_', time, '.pdf'))

%%
figure('units','normalized','outerposition',[0 0 1 1])  % output graph as full screen
semilogy(1:10, [NN_rmse_record; krig_rmse_record(4, :); svm_rmse_record(4, :)]);
hold on
legend('NN', 'krig ardsquaredexponential', 'svm polynomial');
title('Training RMSE of different dimensions')
xlabel('Dimension of Data')
ylabel('RMSE')
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

saveas(gcf, strcat('nn_dimension_rmse_', time, '.pdf'))

