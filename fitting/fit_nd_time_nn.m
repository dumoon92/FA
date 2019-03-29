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
figure
load(strcat('nn_dimension_', time, '.mat'), 'krig_rmse_record', 'krig_time_record', 'krig_kernel', 'svm_rmse_record', 'svm_time_record', 'svm_kernel');
semilogy(N_train, [NN_time_record; krig_time_record(4, :); svm_time_record(4, :)]);
legend('NN', 'krig ardsquaredexponential', 'svm polynomial');
title('Training Time of different dimensions')
xlabel('Dimension of Data')
ylabel('Training Time (Seconds)')
saveas(gcf, strcat('nn_dimension_', time, '.pdf'))
% legend(['NN', krig_kernel, svm_kernel])

%%
figure
semilogy(N_train, [NN_rmse_record; krig_rmse_record(4, :); svm_rmse_record(4, :)]);
hold on
legend('NN', 'krig ardsquaredexponential', 'svm polynomial');
title('Training RMSE of different dimensions')
xlabel('Dimension of Data')
ylabel('RMSE')
saveas(gcf, strcat('nn_dimension_', time, '.pdf'))

