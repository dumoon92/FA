clear
%% subfunction of fit_nd_time
% function [time_svm, rmse_svm, svmMdl, time_krig, rmse_krig, krigMdl] = my_fit_nd_time(N_train, N_test, considered_variable, svm_kernel, krig_kernel)
% %% generate data
% % N_train = 1000;
% % N_test = 10;
% % considered_variable =  ["S_w", "W_fw", "A", "Lambda", "q", "lambda", "tC", "N_z", "W_dg", "W_p"];
% % considered_variable = [];
% 
% [X_train, W_train] = my_generate_10_variable_function(N_train, considered_variable);
% [X_test, Y_test] = my_generate_10_variable_function(N_test, considered_variable);
% 
load('FairLeadMaxTensionTraining.mat')
X_train = data.Input;
Y_train = data.Output;
[X_train, Y_train] = my_remove_nan(X_train, Y_train);

load('FairLeadMaxTensionReference.mat')
if iscell(data.Input)
    X_test = cell2mat(data.Input);
else
    X_test = data.Input;
end 
Y_test = data.Output;
[X_test, Y_test] = my_remove_nan(X_test, Y_test);
X_test = my_row_normalize(X_test);
Y_test = my_row_normalize(Y_test);

svm_kernel = 'gaussian';
krig_kernel = 'squaredexponential';
N_test = length(Y_test);
%% training and testing
%  use normalized RMSE with the definition: NRMSD = RMSD/(y_max-y_min)

tic
[~, svmMdl] = my_fitrsvm(X_train, Y_train, svm_kernel);  % training model, svmMdl is model
time_svm = toc  % training time
Y_svm_predict = predict(svmMdl, X_test);  % using model generate fitted data
rmse_svm = sqrt(immse(Y_test, Y_svm_predict))/double(N_test)/mean(Y_test)  % compare generated fitted data and ideal data, averaged per point
 
 tic
[~, krigMdl] = my_fitrkrig(X_train, Y_train, krig_kernel);  % training model,  krigMdl is model
time_krig = toc  % training time
Y_krig_predict = predict(krigMdl, X_test);  % using model generate fitted data
rmse_krig = sqrt(immse(Y_test, Y_krig_predict))/double(N_test)/mean(Y_test)   % compare generated fitted data and ideal data, averaged per point

%% plot
% figure
% k_moving_average = 100;
% plot(Y_test);
% hold on
% grid minor
% plot(Y_svm_predict)
% plot(Y_krig_predict)
% legend('Ref', 'SVM', 'Krig')

tic
NN_Mdl = fitnet(20, 'trainlm');
surrogate = train(NN_Mdl, X_train', Y_train', 'useGPU','yes');
Y_NN_predict = (surrogate(X_test'))';
time_NN = toc
rmse_NN = sqrt(immse(Y_test, Y_NN_predict))/double(N_test)/mean(Y_test)   % compare generated fitted data and ideal data, averaged per point

