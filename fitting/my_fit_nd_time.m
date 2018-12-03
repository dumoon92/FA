%% subfunction of fit_nd_time
function [time_svm, rmse_svm, svmMdl, time_krig, rmse_krig, krigMdl] = my_fit_nd_time(N_train, N_test, considered_variable, svm_kernel, krig_kernel)
%% generate data
% N_train = 1000;
% N_test = 10;
% considered_variable =  ["S_w", "W_fw", "A", "Lambda", "q", "lambda", "tC", "N_z", "W_dg", "W_p"];
% considered_variable = [];

[X_train, W_train] = my_generate_10_variable_function(N_train, considered_variable);
[X_test, W_test] = my_generate_10_variable_function(N_test, considered_variable);

%% training and testing
%  use normalized RMSE with the definition: NRMSD = RMSD/(y_max-y_min)

 tic
[~, svmMdl] = my_fitrsvm(X_train, W_train, svm_kernel);  % training model, svmMdl is model
time_svm = toc;  % training time
W_svm_predict = predict(svmMdl, X_test);  % using model generate fitted data
rmse_svm = sqrt(immse(W_test, W_svm_predict))/double(N_test)/mean(W_test);  % compare generated fitted data and ideal data, averaged per point
 
 tic
[~, krigMdl] = my_fitrkrig(X_train, W_train, krig_kernel);  % training model,  krigMdl is model
time_krig = toc;  % training time
W_krig_predict = predict(krigMdl, X_test);  % using model generate fitted data
rmse_krig = sqrt(immse(W_test, W_krig_predict))/double(N_test)/mean(W_test);   % compare generated fitted data and ideal data, averaged per point

f = 0;
