clear
%% training set
load('FairLeadMaxTensionTraining.mat')
X_train = data.Input;
Y_train = data.Output;
[X_train, Y_train] = remove_nan(X_train, Y_train);

%% testing set
load('FairLeadMaxTensionReference.mat')
X_test = cell2mat(data.Input);
Y_test = data.Output;
[X_test, Y_test] = remove_nan(X_test, Y_test);
X_test = normalize(X_test, 'range', [0,1]);
Y_test = normalize(Y_test, 'range', [0,1]);

%% Attention, Traing now!!!
N_test = length(Y_test);

tic
Mdl = fitrsvm(X_train,Y_train);
Y_SVM_predict = predict(Mdl, X_test);
time_SVM = toc

rmse_SVM = sqrt(immse(Y_test, Y_SVM_predict))/N_test/mean(Y_test)

tic
NN_Mdl = fitnet(20, 'trainlm');
surrogate = train(NN_Mdl, X_train', Y_train');
Y_NN_predict = (surrogate(X_test'))';
time_NN = toc
rmse_NN = sqrt(immse(Y_test, Y_NN_predict))/double(N_test)/mean(Y_test)   % compare generated fitted data and ideal data, averaged per point
