close all
kernel = 'gaussian';

load('data')

%% training
X = train(:,1:3);
z_vec = train(:, 4);

tic
[z_svm, svmMdl] = my_fitrsvm(X, z_vec, kernel);
err_svm = immse(z_vec, z_svm)
time_svm = toc

tic
[z_krig, krigMdl] = my_fitrkrig(X, z_vec);
err_krig = immse(z_vec, z_krig)
time_krig = toc

%% predict
X_test = test(:,1:3);
z_vec_test = test(:, 4);
z_predict_svm = predict(svmMdl, X_test);
err = immse(z_predict_svm, z_vec_test)

z_predict_krig = predict(krigMdl, X_test);
err = immse(z_predict_krig, z_vec_test)