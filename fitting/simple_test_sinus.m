svm_kernel = 'gaussian';
t = [0:0.1:2*pi]';
y = sin(t);
test_t = [2*pi:0.1:3*pi]';
[predict_train_data, svmMdl] = my_fitrsvm(t, y, svm_kernel);
predict_data = predict(svmMdl, test_t);
figure
plot(t, y, t, predict_train_data, test_t, predict_data, test_t, sin(test_t));

type = 'function estimation';
% [gam, sig2] = tunelssvm({t,y,type,[],[],'RBF_kernel'},'simplex',...
%     'leaveoneoutlssvm',{'mse'});
% [alpha,b] = trainlssvm({t,y,type,gam,sig2,'RBF_kernel'});

gam = 10; sig2 = 0.2; kernel = 'RBF_kernel';
model = initlssvm(t,y,type,gam,sig2,kernel);
model = trainlssvm(model);
predict_data = simlssvm(model, test_t);
plotlssvm(model);
% plotlssvm({t,y,type,gam,sig2,'RBF_kernel'},{alpha,b});