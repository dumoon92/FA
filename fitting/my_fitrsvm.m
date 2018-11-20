function [Y_svm, svmMdl] = my_fitrsvm(X, Y, kernel)
% Kernel Function: gaussian (default) or rbf, linear, polynomial
% @ https://www.mathworks.com/help/stats/fitrsvm.html
svmMdl = fitrsvm(X,Y, 'KernelFunction', kernel);
Y_svm= resubPredict(svmMdl);