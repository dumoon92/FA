function [Y_svm, svmMdl] = my_fitrsvm(X, Y, kernel)
svmMdl = fitrsvm(X,Y, 'KernelFunction', kernel);
Y_svm= resubPredict(svmMdl);