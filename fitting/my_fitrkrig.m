function [Y_krig, gprMdl] = my_fitrkrig(X, Y)
gprMdl = fitrgp(X,Y,'Basis','linear',...
      'FitMethod','exact','PredictMethod','exact');
Y_krig = resubPredict(gprMdl);