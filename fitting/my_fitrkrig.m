function [Y_krig, gprMdl] = my_fitrkrig(X, Y, krig_kernel)
% Kernel Function: exponential, squaredexpnential (default), usw
% @ https://www.mathworks.com/help/stats/fitrgp.html#butnn96-2

% calculate KernelParameters with variable phi
switch krig_kernel
    case {'exponential' , 'squaredexponential' , 'matern32' , 'matern52'}
        phi = [mean(std(X)),std(Y)/sqrt(2)];
    case 'rationalquadratic'
        phi = [mean(std(X));1;std(Y)/sqrt(2)];
    case {'ardexponential' , 'ardsquaredexponential' , 'ardmatern32' , 'ardmatern52'}
        phi = [std(X)';std(Y)/sqrt(2)];
    case 'ardrationalquadratic'
        phi = [std(X)';1;std(Y)/sqrt(2)];
end

gprMdl = fitrgp(X,Y,'KernelFunction','squaredexponential','KernelParameters',phi);
% gprMdl = fitrgp(X,Y,'Basis','linear',...
%         'FitMethod','exact','PredictMethod','exact',...
%         'KernelFunction',krig_kernel, 'KernelParameters', phi);


Y_krig = resubPredict(gprMdl);