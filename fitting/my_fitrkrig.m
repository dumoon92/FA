function [Y_krig, gprMdl] = my_fitrkrig(X, Y, krig_kernel)
% Kernel Function: exponential, squaredexpnential (default), usw
% @ https://www.mathworks.com/help/stats/fitrgp.html#butnn96-2

% calculate KernelParameters with variable phi
krig_kernel = char(krig_kernel);
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

gprMdl = fitrgp(X,Y,'KernelFunction', krig_kernel, 'KernelParameters',phi, 'SigmaLowerBound',0.2, 'Standardize', true);
% gprMdl = fitrgp(X,Y,'Basis','linear',...
%         'FitMethod','exact','PredictMethod','exact',...
%         'KernelFunction',krig_kernel, 'KernelParameters', phi);

%  About  'SigmaLowerBound',0.02
% if using default value, maximum number is 8900, or the error will be
% The Cholesky factor needed for making predictions cannot be computed. When calling fitrgp, try changing the initial values of
% 'KernelParameters' and 'Sigma'. Also consider setting 'Standardize' to true and increasing the value of 'SigmaLowerBound'.
% follow this https://de.mathworks.com/matlabcentral/answers/253605-kernel-custom-problem-firgp


Y_krig = resubPredict(gprMdl);