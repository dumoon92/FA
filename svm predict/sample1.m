

clear all;
clc;

format compact;
%%

data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat')
data=data.WG10_DHI
x=data.Time;y=data.Data;
n=size(x,1);
figure;
t=100;
for i=1:10000
    x_train = x(i:i+t-1,:);
    y_train=y(i:i+t-1,:);
    model = libsvmtrain(y_train,x_train,'-s 4 -t 2 -c 2.2 -g 2.8 -h 0');
    
    x_test = x(i+t,:);
    y_test=y(i+t,:);
    [py,mse,devalue] = libsvmpredict(y_test,x_test,model);
    
    plot(x_test,y_test,'b');
    hold on;
    plot(x_test,py,'r');
    hold on
end



