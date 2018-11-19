close all
clear

kernel = 'gaussian';

x = -2:0.25:2; 
y = -2:0.25:2;
z = cos(x)'*sin(y);

mesh(x,y,z)
xlabel('x axis'); ylabel('y axis'); zlabel(' axis');
title('surface z = cos(x)sin(y)');

[x_vec, y_vec] = meshgrid(x', y');
x_length = length(x);
y_length = length(y);
x_vec = reshape(x_vec, [], 1);
y_vec = reshape(y_vec, [], 1);
z_vec = reshape(z, [], 1);

z_svm = my_fitrsvm([x_vec, y_vec], z_vec, kernel);
z_svm = reshape(z_svm, x_length, y_length);
err_svm = immse(z, z_svm)

z_krig = my_fitrkrig([x_vec, y_vec], z_vec);
z_krig = reshape(z_krig, x_length, y_length);
err_krig = immse(z, z_krig)

figure
mesh(x,y,z_svm)
xlabel('x axis'); ylabel('y axis'); zlabel(' axis');
title('SVM fitting surface z = cos(x)sin(y)');

figure
mesh(x,y,z_krig)
xlabel('x axis'); ylabel('y axis'); zlabel(' axis');
title('Krig fitting surface z = cos(x)sin(y)');
