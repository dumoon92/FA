%% generate 4-d data as fitting data source
%% train data
x1 = -2:0.25:2; 
x2 = -2:0.25:2;
x3 = -2:0.25:2;

[x1_vec, x2_vec, x3_vec] = meshgrid(x1', x2', x3');
x1_vec = reshape(x1_vec, [], 1);
x2_vec = reshape(x2_vec, [], 1);
x3_vec = reshape(x3_vec, [], 1);

z_vec = cos(x1_vec).*sin(x2_vec).*exp(x3_vec); %     function

train = [x1_vec, x2_vec, x3_vec, z_vec];

%% test data
x1 = -2:0.05:2; 
x2 = -2:0.05:2;
x3 = -2:0.05:2;

[x1_vec, x2_vec, x3_vec] = meshgrid(x1', x2', x3');
x1_vec = reshape(x1_vec, [], 1);
x2_vec = reshape(x2_vec, [], 1);
x3_vec = reshape(x3_vec, [], 1);

z_vec = cos(x1_vec).*sin(x2_vec).*exp(x3_vec); 

test = [x1_vec, x2_vec, x3_vec, z_vec];
