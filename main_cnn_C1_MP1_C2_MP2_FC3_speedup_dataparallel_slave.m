
function [pred, dk1, db1, dk2, db2, dWeight3, dbias3] = main_cnn_C1_MP1_C2_MP2_FC3_speedup_dataparallel_slave(x,y,params, k1, b1, k2, b2, Weight3, bias3)

% for C1-MP1
H1 = params.H1;
W1 = params.W1;
B1 = params.B1; % number of channels
F1 = params.F1; % number of feature maps
h1 = params.h1;
w1 = params.w1;
OH1 = H1 - h1 + 1;
OW1 = W1 - w1 + 1;
poolsize1 = params.poolsize1;
% for C2-MP2
H2 = OH1/poolsize1;
W2 = OW1/poolsize1;
B2 = F1;
F2 = params.F2;
h2 = params.h2;
w2 = params.w2;
OH2 = H2 - h2 + 1;
OW2 = W2 - w2 + 1;
poolsize2 = params.poolsize2;
H3 = OH2 / poolsize2;
W3 = OW2 / poolsize2;
% for FC3
n3 = (H3 * W3) * F2;
n4 = size(y, 1); 
loss_type = params.loss_type; %'cross-entropy';% '' or 'cross-entropy'

%% fprop
% fprop-I1
N = size(x,1);
a1 = permute(x, [2,3,4,1]);
% fprop-C1
a1_mat = zeros(OH1*OW1*N, h1*w1*B1);
for i=1:N
    for j=1:B1
        a1_mat(((i-1)*OH1*OW1+1):(i*OH1*OW1), ((j-1)*h1*w1+1):(j*h1*w1))=im2col(a1(:,:, j, i), [h1 w1], 'sliding')'; % (OH1*OW1) x (h1*w1)
    end
end
k1_mat = reshape(k1, h1*w1*B1, F1);
zc1_mat = bsxfun(@plus, a1_mat * k1_mat, b1); % x1_mat: (OH1*OW1*N) x (h1*w1*B1), k1_mat: (h1*w1*B1) x (F1), b1: (1) x (F1)
zc1 = permute(reshape(zc1_mat, OH1, OW1, N, F1), [1 2 4 3]); % [OH1, OW1, F1, N]

% fprop-C1-neuron
if 0
    ac1 = sigmoid(zc1);
elseif 0
    ac1 = tanh_opt(zc1);
elseif 1
    ac1 = relu(zc1);
end

% fprop-MP1
[zp1, ix1] = MaxPooling(ac1, [poolsize1, poolsize1]);
ap1 = zp1;

% fprop-I2
a2 = ap1; % H2 x W2 x F1 x N
% fprop-C2
a2_mat = zeros(OH2*OW2*N, h2*w2*B2);
for i=1:N
    for j=1:F1
        a2_mat(((i-1)*OH2*OW2+1):(i*OH2*OW2), ((j-1)*h2*w2+1):(j*h2*w2)) = im2col(a2(:,:, j, i), [h2 w2], 'sliding')';
    end
end
k2_mat = reshape(k2, h2*w2*B2, F2);
zc2_mat = bsxfun(@plus, a2_mat * k2_mat, b2); % a2_mat: (OH2*OW2*N) x (h2*w2*B2), k2_mat: (h2*w2*B2) x (F2), b2: (1) x (F2)
zc2 = permute(reshape(zc2_mat, OH2, OW2, N, F2), [1 2 4 3]); % [OH2, OW2, F2, N]
% fprop-C2-neuron
if 0
    ac2 = sigmoid(zc2);
elseif 0
    ac2 = tanh_opt(zc2);
elseif 1
    ac2 = relu(zc2);
end
% fprop-MP2
[zp2, ix2] = MaxPooling(ac2, [poolsize2, poolsize2]);
ap2 = zp2;

% I3
a3 = reshape(ap2, n3, N);
% FC3
z4 = bsxfun(@plus, Weight3'*a3, bias3);  % n4 x N

% Output
if strcmp(loss_type,'euclidean')
    if 0     % sigmoid
        a4 = sigmoid(z4);
    elseif 0 % tanh
        a4 = tanh_opt(z4);
    elseif 0 % linear
        a4 = z4;
    elseif 1 % relu
        a4 = relu(z4);
    end
elseif strcmp(loss_type, 'cross-entropy')
    a4 = softmax(z4);
end
pred = a4;

%% bprop
% bp-error
if strcmp(loss_type,'euclidean')
    delta_a4 = -(y - a4);
    if 0        % sigmoid
        delta_z4 = delta_a4 .* a4 .* (1 - a4);
    elseif 0    % tanh
        delta_z4 = delta_a4 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a4.^2));
    elseif 0    % linear
        delta_z4 = delta_a4;
    elseif 1    % relu
        delta_z4 = delta_a4 .* double(a4 > 0);
    end
elseif strcmp(loss_type, 'cross-entropy')
    delta_z4 = -(y - a4);
end
% bp-FC3-gradient
dWeight3 = (a3 * delta_z4');
dbias3 = sum(delta_z4, 2);
% bp-FC3-error
delta_ap2 = Weight3 * delta_z4;

% bp-MP2-error
% bp-MP2-error-upsampling
delta_ac2 = zeros(OH2, OW2, F2, N);
delta_ac2(ix2) = delta_ap2;

% bp-C2-neuron-error
if 0        % sigmoid
    delta_zc2 = delta_ac2 .* ac2 .* (1 - ac2);
elseif 0    % tanh
    delta_zc2 = delta_ac2 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ac2.^2));
elseif 1    % relu
    delta_zc2 = delta_ac2 .* double(ac2 > 0);
end

% bp-C2-gradient
delta_zc2_mat = reshape(permute(delta_zc2,[1,2,4,3]), OH2*OW2*N, F2);
dk2_mat = a2_mat' * delta_zc2_mat; % (h2*w2*B2, F2)
dk2 = reshape(dk2_mat, h2, w2, B2, F2); % h2 x w2 x B2 x F2
db2 = sum(delta_zc2_mat, 1); % (1, F2)

% bp-MP1-error(backpropagation by conv kernels)
delta_a2_mat = delta_zc2_mat * k2_mat'; % (OH2*OW2*N, h2*w2*B2)
delta_a2 = zeros(H2, W2, B2, N);
for i=1:N
    for j=1:B2
        delta_a2(:,:,j,i) = mergePatch(delta_a2_mat(((i-1)*OH2*OW2+1):(i*OH2*OW2), ((j-1)*h2*w2+1):(j*h2*w2)), h2, w2, H2, W2);
    end
end

delta_ap1 = delta_a2;
% bp-MP1-error-upsampling
% delta_zp1 = permute(reshape(delta_ap1, H2, W2, F1, N), [1,2,4,3]);
delta_ac1 = zeros(OH1, OW1, F1, N);
delta_ac1(ix1) = delta_ap1;

% bp-C1-neuron-error
if 0        % sigmoid
    delta_zc1 = delta_ac1 .* ac1 .* (1 - ac1);
elseif 0    % tanh
    delta_zc1 = delta_ac1 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ac1.^2));
elseif 1    % relu
    delta_zc1 = delta_ac1 .* double(ac1 > 0);
end
% bp-C1-gradient
delta_zc1_mat = reshape(permute(delta_zc1,[1,2,4,3]), OH1*OW1*N, F1);
dk1_mat = a1_mat' * delta_zc1_mat; % (h1*w1*B1, F1)
dk1 = reshape(dk1_mat, h1, w1, B1, F1); % h1 x w1 x B1 x F1
db1 = sum(delta_zc1_mat, 1); % (1, F1)

end