function [results] = main_cnn_C1_MP1_FC2_speedup(TrnX, TrnY, TstX, TstY, params)
%%
%% input + conv1 + maxpool1 + fc2 + output

lr              = params.lr;
momentum        = params.momentum;
maxepoches      = params.maxepoches;
trnbatchsize    = params.trnbatchsize;
tstbatchsize    = params.tstbatchsize;

nTrn = size(TrnX, 1);
TrnYM = full(labvec2labmat(TrnY))'; % nTrn x c
nTst = size(TstX, 1);
TstYM = full(labvec2labmat(TstY))'; % nTst x c

%% mean normalization
TrnX_mean = mean(TrnX, 1);
TrnX = TrnX - repmat(TrnX_mean, nTrn, 1);
TstX = TstX - repmat(TrnX_mean, nTst, 1);
%% intensity/127.5-1.0
% TrnX = (TrnX*2)-1.0;
% TstX = (TstX*2)-1.0;
%% mean-std normalization
% TrnX_mean = mean(TrnX, 1);
% TrnX_std  = max(std(TrnX),eps);
% TrnX = bsxfun(@minus, TrnX, TrnX_mean);
% TrnX = bsxfun(@rdivide, TrnX, TrnX_std);
% TstX = bsxfun(@minus, TstX, TrnX_mean);
% TstX = bsxfun(@rdivide, TstX, TrnX_std);

%%
rndidx = randperm(nTrn);
TrnX = TrnX(rndidx, :);
TrnYM = TrnYM(:, rndidx);
TrnY = TrnY(rndidx);
ntrnbatches = floor(nTrn/trnbatchsize);
ntstbatches = floor(nTst/tstbatchsize);

%% initialization
H1 = params.H1;
W1 = params.W1;
B1 = params.B1; % number of channels
F1 = params.F1; % number of feature maps
h1 = params.h1;
w1 = params.w1;
OH1 = H1 - h1 + 1;
OW1 = W1 - w1 + 1;
k1      = normrnd(0, 0.01, [h1, w1, B1, F1]);
b1      = 0.1*ones(1, F1);
dk1_old = zeros(h1, w1, B1, F1);
db1_old = zeros(1, F1);

poolsize1 = params.poolsize1;
H2 = OH1/poolsize1;
W2 = OW1/poolsize1; 
n3 = H2 * W2 * F1;
n4 = size(TrnYM,1);
% normal distribution
Weight2 = normrnd(0,0.01,[n3, n4]); 
bias2 = 0.1*ones(n4, 1);
dWeight2_old = 0;
dbias2_old = 0;
loss_type = params.loss_type; %'cross-entropy';% '' or 'cross-entropy'

figure(3),
kid = 1;
for i=1:B1
    for j=1:F1
        subplot(B1,F1,kid),imshow(k1(:,:,i,j),[])
        kid = kid + 1;
    end
end
drawnow;

TrnLoss = zeros(1, maxepoches);
TrnAccs = zeros(1, maxepoches);
TstAccs = zeros(1, maxepoches);

for epoch = 1:maxepoches
    
    %% testing
    TstY_pred = [];
    for tstbatch = 1:ntstbatches 
        
        %% prepare mini-batch data
        s1 = (tstbatch-1)*tstbatchsize+1;
        s2 = tstbatch*tstbatchsize; 
        x = reshape(TstX(s1:s2, :), tstbatchsize, H1, W1, B1); %
        y = TstYM(:, s1:s2);
        
        %% fprop
        % fprop-I1
        N = tstbatchsize;
        a1 = permute(x, [2,3,4,1]);
        % fprop-C1
        a1_mat = zeros(OH1*OW1*N, h1*w1*B1);
        for i=1:N 
            for j=1:B1
                 a1_mat(((i-1)*OH1*OW1+1):(i*OH1*OW1), ((j-1)*h1*w1+1):(j*h1*w1))=im2col(a1(:,:, j, i), [h1 w1], 'sliding')'; % (OH1*OW1) x (h1*w1)
            end 
        end
        k1_mat = reshape(k1, h1*w1*B1, F1);
        zc1_mat = bsxfun(@plus, a1_mat * k1_mat, b1); % a1_mat: (OH1*OW1*N) x (h1*w1*B1), k1_mat: (h1*w1*B1) x (F1), b1: (1) x (F1)
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
        
        % I2
        a2 = reshape(ap1, n3, N);
        % FC2
        z2 = bsxfun(@plus, Weight2'*a2, bias2);  % n4 x N
        
        % Output
        if strcmp(loss_type,'euclidean')
            if 0     % sigmoid
                a3 = sigmoid(z2);
            elseif 0 % tanh
                a3 = tanh_opt(z2);
            elseif 0 % linear
                a3 = z2;
            elseif 1 % relu
                a3 = relu(z2);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a3 = softmax(z2);                       
        end
        TstY_pred = cat(2, TstY_pred, a3); 
    end
    
    if strcmp(loss_type, 'euclidean') 
        error = TstYM - TstY_pred;
        TrnLoss(epoch) = 0.5 * trace(error'*error) / nTrn;
    elseif strcmp(loss_type, 'cross-entropy')
        TrnLoss(epoch) = - sum(sum(TstYM .* log(TstY_pred))) / nTrn;
    end
    [~, TstY_pred] = max(TstY_pred, [], 1);
    TstAccs(epoch) = length(find(TstY_pred'==TstY))/nTst; 
    
    %% training
    TrnY_pred = [];
    for trnbatch = 1:ntrnbatches 
        s1 = (trnbatch-1)*trnbatchsize+1;
        s2 = trnbatch*trnbatchsize;
        
        x = reshape(TrnX(s1:s2, :), trnbatchsize, H1, W1, B1); %
        y = TrnYM(:, s1:s2);
        
        %% fprop
        % fprop-I1
        N = trnbatchsize;
        a1 = permute(x, [2,3,4,1]); % H1 x W1 x B1 x N
        % fprop-C1
        a1_mat = zeros(OH1*OW1*N, h1*w1*B1); % OH1 = H1 - h1 + 1, OW1 = W1 - w1 + 1
        for i=1:N 
            for j=1:B1
                 a1_mat(((i-1)*OH1*OW1+1):(i*OH1*OW1), ((j-1)*h1*w1+1):(j*h1*w1))=im2col(a1(:,:, j, i), [h1 w1], 'sliding')'; % (OH1*OW1) x (h1*w1)
            end 
        end
        k1_mat = reshape(k1, h1*w1*B1, F1);
        zc1_mat = bsxfun(@plus, a1_mat * k1_mat, b1); % a1_mat: (OH1*OW1*N) x (h1*w1*B1), k1_mat: (h1*w1*B1) x (F1), b1: (1) x (F1)
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
        [zp1, ix1] = MaxPooling(ac1, [poolsize1, poolsize1]); % H2 x W2 x F1 x N, H2 = OH1 / poolsize1, W2 = OW2 / poolsize1
        ap1 = zp1;
        
        % I2
        a2 = reshape(ap1, n3, N);  % n3 x N, n3 = H2 * W2 * F1
        % FC2
        z2 = bsxfun(@plus, Weight2'*a2, bias2);  % n4 x N
        
        % Output
        if strcmp(loss_type,'euclidean')
            if 0     % sigmoid
                a3 = sigmoid(z2);
            elseif 0 % tanh
                a3 = tanh_opt(z2);
            elseif 0 % linear
                a3 = z2;
            elseif 1 % relu
                a3 = relu(z2);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a3 = softmax(z2);                       
        end
        TrnY_pred = cat(2, TrnY_pred, a3); 
        
        %% bprop
        % bp-error
        if strcmp(loss_type,'euclidean')
            delta_a3 = -(y - a3);  
            if 0        % sigmoid
                delta_z3 = delta_a3 .* a3 .* (1 - a3); 
            elseif 0    % tanh
                delta_z3 = delta_a3 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a3.^2));
            elseif 0    % linear
                delta_z3 = delta_a3;
            elseif 1    % relu
                delta_z3 = delta_a3 .* double(a3 > 0);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            delta_z3 = -(y - a3);  % n4 x N   
        end 
        % bp-FC2-gradient
        dWeight2 = (a2 * delta_z3') / N; % n3 x n4
        dbias2 = sum(delta_z3, 2) / N; % 1 x n4
        % bp-FC2-error
        delta_ap1 = Weight2 * delta_z3;  % n3 x N
        delta_ap1 = reshape(delta_ap1, H2, W2, F1, N);
        
        % bp-MP1-error
        % bp-MP1-error-upsampling
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
        dk1_mat = a1_mat' * delta_zc1_mat;  
        dk1 = reshape(dk1_mat, h1, w1, B1, F1) / N;  
        db1 = sum(delta_zc1_mat, 1) / N;  
        
        
        %% update parameters  
        % for C1
        dk1 = momentum * dk1_old - lr * dk1; dk1_old = dk1;
        db1 = momentum * db1_old - lr * db1; db1_old = db1;
        % for FC2
        dWeight2 = momentum * dWeight2_old - lr * dWeight2; dWeight2_old = dWeight2;
        dbias2 = momentum * dbias2_old - lr * dbias2; dbias2_old = dbias2;
                
        % update weights 
        % for C1
        k1 = k1 + dk1;
        b1 = b1 + db1; 
        % for FC2
        Weight2 = Weight2 + dWeight2;
        bias2 = bias2 + dbias2;
        
    end   
    [~, TrnY_pred] = max(TrnY_pred, [], 1);
    TrnAccs(epoch) = length(find(TrnY_pred'==TrnY))/nTrn;
    fprintf('epoch=%02d,loss=%.6f, trn_acc=%.6f, tst_acc=%.6f\n', epoch, TrnLoss(epoch), TrnAccs(epoch), TstAccs(epoch));

    if mod(epoch, 20) ==0
       lr = lr / 20; 
    end
    
    figure(3),
    kid = 1;
    for i=1:B1
        for j=1:F1
            subplot(B1,F1,kid),imshow(k1(:,:,i,j),[])
            kid = kid + 1;
        end
    end
    drawnow;
end

results = [];
results.TrnLoss = TrnLoss;
results.TrnAccs = TrnAccs;
results.TstAccs = TstAccs;
results.k1      = k1;
results.b1      = b1;
results.Weight2      = Weight2;
results.bias2      = bias2; 

end


function a = sigmoid(z)
%% sigmoid function
% z: d x batchsize 
a = 1.0 ./ (1.0 + exp(-z));
end

function dz = sigmoid_d(z)
a = sigmoid(z);
dz = a .* (1 - a);
end

function a = tanh_opt(z)
%% hyperbolic tangent function
a = 1.7159 * tanh( 2/3 .* z);
end

function dz = tanh_opt_d(z)
%%  
a = tanh_opt(z);
dz = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a.^2);
end

function a = softmax(z)
%% softmax function
% z: d x batchsize 
z = exp(bsxfun(@minus, z, max(z, [], 1)));
a = bsxfun(@rdivide, z, sum(z, 1));
end

function a = relu(z)
%% relu function
a = max(0, z);
end