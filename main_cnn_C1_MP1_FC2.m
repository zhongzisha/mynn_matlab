function [results] = main_cnn_C1_MP1_FC2(TrnX, TrnY, TstX, TstY, params)
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
k1  = cell(B1, F1);
dk1 = cell(B1, F1);
b1  = cell(1,  F1);
db1 = cell(1,  F1);
k1_old = cell(B1, F1);
b1_old = cell(1,  F1);
dk1_old = cell(B1, F1);
db1_old = cell(1,  F1);
for i=1:B1
    for j=1:F1
        k1{i,j} = normrnd(0,0.01, [h1, w1]); 
        b1{1,j} = 0;
        k1_old{i,j} = normrnd(0,0.01, [h1, w1]); 
        b1_old{1,j} = 0;
        dk1_old{i,j} = 0;
        db1_old{j} = 0;
    end
end
poolsize1 = params.poolsize1;
n2 = (((H1-h1+1)/poolsize1)*((W1-w1+1)/poolsize1)) * F1;
n3 = size(TrnYM,1);
% normal distribution
W2 = normrnd(0,0.1,[n2, n3]); b2 = zeros(n3, 1);
dW2_old = 0;
db2_old = 0; 
loss_type = params.loss_type; %'cross-entropy';% '' or 'cross-entropy'
TrnLoss = zeros(1, maxepoches);
TrnAccs = zeros(1, maxepoches);
TstAccs = zeros(1, maxepoches);

for epoch = 1:maxepoches
    
    %% testing
    TstY_pred = [];
    for tstbatch = 1:ntstbatches 
        
        %% preparing mini-batch data
        s1 = (tstbatch-1)*tstbatchsize+1;
        s2 = tstbatch*tstbatchsize; 
        x = reshape(TstX(s1:s2, :), tstbatchsize, H1, W1, B1); % N x H x W x B
        y = TstYM(:, s1:s2);
        
        %% fprop
        % fp-I1
        a1 = cell(1, B1);
        for i=1:B1
           a1{i} = permute(x(:,:,:,i),[2,3,1]); % H x W x N
        end
        
        % fp-C1
        zc1 = cell(1, F1);
        for j=1:F1 
            zc1{j} = 0;
            for i=1:B1
                zc1{j} = zc1{j} + convn(a1{i}, k1{i,j}, 'valid');
            end
            zc1{j} = zc1{j} + b1{j};
        end
        
        % fp-C1-neuron
        ac1 = cell(1, F1);
        for j=1:F1 
            if 0    
                ac1{j} = sigmoid(zc1{j}); 
            elseif 0 
                ac1{j} = tanh_opt(zc1{j}); 
            elseif 1 
                ac1{j} = relu(zc1{j}); 
            end
        end
        
        % fp-MP1
        zp1 = cell(1, F1);
        ix1 = cell(1, F1);
        ap1 = cell(1, F1);
        for j=1:F1
            [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
            ap1{j} = zp1{j};
        end
        
        % fp-FC2
        a2 = [];
        for j=1:F1
           a2 = cat(1, a2, reshape(ap1{j}, size(ap1{j}, 1)*size(ap1{j}, 2), size(ap1{j},3))); 
        end 
        z2 = W2'*a2 + repmat(b2, 1, size(a2, 2));  
        
        % fp-Output
        if strcmp(loss_type,'euclidean')
            if 0    % sigmoid
                a3 = sigmoid(z2);
            elseif 0% tanh
                a3 = tanh_opt(z2);
            elseif 0% linear
                a3 = z2;
            elseif 1% relu
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
        
        %% preparing mini-batch data
        s1 = (trnbatch-1)*trnbatchsize+1;
        s2 = trnbatch*trnbatchsize; 
        x = reshape(TrnX(s1:s2, :), trnbatchsize, H1, W1, B1); % N x H x W x B
        y = TrnYM(:, s1:s2);
        
        %% fprop
        % fp-I1
        a1 = cell(1, B1);
        for i=1:B1
           a1{i} = permute(x(:,:,:,i),[2,3,1]); % H x W * batchsize
        end
        
        % fp-C1
        zc1 = cell(1, F1);
        for j=1:F1 
            zc1{j} = 0;
            for i=1:B1
                zc1{j} = zc1{j} + convn(a1{i}, k1{i,j}, 'valid');
            end
            zc1{j} = zc1{j} + b1{j};
        end
        
        % fp-C1-neuron
        ac1 = cell(1, F1);
        for j=1:F1 
            if 0    
                ac1{j} = sigmoid(zc1{j}); 
            elseif 0 
                ac1{j} = tanh_opt(zc1{j}); 
            elseif 1 
                ac1{j} = relu(zc1{j}); 
            end
        end
        
        % fp-MP1
        zp1 = cell(1, F1);
        ix1 = cell(1, F1);
        ap1 = cell(1, F1);
        for j=1:F1
            [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
            ap1{j} = zp1{j};
        end
        
        % fp-FC2
        a2 = [];
        for j=1:F1
           a2 = cat(1, a2, reshape(ap1{j}, size(ap1{j}, 1)*size(ap1{j}, 2), size(ap1{j},3))); 
        end 
        z2 = W2'*a2 + repmat(b2, 1, size(a2, 2));  
        
        % fp-Output
        if strcmp(loss_type,'euclidean')
            if 0    % sigmoid
                a3 = sigmoid(z2);
            elseif 0% tanh
                a3 = tanh_opt(z2);
            elseif 0% linear
                a3 = z2;
            elseif 1% relu
                a3 = relu(z2);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a3 = softmax(z2);                       
        end
        TrnY_pred = cat(2, TrnY_pred, a3); 
        
        %% bprop
        % bp-Output-Error
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
            delta_z3 = -(y - a3);                  
        end 
        
        % bp-FC2-gradient, bp-FC2-Error
        dW2 = (a2 * delta_z3') / size(a2, 2);
        db2 = mean(delta_z3, 2); 
        delta_ap1 = W2 * delta_z3;       
        if 0        % sigmoid
            delta_zp1 = delta_ap1; %delta_ap1 .* a2 .* (1 - a2);  
        elseif 0    % tanh
            delta_zp1 = delta_ap1; %delta_ap1 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a2.^2));
        elseif 1    % relu
            delta_zp1 = delta_ap1; %delta_ap1 .* double(a2 > 0);
        end
        % delta_a1 = W1 * delta_z2;
        % delta_z1 = delta_a1 .* a1 .* (ones(n1, size(a1, 2)) - a1);
        
        % bp for mp1 and compute error for c1-neuron
        % delta_z2 is the error sensitity of the layer after maxpooling, so at first, we should reshape them into the feature maps 
        delta_zp1 = permute(reshape(delta_zp1, (H1-h1+1)/poolsize1, (W1-w1+1)/poolsize1, F1, trnbatchsize), [1,2,4,3]);
        delta_ac1 = cell(1, F1);
        for j=1:F1
           delta_ac1{j} = zeros(H1-h1+1, W1-w1+1, trnbatchsize); 
           delta_ac1{j}(ix1{j}) = delta_zp1(:,:,:,j); 
        end
        
        % bp-C1-neuron
        delta_zc1 = cell(1, F1);
        for j=1:F1 
           if 0        % sigmoid
               delta_zc1{j} = delta_ac1{j}  .* ac1{j} .* (1 - ac1{j});
           elseif 0    % tanh
               delta_zc1{j} = delta_ac1{j}  .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ac1{j}.^2));
           elseif 1    % relu
               delta_zc1{j} = delta_ac1{j}  .* double(ac1{j} > 0);
           end
        end
        
        % bp-C1-gradient
        for j=1:F1
           for i=1:B1 
               dk1{i,j} = 0;
               for jj=1:trnbatchsize 
                   dk1{i,j} = dk1{i,j} + imrotate(conv2(a1{i}(:,:,jj), imrotate(delta_zc1{j}(:,:,jj), 180), 'valid'), 180);
               end 
               dk1{i,j} = dk1{i,j} / trnbatchsize;
           end
           db1{j} = sum(sum(reshape(delta_zc1{j}, size(delta_zc1{j},1)*size(delta_zc1{j},2), trnbatchsize), 1)) / trnbatchsize;
        end
        
        %% update parameters  
        % C1, MP1
        for j=1:F1
            for i=1:B1
                dk1{i,j} = momentum * dk1_old{i,j} - lr * dk1{i,j};
                dk1_old{i,j} = dk1{i,j};
            end
            db1{j} = momentum * db1_old{j} - lr * db1{j};
            db1_old{j} = db1{j};
        end
        % FC2
        dW2 = momentum * dW2_old - lr * dW2; dW2_old = dW2;
        db2 = momentum * db2_old - lr * db2; db2_old = db2;
        
        % C1, MP1
        for j=1:F1
            for i=1:B1
                k1{i,j} = k1{i,j} + dk1{i,j};
            end
            b1{j} = b1{j} + db1{j};
        end
        % FC2
        W2 = W2 + dW2;
        b2 = b2 + db2;
        
    end   
    [~, TrnY_pred] = max(TrnY_pred, [], 1);
    TrnAccs(epoch) = length(find(TrnY_pred'==TrnY))/nTrn;
    fprintf('epoch=%02d,loss=%.6f, trn_acc=%.6f, tst_acc=%.6f\n', epoch, TrnLoss(epoch), TrnAccs(epoch), TstAccs(epoch));
    
    % change learning rate
    if mod(epoch, 10) ==0
       lr = lr / 10; 
    end
    
end

results = [];
results.TrnLoss = TrnLoss;
results.TrnAccs = TrnAccs;
results.TstAccs = TstAccs;
results.k1      = k1;
results.b1      = b1; 
results.W2      = W2;
results.b2      = b2; 

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