function [results] = main_nn_FC1_FC2(TrnX, TrnY, TstX, TstY, params)
%%
%% input + fc1(W1, b1) + fc2(W2, b2) + fc3(W3, b3) + output(sigmoid+euclideanloss, softmax+cross-entropyloss)

lr              = params.lr;            % learning rate
momentum        = params.momentum;      % momentum term
maxepoches      = params.maxepoches;    % number of epoches
trnbatchsize    = params.trnbatchsize;  % batchsize for trn set
tstbatchsize    = params.tstbatchsize;  % batchsize for tst set

[nTrn, n1] = size(TrnX);
TrnYM = full(labvec2labmat(TrnY))'; % nTrn x c
[nTst, n1] = size(TstX);
TstYM = full(labvec2labmat(TstY))'; % nTst x c

%% mean normalization
% TrnX_mean = mean(TrnX, 1);
% TrnX = TrnX - repmat(TrnX_mean, nTrn, 1);
% TstX = TstX - repmat(TrnX_mean, nTst, 1);
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
TrnX = TrnX(rndidx, :)';
TrnYM = TrnYM(:, rndidx);
TrnY = TrnY(rndidx);
TstX = TstX';
ntrnbatches = floor(nTrn/trnbatchsize);
ntstbatches = floor(nTst/tstbatchsize);

%% initialization
n2 = params.n2;             % number of hidden nodes in second layer
n3 = params.n3;             % number of hidden nodes in third layer
n4 = size(TrnYM,1);
% normal distribution
W1 = normrnd(0,0.1,[n1, n2]); b1 = zeros(n2, 1);
W2 = normrnd(0,0.1,[n2, n3]); b2 = zeros(n3, 1);
W3 = normrnd(0,0.1,[n3, n4]); b3 = zeros(n4, 1);
dW3_old = 0;
db3_old = 0;
dW2_old = 0;
db2_old = 0;
dW1_old = 0;
db1_old = 0;

loss_type = params.loss_type; %'cross-entropy';% '' or 'cross-entropy'
%% training
% lr = 0.00001;
% momentum = 0.0;
% maxepoches = 20;
TrnLoss = zeros(1, maxepoches);
TrnAccs = zeros(1, maxepoches);
TstAccs = zeros(1, maxepoches);
for epoch = 1:maxepoches
    
    %% testing
    TstY_pred = [];
    for tstbatch = 1:ntstbatches 
        s1 = (tstbatch-1)*tstbatchsize+1;
        s2 = tstbatch*tstbatchsize; 
        x = TstX(:, s1:s2); %
        y = TstYM(:, s1:s2);
        
        % fprop
        a1 = x;                                     
        z1 = W1'*a1 + repmat(b1, 1, size(a1, 2));  
        if 0        % sigmoid
            a2 = sigmoid(z1);
        elseif 0    % tanh
            a2 = tanh_opt(z1);
        elseif 1    % relu
            a2 = relu(z1);
        end
        z2 = W2'*a2 + repmat(b2, 1, size(a2, 2)); 
        if 0        % sigmoid
            a3 = sigmoid(z2);
        elseif 0    % tanh
           a3 = tanh_opt(z2); 
        elseif 1    % relu
            a3 = relu(z2);
        end
        z3 = W3'*a3 + repmat(b3, 1, size(a3, 2));   
        if strcmp(loss_type,'euclidean')
            if 0    % sigmoid
                a4 = sigmoid(z3);
            elseif 0% tanh
                a4 = tanh_opt(z3);
            elseif 1% relu
                a4 = relu(z3);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a4 = softmax(z3);                       
        end
        TstY_pred = cat(2, TstY_pred, a4); 
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
        
        x = TrnX(:, s1:s2); %
        y = TrnYM(:, s1:s2);
        
        % fprop
        a1 = x;                                    
        z1 = W1'*a1 + repmat(b1, 1, size(a1, 2));   
        if 0            % sigmoid
            a2 = sigmoid(z1);
        elseif 0        % tanh
           a2 = tanh_opt(z1); 
        elseif 1        % relu
            a2 = relu(z1);
        end
        z2 = W2'*a2 + repmat(b2, 1, size(a2, 2));   
        if 0            % sigmoid
            a3 = sigmoid(z2);
        elseif 0        % tanh
           a3 = tanh_opt(z2); 
        elseif 1        % relu
            a3 = relu(z2);
        end
        z3 = W3'*a3 + repmat(b3, 1, size(a3, 2));   
        if strcmp(loss_type,'euclidean')
            if 0        % sigmoid
                a4 = sigmoid(z3);
            elseif 0    % tanh
               a4 = tanh_opt(z3); 
            elseif 1    % relu
                a4 = relu(z3);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            a4 = softmax(z3);                       
        end
        TrnY_pred = cat(2, TrnY_pred, a4); 
        
        % bprop
        if strcmp(loss_type,'euclidean')
            delta_a4 = -(y - a4);  
            if 0        % sigmoid
                delta_z4 = delta_a4 .* a4 .* (1 - a4); 
            elseif 0    % tanh
                delta_z4 = delta_a4 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a4.^2));
            elseif 1    % relu
                delta_z4 = delta_a4 .* double(a4 > 0);
            end
        elseif strcmp(loss_type, 'cross-entropy')
            delta_z4 = -(y - a4);                  
        end
        delta_a3 = W3 * delta_z4;       
        if 0        % sigmoid
            delta_z3 = delta_a3 .* a3 .* (1 - a3);  
        elseif 0    % tanh
            delta_z3 = delta_a3 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a3.^2));
        elseif 1    % relu
            delta_z3 = delta_a3 .* double(a3 > 0);
        end
        delta_a2 = W2 * delta_z3; 
        if 0        % sigmoid
            delta_z2 = delta_a2 .* a2 .* (1 - a2);  
        elseif 0    % tanh
            delta_z2 = delta_a2 .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a2.^2));
        elseif 1    % relu
            delta_z2 = delta_a2 .* double(a2 > 0);
        end
        % delta_a1 = W1 * delta_z2;
        % delta_z1 = delta_a1 .* a1 .* (ones(n1, size(a1, 2)) - a1);
        
        %% update parameters
        % compute gradient increments
        dW3 = (a3 * delta_z4') / size(a3, 2); 
        db3 = mean(delta_z4, 2);
        dW2 = (a2 * delta_z3') / size(a2, 2);
        db2 = mean(delta_z3, 2);
        dW1 = (a1 * delta_z2') / size(a1, 2);
        db1 = mean(delta_z2, 2);
        
        dW3 = momentum * dW3_old - lr * dW3; dW3_old = dW3;
        db3 = momentum * db3_old - lr * db3; db3_old = db3;
        
        dW2 = momentum * dW2_old - lr * dW2; dW2_old = dW2;
        db2 = momentum * db2_old - lr * db2; db2_old = db2;
        
        dW1 = momentum * dW1_old - lr * dW1; dW1_old = dW1;
        db1 = momentum * db1_old - lr * db1; db1_old = db1; 
        
        % update weights
        W3 = W3 + dW3;
        b3 = b3 + db3;
        W2 = W2 + dW2;
        b2 = b2 + db2;
        W1 = W1 + dW1;
        b1 = b1 + db1;
        
    end   
    [~, TrnY_pred] = max(TrnY_pred, [], 1);
    TrnAccs(epoch) = length(find(TrnY_pred'==TrnY))/nTrn;
    fprintf('epoch=%02d,loss=%.6f, trn_acc=%.6f, tst_acc=%.6f\n', epoch, TrnLoss(epoch), TrnAccs(epoch), TstAccs(epoch));
    figure(1), plot(epoch, TrnLoss(epoch), 'r.'); hold on; drawnow
    figure(2), plot(epoch, TrnAccs(epoch), 'r.'); hold on; plot(epoch, TstAccs(epoch), 'b.'); hold on; drawnow
    
    if mod(epoch, 10) ==0
       lr = lr / 10; 
    end
    
end

results = [];
results.TrnLoss = TrnLoss;
results.TrnAccs = TrnAccs;
results.TstAccs = TstAccs;
results.W1      = W1;
results.b1      = b1;
results.W2      = W2;
results.b2      = b2;
results.W3      = W3;
results.b3      = b3;


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