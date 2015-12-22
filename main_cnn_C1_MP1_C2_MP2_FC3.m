function [results] = main_cnn_C1_MP1_C2_MP2_FC3(TrnX, TrnY, TstX, TstY, params)
%%
%% input + conv1 + maxpool1 + conv2 + maxpool2 + fc3 + output

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
k1      = cell(B1, F1);
dk1     = cell(B1, F1);
b1      = cell(1,  F1);
db1     = cell(1,  F1);
k1_old  = cell(B1, F1);
b1_old  = cell(1,  F1);
dk1_old = cell(B1, F1);
db1_old = cell(1,  F1);
for i=1:B1
    for j=1:F1
        k1{i,j} = normrnd(0,0.1, [h1, w1]); 
        b1{1,j} = 0;
        k1_old{i,j} = normrnd(0,0.1, [h1, w1]); 
        b1_old{1,j} = 0;
        dk1_old{i,j} = 0;
        db1_old{j} = 0;
    end
end
poolsize1 = params.poolsize1;
H2 = (H1-h1+1)/poolsize1;
W2 = (W1-w1+1)/poolsize1;
B2 = F1;
F2 = params.F2;
h2 = params.h2;
w2 = params.w2;
k2      = cell(B2, F2);
dk2     = cell(B2, F2);
b2      = cell(1,  F2);
db2     = cell(1,  F2);
k2_old  = cell(B2, F2);
b2_old  = cell(1,  F2);
dk2_old = cell(B2, F2);
db2_old = cell(1,  F2);
for i=1:B2
    for j=1:F2
        k2{i,j} = normrnd(0, 0.1, [h2, w2]);
        b2{1,j} = 0;
        k2_old{i,j} = normrnd(0, 0.1, [h2, w2]);
        b2_old{1,j} = 0;
        dk2_old{i,j} = 0;
        db2_old{1,j} = 0;
    end
end
poolsize2 = params.poolsize2;
% for FC1
n3 = (((H2-h2+1)/poolsize2)*((W2-w2+1)/poolsize2)) * F2;
n4 = size(TrnYM,1);
% normal distribution
W3 = normrnd(0,0.1,[n3, n4]); b3 = zeros(n4, 1);
dW3_old = 0;
db3_old = 0;
loss_type = params.loss_type; %'cross-entropy';% '' or 'cross-entropy'

TrnLoss = zeros(1, maxepoches);
TrnAccs = zeros(1, maxepoches);
TstAccs = zeros(1, maxepoches);

fig1 = figure;
fig2 = figure;
fig3 = figure;
fig4 = figure;
% for C1
figure(3),
kid = 1;
for i=1:B1
    for j=1:F1
        subplot(B1,F1,kid),imshow(k1{i,j},[])
        kid = kid + 1;
    end
end
drawnow;
% for C2
figure(4),
kid = 1;
for i=1:B2
    for j=1:F2
        subplot(B2,F2,kid),imshow(k2{i,j},[])
        kid = kid + 1;
    end
end
drawnow;

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
        a1 = cell(1, B1);
        for i=1:B1
           a1{i} = permute(x(:,:,:,i),[2,3,1]); % H1 x W1 * batchsize
        end 
        % fprop-C1
        zc1 = cell(1, F1);
        for j=1:F1 
            zc1{j} = 0;
            for i=1:B1
                zc1{j} = zc1{j} + convn(a1{i}, k1{i,j}, 'valid');
            end
            zc1{j} = zc1{j} + b1{j};
        end 
        % fprop-C1-neuron
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
        % fprop-MP1
        zp1 = cell(1, F1);
        ix1 = cell(1, F1);
        ap1 = cell(1, F1);
        for j=1:F1
            [zp1{j}, ix1{j}] = MaxPooling(ac1{j}, [poolsize1, poolsize1]);
            ap1{j} = zp1{j};
        end
        
        
        % fprop-I2
        a2 = cell(1, B2);
        for i=1:B2
           a2{i} = ap1{i}; 
        end
        % fprop-C2
        zc2 = cell(1, F2);
        for j=1:F2
           zc2{j} = 0;
           for i=1:B2
               zc2{j} = zc2{j} + convn(a2{i}, k2{i,j}, 'valid');
           end
           zc2{j} = zc2{j} + b2{j};
        end
        % fprop-C2-neuron
        ac2 = cell(1, F2);
        for j=1:F2
           if 0
               ac2{j} = sigmoid(zc2{j}); 
           elseif 0
               ac2{j} = tanh_opt(zc2{j}); 
           elseif 1
               ac2{j} = relu(zc2{j}); 
           end
        end
        % fprop-MP2 
        zp2 = cell(1, F2);
        ix2 = cell(1, F2);
        ap2 = cell(1, F2);
        for j=1:F2
            [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
            ap2{j} = zp2{j};
        end
        
        % I3
        a3 = [];
        for j=1:F2
           a3 = cat(1, a3, reshape(ap2{j}, size(ap2{j}, 1)*size(ap2{j}, 2), size(ap2{j},3))); 
        end 
        % FC3
        z4 = W3'*a3 + repmat(b3, 1, size(a3, 2));  
        
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
        
        x = reshape(TrnX(s1:s2, :), trnbatchsize, H1, W1, B1); %
        y = TrnYM(:, s1:s2);
        
        %% fprop
        % fp-I1
        a1 = cell(1, B1);
        for i=1:B1
           a1{i} = permute(x(:,:,:,i),[2,3,1]); % H1 x W1 * batchsize
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
        
        
        % fp-I2
        a2 = cell(1, B2);
        for i=1:B2
           a2{i} = ap1{i}; 
        end
        % fp-C2
        zc2 = cell(1, F2);
        for j=1:F2
           zc2{j} = 0;
           for i=1:B2
               zc2{j} = zc2{j} + convn(a2{i}, k2{i,j}, 'valid');
           end
           zc2{j} = zc2{j} + b2{j};
        end
        % fp-C2-neuron
        ac2 = cell(1, F2);
        for j=1:F2
           if 0
               ac2{j} = sigmoid(zc2{j}); 
           elseif 0
               ac2{j} = tanh_opt(zc2{j}); 
           elseif 1
               ac2{j} = relu(zc2{j}); 
           end
        end
        % fp-MP2 
        zp2 = cell(1, F2);
        ix2 = cell(1, F2);
        ap2 = cell(1, F2);
        for j=1:F2
            [zp2{j}, ix2{j}] = MaxPooling(ac2{j}, [poolsize2, poolsize2]);
            ap2{j} = zp2{j};
        end
        
        % fp-I3
        a3 = [];
        for j=1:F2
           a3 = cat(1, a3, reshape(ap2{j}, size(ap2{j}, 1)*size(ap2{j}, 2), size(ap2{j},3))); 
        end 
        % fp-FC3
        z4 = W3'*a3 + repmat(b3, 1, size(a3, 2));  
        
        % fp-Output
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
        TrnY_pred = cat(2, TrnY_pred, a4); 
        
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
        dW3 = (a3 * delta_z4') / size(a3, 2);
        db3 = mean(delta_z4, 2);
        % bp-FC3-error
        delta_ap2 = W3 * delta_z4;  
        
        % bp-MP2-error
        delta_zp2 = permute(reshape(delta_ap2, (H2-h2+1)/poolsize2, (W2-w2+1)/poolsize2, F2, trnbatchsize), [1,2,4,3]);
        % bp-MP2-error-upsampling
        delta_ac2 = cell(1, F2);
        for j=1:F2 
           delta_ac2{j} = zeros(H2-h2+1, W2-w2+1, trnbatchsize); 
           delta_ac2{j}(ix2{j}) = delta_zp2(:,:,:,j);
        end
        
        % bp-C2-neuron-error
        delta_zc2 = cell(1, F2);
        for j=1:F2
           if 0        % sigmoid
               delta_zc2{j} = delta_ac2{j}  .* ac2{j} .* (1 - ac2{j});
           elseif 0    % tanh
               delta_zc2{j} = delta_ac2{j}  .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ac2{j}.^2));
           elseif 1    % relu
               delta_zc2{j} = delta_ac2{j}  .* double(ac2{j} > 0);
           end
        end
        % bp-C2-gradient
        for j=1:F2
           for i=1:B2 
               dk2{i,j} = 0;
               for jj=1:trnbatchsize 
                   dk2{i,j} = dk2{i,j} + imrotate(conv2(a2{i}(:,:,jj), imrotate(delta_zc2{j}(:,:,jj), 180), 'valid'), 180);
               end
               dk2{i,j} = dk2{i,j}/trnbatchsize;
           end
           db2{j} = sum(sum(reshape(delta_zc2{j}, size(delta_zc2{j},1)*size(delta_zc2{j},2), trnbatchsize), 1))/trnbatchsize;
        end
        
        % bp-MP1-error(backpropagation by conv kernels)
        delta_a2 = cell(1, B2);
        for i=1:B2
           delta_a2{i} = 0;
           for j=1:F2
               delta_a2{i} = delta_a2{i} + convn(delta_zc2{j}, imrotate(k2{i,j}, 180), 'full');
           end
        end
        delta_ap1 = delta_a2;
        % bp-MP1-error-upsampling
        delta_zp1 = cell(1, F1);
        delta_ac1 = cell(1, F1);
        for j=1:F1
            delta_zp1{j} = delta_ap1{j};
            delta_ac1{j} = zeros(H1-h1+1, W1-w1+1, trnbatchsize);
            delta_ac1{j}(ix1{j})=delta_zp1{j};
        end
        
        % bp-C1-neuron-error
        delta_zc1 = cell(1, F1);
        for j=1:F1 
            if 0        % sigmoid
                delta_zc1{j} = delta_ac1{j} .* ac1{j} .* (1 - ac1{j});
            elseif 0    % tanh
                delta_zc1{j} = delta_ac1{j} .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * ac1{j}.^2));
            elseif 1    % relu
                delta_zc1{j} = delta_ac1{j} .* double(ac1{j} > 0);
            end
        end
        % bp-C1-gradient
        for j=1:F1
            for i=1:B1
                dk1{i,j} = 0;
                for jj=1:trnbatchsize
                    dk1{i,j} = dk1{i,j} + imrotate(conv2(a1{i}(:,:,jj), imrotate(delta_zc1{j}(:,:,jj), 180), 'valid'), 180);
                end
                dk1{i,j} = dk1{i,j}/trnbatchsize;
            end
            db1{j} = sum(sum(reshape(delta_zc1{j}, size(delta_zc1{j},1)*size(delta_zc1{j},2), trnbatchsize), 1))/trnbatchsize;
        end
        
        
        %% compute update parameters  
        
        % C1, MP1
        for j=1:F1
            for i=1:B1
                dk1{i,j} = momentum * dk1_old{i,j} - lr * dk1{i,j};
                dk1_old{i,j} = dk1{i,j};
            end
            db1{j} = momentum * db1_old{j} - lr * db1{j};
            db1_old{j} = db1{j};
        end
        % C2, MP2
        for j=1:F2
            for i=1:B2
                dk2{i,j} = momentum * dk2_old{i,j} - lr * dk2{i,j};
                dk2_old{i,j} = dk2{i,j};
            end
            db2{j} = momentum * db2_old{j} - lr * db2{j};
            db2_old{j} = db2{j};
        end
        % FC3
        dW3 = momentum * dW3_old - lr * dW3; dW3_old = dW3;
        db3 = momentum * db3_old - lr * db3; db3_old = db3;
        
        %%
        for j=1:F1
            for i=1:B1
                k1{i,j} = k1{i,j} + dk1{i,j};
            end
            b1{j} = b1{j} + db1{j};
        end
        for j=1:F2
            for i=1:B2
                k2{i,j} = k2{i,j} + dk2{i,j};
            end
            b2{j} = b2{j} + db2{j};
        end
        W3 = W3 + dW3;
        b3 = b3 + db3;
        
    end   
    [~, TrnY_pred] = max(TrnY_pred, [], 1);
    TrnAccs(epoch) = length(find(TrnY_pred'==TrnY))/nTrn;
    fprintf('epoch=%02d,loss=%.6f, trn_acc=%.6f, tst_acc=%.6f\n', epoch, TrnLoss(epoch), TrnAccs(epoch), TstAccs(epoch));
    % plot training loss
    figure(1), plot(epoch, TrnLoss(epoch), 'r.'); hold on; drawnow
    % plot train accuracy and test accuracy
    figure(2), plot(epoch, TrnAccs(epoch), 'r.'); hold on; plot(epoch, TstAccs(epoch), 'b.'); hold on; drawnow
    % plot the C1 kernels
    figure(3), 
    kid = 1;
    for i=1:B1
        for j=1:F1
           subplot(B1,F1,kid),imshow(k1{i,j},[])
           kid = kid + 1;
       end
    end
    drawnow;
    % plot the C2 kernels
    figure(4), 
    kid = 1;
    for i=1:B2
        for j=1:F2
           subplot(B2,F2,kid),imshow(k2{i,j},[])
           kid = kid + 1;
       end
    end
    drawnow;
    
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
results.k2      = k2;
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