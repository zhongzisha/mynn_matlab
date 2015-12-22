clear;
N = 10;
H = 5;
W = 5;
C = 3;
M = 2;
K = 3;
x = rand(H, W, C, N); %  N samples, C channels
k = rand(K, K, C, M); %  M filters

%% y = correlate(x,k,'valid'), (H-K+1) x (W-K+1) x M x N

%% the following is 
a1 = cell(1,C);
for i=1:C
    a1{i} = squeeze(x(:,:,i,:)); % H x W x N
end
k1 = cell(C, M);
for i=1:C
    for j=1:M
        k1{i,j} = squeeze(k(:,:,i,j));
    end
end
zc1 = cell(1, M);
for j=1:M
    zc1{j} = 0;
    for i=1:C
       zc1{j} = zc1{j} + convn(a1{i}, imrotate(k1{i,j},180), 'valid'); 
    end
end


%% the following is image correlation using matrix multiplication
% im2col
xx = [];
for i=1:N
   xxx=[];
   for j=1:C
      xxx=cat(1, xxx, im2col(x(:,:,j,i),[K,K])); 
   end
   xx = cat(2, xx, xxx);
end
kk1 = [];
for i=1:M
   kk11=[];
   for j=1:C
      kk11 = cat(1, kk11, reshape(k(:,:,j,i),K*K,1)); 
   end
   kk1 = cat(2, kk1, kk11);
end

y1 = xx'*kk1;
y = reshape(y1, H-K+1, W-K+1, M, N);


