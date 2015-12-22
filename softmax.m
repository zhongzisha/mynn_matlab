
function a = softmax(z)
%% softmax function
% z: d x batchsize 
z = exp(bsxfun(@minus, z, max(z, [], 1)));
a = bsxfun(@rdivide, z, sum(z, 1));
end