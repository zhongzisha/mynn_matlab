
function a = sigmoid(z)
%% sigmoid function
% z: d x batchsize 
a = 1.0 ./ (1.0 + exp(-z));
end