
function dz = sigmoid_d(z)
a = sigmoid(z);
dz = a .* (1 - a);
end
