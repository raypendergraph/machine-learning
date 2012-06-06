function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
fprintf("Theta1 %i x %i\n",size(Theta1))
Theta2_grad = zeros(size(Theta2));
fprintf("Theta2 %i x %i\n",size(Theta2))

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% forward propagate
%a_1 =    % dont forget to add bias unit column! should be (5000 x 401)
%z_2 =    % (5000 x 25)
%a_2 =    % (5000 x 26)
%a_2 = ones(m, 
%z_3 =    % (5000 x 10)
%a_3 =    % = H_theta (5000 x 10)


%a_1 = ones(m, size( X, 2 ) + 1);
%a_1(:,2:end) = X;

%z_2 = a_1 * Theta1';
%fprintf("z_2 %i x %i\n",size(z_2))
%a_2 = ones(m, size( Theta1, 1 ) + 1 );
%fprintf("a_2 %i x %i\n",size(a_2))
%a_2(:,2:end) = sigmoid(z_2);

%z_3 = a_2 * Theta2';
%a_3 = ones(m, size( Theta2,1 ) + 1);
%a_3(:,2:end) = sigmoid(z_3);
%fprintf("a_3 %i x %i\n",size(a_3))

%You need these matrices for both cost function and backprop.

% unroll y
%y=   % calculate y (5000 x 10) (can be done with repmat and ==)

% cost function
%J=  % (1 x 1) calculate in matrix form then use a double sum

% back propagate
%delta_3= % (5000 x 10)
% calculate delta_2 in 2 parts (because you have to remove the bias column)
%delta_2= % (5000 x 26)
%delta_2= % (5000 x 25)
%Theta1_grad= % (25 x 401)
%Theta2_grad= % (10 x 26)
    
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


yy = zeros(size(y),num_labels);
for i=1:size(X)
  yy(i,y(i)) = 1;
end
 
X = [ones(m,1) X];
% cost
for  i=1:m
  a1 = X(i,:);
  z2 = Theta1*a1';
  a2 = sigmoid(z2);
  z3 = Theta2*[1; a2];
  a3 = sigmoid(z3);
 
  J += -yy(i,:)*log(a3)-(1-yy(i,:))*log(1-a3);
end
 
J /= m;
 
J += (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
 
t=1;
for t=1:m
  % forward pass
  a1 = X(t,:);
  z2 = Theta1*a1';
  a2 = [1; sigmoid(z2)];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
 
  % backprop
  delta3 = a3-yy(t,:)';
  delta2 = (Theta2'*delta3).*[1; sigmoidGradient(z2)];
  delta2 = delta2(2:end);
 
  Theta1_grad = Theta1_grad + delta2*a1;
  Theta2_grad = Theta2_grad + delta3*a2';
end
 
Theta1_grad = (1/m)*Theta1_grad+(lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Theta2_grad+(lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
 















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
