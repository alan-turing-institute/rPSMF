%% PSMF Convergence Experiment - Generate synthetic data
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%

clear,
clc,
close all,

r = 1;
d = 4;
n = 1000;
Q = 0.01 * eye(r);
R = 0.1 * eye(d);

% Generate data

Ctrue = 5 * abs(randn(d,r));
X0true = randn(r,1);

t = 1;
X(:,t) = X0true + chol(Q)' * randn(r,1);
Y(:,t) = Ctrue * X(:,t) + chol(R)' * randn(d,1);

for t = 2:n
    X(:,t) = X(:,t-1) + chol(Q)' * randn(r,1);
    Y(:,t) = Ctrue * X(:,t) + chol(R)' * randn(d,1);
end

save('data.mat')
