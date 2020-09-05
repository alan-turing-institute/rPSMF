%% PSMF Changepoint Detection Experiment - PSMF algorithm
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%
function [X] = PSMF(r,Y,Q,A,R,H,V,P0,C,X,m,n)

P = zeros(2 * r, 2 * r,n);
PP = P;

t = 1;
X0 = chol(Q)' * randn(2 * r,1);
Xp(:,t) = A * X0;
PP(:,:,t) = A * P0 * A' + Q;

barR = R + (H * Xp(:,t))' * V * (H * Xp(:,t));
S = C * (H * PP(:,:,t) * H') * C' + barR;
X(:,t) = Xp(:,t) + (PP(:,:,t) * H' * C' / S) * (Y(:,t) - C * H * Xp(:,t));
P(:,:,t) = PP(:,:,t) - PP(:,:,t) * H' * C' / S * C * H * PP(:,:,t);

eta_k = trace(C * (H * PP(:,:,t) * H') * C' + R) / m;
Nt =  Xp(:,t)' * H' * V * H * Xp(:,t) + eta_k;

C = C + ((Y(:,t) - C * H * Xp(:,t)) * Xp(:,t)' * H' * V) / (Nt);
V = V - (V * H * (Xp(:,t) * Xp(:,t)') * H' * V) / (Nt);

for t = 2:length(Y)

    Xp(:,t) = A * X(:,t-1);
    PP(:,:,t) = A * P(:,:,t) * A' + Q;

    barR = R + (H * Xp(:,t))' * V * (H * Xp(:,t));
    S = C * (H * PP(:,:,t) * H') * C' + barR;
    X(:,t) = Xp(:,t) + (PP(:,:,t) * H' * C' / S) * (Y(:,t) - C * H * Xp(:,t));
    P(:,:,t) = PP(:,:,t) - PP(:,:,t) * H' * C' / S * C * H * PP(:,:,t);

    eta_k = trace(C * H * PP(:,:,t) * H' * C' + R) / m;
    Nt =  Xp(:,t)' * H' * V * H * Xp(:,t) + eta_k;

    C = C + ((Y(:,t) - C * H * Xp(:,t)) * Xp(:,t)' * H' * V) / (Nt);
    V = V - (V * H * (Xp(:,t) * Xp(:,t)') * H' * V) / (Nt);

end

end
