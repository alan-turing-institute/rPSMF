%% PSMF Convergence Experiment - Compute Wasserstein-2 distance
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%
function K = Wasserstein2(m1,m2,S1,S2)

S1sq = sqrtm(S1);

K2 = norm(m1-m2)^2 + trace(S1 + S2 - 2 * sqrtm(S1sq * S2 * S1sq));
K = sqrt(K2);

end
