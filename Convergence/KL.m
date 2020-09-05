%% PSMF Convergence Experiment - Compute KL divergence
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%
function K = KL(m0,m1,S0,S1)

d = length(m1);

K = 0.5 * (trace(S1\S0) * (m1-m0)' / S1 * (m1-m0)  - d + log(det(S1)) - log(det(S0)));

end
