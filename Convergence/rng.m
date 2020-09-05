%% PSMF Convergence Experiment - Set random seed
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%
function rng(x)
  randn('seed', x)
  rand('seed', x)
end
