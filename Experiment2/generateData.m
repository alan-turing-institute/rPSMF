%% PSMF Changepoint Detection Experiment - Generate data
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%
function [Y] = generateData(d,L,k,k2)

m = 4;  % mass [kg]
b = 0.5;  % damping constant [kg/m/s]

m2 = 2;
b2 = 0.5;

% Define the state-space matrices

% The dynamic system is defined by these matrices:
%
% (d/dt) x = A x + B u
%        y = C x + D u
%
% where
%
% x: internal state
% u: input to system
% y: output from system

A1 = [0 1; (-k/m) (-b/m)];   % state --> state derivative
A2 = [0 1; (-k2/m2) (-b2/m2)];   % state --> state derivative
A = [A1, zeros(size(A1)); zeros(size(A1)), A2];
B1 = [0 ; (1/m)];            % input --> state derivative
B2 = [0; (1/m2)];
B = [B1;B2];


x = [1 ; 0; 0 ; -1];                % initial state

t = 0;                      % initial time [s]
u = 1;                      % initial input

% Simulation parameters

N = L;                    % number of time steps
dt = 0.1;                   % time step [seconds]

% allocate space for the results

xs = nan(4, N);             % time series of true state
ts = nan(1, N);             % time

% Do the simulation

for ii=1:length(xs)
    % record the state
    xs(:, ii) = x;
    ts(ii) = t;
    
    % advance the state
    %   x     = expm(A*dt)*(x + B*u);
    x = (eye(4) + dt*A) * (x + B * u);
    t = t + dt;
    
end

% Plot the results

% figure(1);
% subplot(2,1,1);
% plot(ts, xs, '.-');
% ylabel('position [m]');
% xlabel('time [s]');
% grid on
% 
% subplot(2,1,2);
% plot(ts, xs(2,:), '.-');
% ylabel('velocity [m/s]');
% xlabel('time [s]');
% grid on

C = randn(d,4);

var = 1;

Y = C * xs + sqrt(var) * randn(size(C*xs));
% Y = C * xs + trnd(1.5,size(C*xs));

% figure(2),
% for i = 1:m
%     subplot(5,4,i),plot(Y(i,:));
% end

end
