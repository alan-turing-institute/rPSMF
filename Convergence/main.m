%% PSMF Convergence Experiment - Main script
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%

clc;
clear;
close all;

load data.mat

rng(5),

% Standard Kalman

X0k = 5 * randn(r,1);
P0k = 5 * eye(r);
Ck = Ctrue;

t = 1;
Ppk = P0k + Q;
Sk = eye(d) / (R + Ck * Ppk * Ck');
Xk(:,t) = X0k + Ppk * Ck' * Sk * (Y(:,t) - Ck * X0k);
Pk(:,:,t) = Ppk - Ppk * Ck' * Sk * Ck * Ppk;

for t = 2:n
    
    Ppk = Pk(:,:,t-1) + Q;
    
    Sk = eye(d) / (R + Ck * Ppk * Ck');
    Xk(:,t) = Xk(:,t-1) + Ppk * Ck' * Sk * (Y(:,t) - Ck * Xk(:,t-1));
    Pk(:,:,t) = Ppk - Ppk * Ck' * Sk * Ck * Ppk;
end

% psmf filter
%
X0ps = 5 * randn(r,1);
P0ps = 5 * P0k;
C = 10 * abs(randn(d,r));
% C = CrootInit * CrootInit';
v = 1;
V = v * eye(r);
R = 1 * R;
Q = 1 * Q;

Cerr = zeros(1,n);

for iter = 1:10000
    Cerrold = Cerr;
    t = 1;
    if iter > 1
        X0ps = Xps(:,1);
        P0ps = Pps(:,1);
        V = v * eye(r);
    end
    
    Ppps = P0ps + Q;
    Xpps = X0ps;
    
    eta_k = trace(R + C * Ppps * C')/d;
    Rb = R + eye(d) .* (Xpps' * V * Xpps);
    Skps = eye(d) / (Rb + C * Ppps * C');
    
    Xps(:,t) = Xpps + Ppps * C' * Skps * (Y(:,t) - C * Xpps);
    Pps(:,:,t) = Ppps - Ppps * C' * Skps * C * Ppps;
    
    C = C + ((Y(:,t) - C * Xpps) * Xpps' * V)/(Xpps' * V * Xpps + eta_k);
    V = V - (V * (Xpps * Xpps') * V)/(Xpps' * V * Xpps + eta_k);
    
    Cerr(t) = norm(C - Ctrue);
    
    for t = 2:n
        
        Ppps = Pps(:,:,t-1) + Q;
        Xpps = Xps(:,t-1);
        
        %         V = V + eye(r) * 1e-6;
        
        eta_k = trace(R + C * Ppps * C')/d;
        Rb = R + eye(d) .* (Xpps' * V * Xpps);
        Skps = eye(d) / (Rb + C * Ppps * C');
        
        Xps(:,t) = Xpps + Ppps * C' * Skps * (Y(:,t) - C * Xpps);
        Pps(:,:,t) = Ppps - Ppps * C' * Skps * C * Ppps;
        
        C = C + ((Y(:,t) - C * Xpps) * Xpps' * V)/(Xpps' * V * Xpps + eta_k);
        V = V - (V * (Xpps * Xpps') * V)/(Xpps' * V * Xpps + eta_k);
        
        Cerr(t) = norm(C - Ctrue);
        
        W(t) = Wasserstein2(Xps(:,t),Xk(:,t),Pps(:,:,t),Pk(:,:,t));
        
        if mod(t,2500) == 0 && iter == 1
            figure(1),
            clf,
            
            subplot(221),loglog(W(1:t)),
            subplot(222),loglog(Cerr(1:t));
            subplot(2,2,3); plot(X(:,1:t)','b');hold on; plot(Xk(:,1:t)','k'); hold on; plot(Xps(:,1:t)','r');
            %             subplot(2,2,4),plot(ErM);
            
            drawnow,
        end
        
    end
    
    ErM(iter) = norm(Y - C * Xps);
    avW(iter) = mean(W);
    CerrI(iter) = norm(C - Ctrue);
    if mod(iter,100) == 0
        
        figure(1),
        clf,
        subplot(221),loglog(avW,'LineWidth',2,'Color','black'),
        xlabel({'Iterations','(a)'});ylabel('Averaged Wasserstein distance','FontSize',12);
        subplot(222),loglog(CerrI,'LineWidth',2,'Color','black');
        xlabel({'Iterations','(b)'});ylabel('$\|C_k - C^\star\|$','Interpreter','latex','FontSize',14);
        subplot(2,2,[3,4]);
        plot(Xk','k'); hold on; plot(Xps','r');
        legend('Optimal filter mean estimate','Approximate filter mean estimate','FontSize',14);
        xlabel({'Time Index','(c)'},'FontSize',14);
        drawnow,
        
        %         display(norm(Cerrold-Cerr))
    end
    
end

save expdata.mat

%%

figure(1),
clf,
subplot(131),loglog(avW,'LineWidth',2,'Color','black'),
xlabel({'Iterations','(a)'},'FontSize',14);
ylabel('Averaged Wasserstein distance','FontSize',11);
subplot(132),loglog(CerrI,'LineWidth',2,'Color','black');
xlabel({'Iterations','(b)'},'FontSize',14);
ylabel('$\|C_k - C^\star\|$','Interpreter','latex','FontSize',14);
subplot(133);
plot(Xk','k'); hold on; plot(Xps','r');
legend('Optimal filter','Approximate filter (PSMF)','FontSize',14);
xlabel({'Time Index','(c)'},'FontSize',14);
drawnow,

print(figure(1),'-depsc','Fig1');
