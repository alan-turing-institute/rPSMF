%% PSMF Changepoint Detection Experiment - Main script
%
% This file is part of the PSMF codebase.
% See the LICENSE file for copyright and licensing information.
%

clc;
clear;
close all;

dofInd = 1;
Results = zeros(5,3);

contDeg = 0.05; % Contamination degree

for dof = 1.5:0.1:1.9

    Iter = 1000;

    bocpdS = 0;

    for z = 1:Iter

        L = 1200;
        m = 20;

        k = 5;  % spring constant [m/s]
        k2 = 5;

        Y = generateData(m,L,k,k2);

        k = 1;  % spring constant [m/s]
        k2 = 1;
        m2 = 3;
        L2 = 600;
        Y2 = generateData(m2,L2,k,k2);

        rows1 = datasample(1:m,m2,'Replace',false);

        for i = 1:m2
            Y(rows1(i),L-L2+1:L) = Y2(i,:);
        end

        for i = 1:m
            for j = 1:L
                if rand < contDeg
                    Y(i,j) = Y(i,j) + trnd(dof);
                end
            end
        end

        SP = L-L2+1;

        r = 10;
        n = length(Y);
        lam = 0.001;

        % specify the state space model
        [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_matern32_to_ss(0.1,0.1);

        dt = 0.001;

        A = expm(F*dt);

        Q = Pinf - A * Pinf * A';

        A = kron(eye(r),A);
        Q = kron(eye(r),Q);
        H = kron(eye(r),H);

        V = 1 * eye(r);
        P0 = 1 * eye(2 * r);

        C = randn(m,r);
        X = randn(2 * r,n);

        Cinit = C;
        Xinit = X;

        R = lam * eye(m);
        nu = 1;
        [X] = PSMF(r,Y,Q,A,R,H,V,P0,C,X,m,n);

        pls = 1:2:(2*r - 1);

        st = 400;
        inds = findchangepts(X(pls,st:n),'Statistic','rms');
        indsRaw = findchangepts(Y(:,st:n),'Statistic','rms');


        PGP(z) = abs(inds + (st - 1) - SP);
        Pstd(z) = abs(indsRaw + (st - 1) - SP);

        Sb = MVBOCPD(Y);
        bocpdS = bocpdS + Sb;


        BOCPDacc = bocpdS / z;
        GPacc = length(find(PGP < 30))/length(PGP);
        StdAcc = length(find(Pstd < 30))/length(Pstd);
        if mod(z,1) == 0
            display(['Iter: ' num2str(z) ' Degrees of freedom of 5% contamination: ' num2str(dof)]);
            display(['PSMF-PELT accuracy: ' num2str((GPacc))]);
            display(['PELT accuracy: ' num2str((StdAcc))]);
            display(['BOCPD accuracy: ' num2str((BOCPDacc))]);
        end
    end

    % hold off;
    % if length(SPn) == 2
    %     SPn - [SP,SP2]
    % end

    % print(figure(3),'-depsc','Chpts')

    % figure,histogram(Pstd,100); hold on; histogram(PGP,100,'EdgeColor','red')

    Results(dofInd,:) = [GPacc,StdAcc,BOCPDacc];
    dofInd = dofInd + 1;

end

% save('res.mat','Results');
