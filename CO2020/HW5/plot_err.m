rawdat = dlmread('err.dat');

nx = rawdat(:,1); err = rawdat(:,2); err2 = rawdat(:,3);
% nx = rawdat(:,1); err = rawdat(:,2); 

figure, clf
loglog(nx, err, 'k-+', nx, err2, 'r--o', 'LineWidth', 2)
% loglog(nx, err, 'k-o', 'LineWidth', 2)
% show line with slope -4
% loglog(nx, err, 'k-o', nx, nx.^(-4)*(err(1)/nx(1)^(-4)*1.2), 'b--', 'LineWidth', 2)
xlabel('nx'), ylabel('L2 Err'), set(gca,'FontSize',14)
xlim([nx(1)/2 nx(end)*1.1])
% ylim([err(end)/2 err(1)*1.1])
screen2jpeg('err.png')
