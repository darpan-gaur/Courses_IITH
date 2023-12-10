rawdat = dlmread('err.dat');

nx = rawdat(:,1); ny = rawdat(:,2); err = rawdat(:,3);

figure, clf
loglog(nx, err, 'k-o', nx, nx.^(-2)*(err(1)/nx(1)^(-2)*1.2), 'b--', 'LineWidth', 2)
xlabel('nx'), ylabel('L2 Err'), set(gca,'FontSize',14)
xlim([nx(1)/2 nx(end)*1.1])
screen2jpeg('err.png')
