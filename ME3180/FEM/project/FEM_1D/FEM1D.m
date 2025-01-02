function FEM1D
clc
clear
%%
%       Simple FEA program using 1D elements
%%
% ================= Read data from the input file ==================
%
% Change the name of the file below to point to your input file
fname = 'FEM_1D_input';
infile=fopen([fname,'.txt'],'r');
[nnode,nelem,nen,coord,connect,props,bodyforce,load,fixnodes] = readInputData(infile);
%%
%
%   Plot Original Mesh
%
  us = zeros(1,nnode);
  plot(coord,us,'b - s','linewidth',2)
  xlabel('Distance x')
  title('Discretization of 1D bar','FontSize',12)
%%
% Stiffness Matrix and Force Vector Calculation
  [K,F] = compute_stiffness_force(nnode,nelem,nen,coord,connect,props,bodyforce,load);
% Apply Boundary Conditions
  [K,F] = applyBC(K,F,nnode,fixnodes);
  %     
  %    Solve the equations
  %
  u = K\F;
  %
%%
%   Plot Deformed Mesh
%
  hold on
  plot(coord+u,us,'r s','linewidth',2)
  legend('Original bar','Deformed Bar')
  axis([0 6 -.5 0.5])
%%
%   Plot Displacement
%
  figure 
  plot(coord,u,'r - s')
  xlabel('Distance x')
  ylabel('Displacement u')
  title('Displacement of 1D bar','FontSize',12)
end