function [K,F] = applyBC(K,F,nnodes,fixnodes)
%      Modify FEM equations to enforce displacement boundary condition
%      To do this we simply replace the equation for the first node with u=0
%
       for a = 1 : nnodes
         K(fixnodes(:,2),a) = 0.;
       end
       K(fixnodes(:,2),fixnodes(:,2)) = 1.;
       F(fixnodes(:,2)) = 0.;
end