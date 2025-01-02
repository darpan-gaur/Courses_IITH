function [K,F] = compute_stiffness_force(nnodes,nelem,nen,coords,connect,props,bodyforce,traction)
%
%
%     obtain gauss points and weights
       ngpoints = nen-1;
       [w,xi] = gauss_quadrature(ngpoints);
%%
%     Assemble the global stiffness and force vector
%
       K = zeros(nnodes,nnodes);
       F = zeros(nnodes,1);
%%
       for lmn = 1 : nelem 
%%
%       Extract the coords of each node on the current element
%
         lmncoords = zeros(nen);
         for a = 1 : nen 
           lmncoords(a) = coords(connect(a,lmn));
         end
%%
%      For the current element, loop over integration points and assemble element stiffness
%
         kel = zeros(nen,nen);
         fel = zeros(nen,1);
%%
         for II = 1 : ngpoints
            [N,dNdxi] = shape_function(nen,xi(II));
%% 
%        Compute dx/dxi, J and dN/dx
%
           dxdxi = 0.;
           for a = 1 : nen 
             dxdxi = dxdxi + dNdxi(a)*lmncoords(a);
           end
           J = abs(dxdxi);
           dNdx = zeros(1,nen);
           for a = 1 : nen
             dNdx(a) = dNdxi(a)/dxdxi;
           end
%%  
%         Add contribution to element stiffness and force vector from current integration pt
%
          for a = 1 : nen
             fel(a) = fel(a) + w(II)*bodyforce*J*N(a);
             for b = 1 : nen 
               kel(a,b) = kel(a,b) + props(1)*props(2)*w(II)*J*dNdx(a)*dNdx(b);
             end
          end
% 
         end
%%
%       Add the stiffness and residual from the current element into global matrices
%
         for a = 1 : nen
           rw = connect(a,lmn);
           F(rw) = F(rw) + fel(a); 
           for b = 1 : nen 
             cl = connect(b,lmn);
             K(rw,cl) = K(rw,cl) + kel(a,b);
           end
         end
       end          
%     Add the extra forcing term from the traction at x=L
%
       F(traction(:,1)) = F(traction(:,2)) + traction(:,3);
end