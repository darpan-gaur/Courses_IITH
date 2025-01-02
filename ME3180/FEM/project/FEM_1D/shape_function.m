function [N,dNdxi] = shape_function(nen,xi)
%
%        Compute N and dN/dxi at the current integration point
%
           N = zeros(1,nen);
           dNdxi = zeros(1,nen);
           if (nen == 3) 
             N(1) = -0.5*xi*(1.-xi);
             N(2) = 0.5*xi*(1.+xi);
             N(3) = (1.-xi^2);
             dNdxi(1) = -0.5+xi;
             dNdxi(2) = 0.5+xi;
             dNdxi(3) = -2.*xi;
           elseif (nen == 2)
             N(1) = 0.5*(1.-xi);
             N(2) = 0.5*(1.+xi);
             dNdxi(1) = -0.5;
             dNdxi(2) = 0.5;
           end
end