function [w,xi] = gauss_quadrature(ngpoints)
%
%    Integration points and weights for 2 point integration
%
       if (ngpoints == 2)
         w = [1,1];
         xi = [-0.5773502692,0.5773502692];
       elseif (ngpoints == 1)
         w = [2.,0.];
         xi = [0.,0.];
       end
end