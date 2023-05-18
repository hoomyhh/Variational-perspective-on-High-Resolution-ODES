function [upperbound,upperbound2,upperbound_laborde] = upperbounds(t_k_0,k_0,sigma,L,x0,iter,X,Y,x_star,x_0,v_0,f)
c=1/sqrt(L);
a=L;

for k = 1:iter
        upperbound(k) = (1/2*norm(x0-x_star)^2+c^4*sigma^2/8*(16*(1+log(k))+38))/(2*c^2*(2*((k+1)^(1/4)-1)^2+(k+1)^(-3/4)*((k+1)^(1/4)-1)));

        upperbound_laborde(k) = (1/(32*c^2)*norm(x0-x_star)^2+c^2*sigma^2*(1+log(k)))/(((k+1)^(1/4)-1)^2);

        if k>=k_0
            upperbound2(k) = ((t_k_0^2/4+t_k_0*a/(2*sqrt(L)))*(f(x_0,X,Y)-f(x_star,X,Y))+1/2*norm(v_0-x_star)^2+c^4*sigma^2*2*(log(k/k_0))+8*sigma^2*c^3*a/sqrt(L)*...
                (k_0^(-1/4)-k^(-1/4))+a^2*c^2*sigma^2/(L)*(k_0^(-1/2)-k^(-1/2)))/(c^2*4*(((k+1)^(1/4)-1)^2)+2*c*a/sqrt(L)*((k+1)^(1/4)-1));  
        end
end

end