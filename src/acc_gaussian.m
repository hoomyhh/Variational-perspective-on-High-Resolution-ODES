function [x_save,error_me_Gaus_noise,upperbound,upperbound_laborde] = acc_gaussian(s_k,f,gradf,x,iter,s,c,sigma,x_star)

t_k_sum = 0;
n = length(x);
L = 1/(c^2);
v=x;
x0=x;
a = (-2*c^2+3)/(4*c^2);
for k=1:iter
    %error_me(k) = norm(gradf(x(:,k)))^2;
    error_me_Gaus_noise(1,k) = f(x)-f(x_star);
        t_k_sum = t_k_sum+s_k(k);
        noise1 = randn(n,1);
        
        x = (1+2*s_k(k)/t_k_sum)^(-1)*(x + 2*s_k(k)/t_k_sum *v)-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)*a/sqrt(L)*(gradf(x)+sigma^(1/2)*noise1);
        x_save(:,k)=x;
        noise1 = randn(n,1);
        v = v - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*a/sqrt(L))*(gradf(x)+sigma^(1/2)*noise1);

       
end
k=1;
 upperbound(k) = (1/2*norm(x0-x_star)^2+c^4*sigma^2/8*(16*(1+log(k))+38))/(2*c^2*(2*((k+1)^(1/4)-1)^2+(k+1)^(-3/4)*((k+1)^(1/4)-1)));

        upperbound_laborde(k) = (1/(32*c^2)*norm(x0-x_star)^2+c^2*sigma^2*(1+log(k)))/(((k+1)^(1/4)-1)^2);

end