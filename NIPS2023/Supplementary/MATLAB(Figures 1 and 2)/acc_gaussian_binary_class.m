function [x_save,error_me_Gaus_noise,x_0,v_0,t_k_0] = acc_gaussian_binary_class(k_0,s_k,nepochs,x,iter,c,sigma,x_star,X,Y,f)

t_k_sum = 0;
[n,d] = size(X);
L = 1/(c^2);
v=x;
grad = sum(-1/n.*(Y.*X)'.*(exp(-Y.*(X*x)))'./(1+exp(-Y.*(X*x)))',2);

x0 = x;
error_me_Gaus_noise = zeros(1,iter);
%a = (-2*c^2+3+sqrt((2*c^2-1)^2+8))/(4*c^2);
a = L;

for k=1:iter
    %error_me(k) = norm(gradf(x(:,k)))^2;



    error_me_Gaus_noise(1,k) = f(x,X,Y)-f(x_star,X,Y);
        t_k_sum = t_k_sum+s_k(k);
        noise1 = randn(d,1);
        
        
        x = (1+2*s_k(k)/t_k_sum)^(-1)*(x + 2*s_k(k)/t_k_sum *v)-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)*a/sqrt(L)*(grad+sigma^(1/2)*noise1);
        x_save(:,k) = x;
        grad = sum(-1/n.*(Y.*X)'.*(exp(-Y.*(X*x)))'./(1+exp(-Y.*(X*x)))',2);

        noise1 = randn(d,1);
        v = v - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*a/sqrt(L))*(grad+sigma^(1/2)*noise1);
        
        if k==k_0
            x_0 = x;
            v_0 = v;
            t_k_0 = t_k_sum;
            
        end
end
    



end