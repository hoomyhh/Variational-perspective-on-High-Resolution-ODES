function [x_save,error_laborde1] = acc_laborde_gaussian_noise(s_k,f,gradf,x0,iter,sigma,c,x_star)

x = x0;
z = x0;
t_k_sum = 0;
n = length(x);
L = 1/(c^2);
for i=1:iter
    %La-borde
    t_k_sum = t_k_sum+s_k(i);
    error_laborde1(1,i) = f(x)-f(x_star);
    noise = randn(n,1);
    x_save(:,i) = x;
%     sigma=0;
%         gard_c = gradf(x);
%         for j = 1:d
%             sigma = sigma +(gard_c-gradf_i(x,j))* (gard_c-gradf_i(x,j))';
%         end
%         sigma = sigma./d;
    y = (2*s_k(i)/t_k_sum)*z + (1-2*s_k(i)/t_k_sum)*x;
    x = y - s_k(i)/sqrt(L)*(gradf(y)+sigma^(1/2)*noise);
    z = z- s_k(i)/(2)* t_k_sum*(gradf(y)+sigma^(1/2)*noise);
end
end