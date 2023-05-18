function [x1_save,error_laborde1] = acc_gaussian_laborde_binary_classification(f,x,X,Y,c,s_k,iter,x_star)
x1=x ;
z=x;
t_k_sum = 0;
L=1/c^2;
sigma=1;
[n,d] = size(X);
for i=1:iter
    %Laborde
    t_k_sum = t_k_sum+s_k(i);

    error_laborde1(1,i) = f(x1,X,Y)-f(x_star,X,Y);
    noise = randn(d,1);
    grad = sum(-1/n.*(Y.*X)'.*(exp(-Y.*(X*x1)))'./(1+exp(-Y.*(X*x1)))',2);

    y = (2*s_k(i)/t_k_sum)*z + (1-2*s_k(i)/t_k_sum)*x1;
    x1 = y - s_k(i)/sqrt(L)*(grad+sigma^(1/2)*noise);
    z = z- s_k(i)/(2)* t_k_sum*(grad+sigma^(1/2)*noise);
    x1_save(:,i) = x1;
end
end