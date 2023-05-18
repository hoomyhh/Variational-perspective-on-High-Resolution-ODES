function [x1_save,x11_save,error_sgd,std_rate,error_sgd2] = sgd_binary_classification(X,Y,s_k,gradf_i,x0,iter,c,x_star,f)
L = 1/(c^2);
x1=x0 ;
v1=x1;
x11=x1;
beta=1/L;
[n,d] = size(X);

for i=1:iter
    j=randi(n);  
    xi = X(j,:);
    yi = Y(j);
    %x1 = x1 - beta*gradf_i(x1,j)-eta*(v1-x1);
    x11 = x11 - beta*(gradf_i(x11,xi,yi));
    grad = sum(-1/n.*(Y.*X)'.*(exp(-Y.*(X*x1)))'./(1+exp(-Y.*(X*x1)))',2);
    noise1 = randn(d,1);
    x1 = x1 - s_k(i)*(grad+noise1);
    x1_save(:,i) = x1;
    x11_save(:,i) = x11;
    error_sgd(1,i)=f(x1,X,Y)-f(x_star,X,Y);
    std_rate(1,i) = norm(x0-x_star)^2/sqrt(i);
    error_sgd2(1,i)=f(x11,X,Y)-f(x_star,X,Y);

end
end