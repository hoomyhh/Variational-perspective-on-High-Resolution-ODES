function [x1_save,error_srvg] = svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star,f)
L = 1/(c^2);
x1=x0 ;
v1=x1;
eta = 0.2;
beta = 1/L;
[n,d] = size(X);
for k=1:nepochs
    xhat = x1;
    ghat = zeros(d,1);
    for p =1:n
        xi = X(p,:);
        yi = Y(p);
        ghat = ghat + gradf_i(xhat,xi,yi);
    end
    ghat = ghat/n;
for i=1:iter
    x1_save(:,(k-1)*iter+i) = x1;
    j=randi(n);  
    xi = X(j,:);
    yi = Y(j);
    %x1 = x1 - beta*gradf_i(x1,j)-eta*(v1-x1);
    x1 = x1 - beta*(gradf_i(x1,xi,yi)-gradf_i(xhat,xi,yi)+ghat);
    v1=x1_save;
    error_srvg(1,(k-1)*iter+i)=f(x1,X,Y)-f(x_star,X,Y);
end
end
end