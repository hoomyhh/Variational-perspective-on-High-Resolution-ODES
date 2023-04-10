clc
clear

rng(20)
d=10;
n=1000;
alpha=3/4;
Alpha=2/3;
m=0;
nepochs = 100;
iter = 1e3;
mont_iter = 100;
%% Vector Norm Squared function
x_star = randn(d,1);
X = randn(n,d)+1;

Y = (1./(1+exp(-X*x_star))>= rand(n,1))*2-1;
f = @(x,X,Y) 1/n*sum(log(1+exp(-(Y.*X)*x)));
gradf_i = @(x,xi,yi) -xi'*yi*exp(-yi*xi*x)/(1+exp(-yi*xi*x));

%% Stochastic version of HR Nag
% First I find the optimal values
x0 = randn(d,1);
L = max(Y.^2.*diag(X*X'));
c=1/sqrt(L);
c1 = 1/L;

%% Main implementation
s_k =@(k) c/(k^alpha);
s_k1 =@(k) c1/(k^Alpha);


for mont=1:mont_iter
%% Gaussian Noise
 x0=randn(d,1);
 sigma=1;
[x_me(:,:,mont),error_me_Gaus(mont,:),upperbound,upperbound2,upperbound_laborde] = acc_gaussian_binary_class(s_k,nepochs,x0,iter*nepochs,c,sigma,x_star,X,Y,f);


%% Laborde
[x_laborde(:,:,mont),error_laborde1(mont,:)] = acc_gaussian_laborde(f,x0,X,Y,c,s_k,iter*nepochs,x_star);
%% SGD
 [x_sgd_noise(:,:,mont),x_sgd(:,:,mont),error_sgd(mont,:),std_rate,error_sgd2(mont,:)] = sgd_binary_classification(X,Y,s_k1,gradf_i,x0,iter*nepochs,c,x_star,f);

%% SVRG
[x_svrg(:,:,mont),error_svrg(mont,:)] = svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star,f);

mont
end

%% GD
 x1=x0 ;
 v1=x1;
for i=1:iter*nepochs
error_GD(i) = f(x1,X,Y)-f(x_star,X,Y);
x1 = x1 - 1/L*sum(-1/n.*(Y.*X)'.*(exp(-Y.*(X*x1)))'./(1+exp(-Y.*(X*x1)))',2);
end
loglog(1/mont_iter*sum(error_me_Gaus,1))
hold on
loglog(1/mont_iter*sum(error_laborde1,1))
loglog(1/mont_iter*sum(error_sgd,1))
loglog(1/mont_iter*sum(error_sgd2,1))
loglog(1/mont_iter*sum(error_svrg,1))
loglog(upperbound)
loglog(upperbound2)
loglog(upperbound_laborde)
loglog(error_GD)
