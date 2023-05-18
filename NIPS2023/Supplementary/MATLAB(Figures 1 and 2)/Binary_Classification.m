
%{
Source code for the paper 
"A Variational Perspective on High-Resolution ODEs" 

Submitted to NIPS-2023
%}




clc
clear

%For almost exact result as the paper uncomment the following line.
%rng(20)

%% Initializations
d=10;%dimension
n=1000;%number of samples
alpha=3/4;%decreasing rate for our method
Alpha=2/3;%decreasing rate for perturbed GD (according to theory)
nepochs = 1;%number of epochs
iter = 2e3;%number of iterations in each epoch
mont_iter = 10;%number of Monte-Carlo Simulations
k_0=1;%tuning parameter for bounds calculation
%% Function
x_star = randn(d,1);
X = randn(n,d)+1;

Y = (1./(1+exp(-X*x_star))>= rand(n,1))*2-1;
f = @(x,X,Y) 1/n*sum(log(1+exp(-(Y.*X)*x)));
gradf_i = @(x,xi,yi) -xi'*yi*exp(-yi*xi*x)/((1+exp(-yi*xi*x)));

%% Main implementation

% First I find the optimal values
x0 = randn(d,1);
L = max(Y.^2.*diag(X*X'));
c=1/sqrt(L);
c1 = 1/L;
s=1/L;
[x_opt,error_srvg] = svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star,f);



s_k =@(k) c/(k^alpha);
s_k1 =@(k) c1/(k^Alpha);


for mont=1:mont_iter

% Gaussian Noise 
 x0=randn(d,1);
 sigma=1;
 [x_me(:,:,mont),error_me_Gaus(mont,:),x_0,v_0,t_k_0] = acc_gaussian_binary_class(k_0,s_k,nepochs,x0,iter*nepochs,c,sigma,x_star,X,Y,f);

% Laborde
 [x_laborde(:,:,mont),error_laborde1(mont,:)] = acc_gaussian_laborde_binary_classification(f,x0,X,Y,c,s_k,iter*nepochs,x_star);

 % SGD
 [x_sgd_noise(:,:,mont),x_sgd(:,:,mont),error_gd_perturbed(mont,:),std_rate,error_sgd(mont,:)] = sgd_binary_classification(X,Y,s_k1,gradf_i,x0,iter*nepochs,c,x_star,f);

% SVRG
 [x_svrg(:,:,mont),error_svrg(mont,:)] = svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star,f);

 % SVRG + NNAG 
 [x_me_svrg(:,:,mont),error_me_svrg(mont,:)] = acc_me_svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star,f,d);

% SGD + NNAG
 [x_me_sgd(:,:,mont),error_me_sgd(mont,:),std_rate] = acc_me_sgd_binary_classification(X,Y,s_k,f,gradf_i,x0,iter*nepochs,c,n,x_star,L);

 % Upper bounds
 L1 = 100;
 L2 = 1000;
 [upperbound_L1,upperbound2_L1,upperbound_Laborde_L1] = upperbounds(t_k_0,k_0,sigma,L1,x0,iter*nepochs,X,Y,x_star,x_0,v_0,f);
 [upperbound_L2,upperbound2_L2,upperbound_Laborde_L2] = upperbounds(t_k_0,k_0,sigma,L2,x0,iter*nepochs,X,Y,x_star,x_0,v_0,f);
mont
end



%% Visualization

Ymatrix1 = [1/mont_iter*sum(error_me_Gaus,1)',1/mont_iter*sum(error_laborde1,1)',1/mont_iter*sum(error_gd_perturbed,1)'];
Ymatrix2 = [1/mont_iter*sum(error_sgd,1)',1/mont_iter*sum(error_svrg,1)',1/mont_iter*sum(error_me_sgd,1)',1/mont_iter*sum(error_me_svrg,1)']; 

Fig2(Ymatrix1,Ymatrix2)

X1 = (1:iter*nepochs);
Ymatrix1 = [upperbound2_L1',upperbound_L1',upperbound_Laborde_L1'];
Ymatrix2 = [upperbound2_L2',upperbound_L2',upperbound_Laborde_L2'];

Fig1(X1,Ymatrix1,Ymatrix2)
