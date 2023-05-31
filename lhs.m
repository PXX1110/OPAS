% 拉丁超立方采样的试验程序
% 每个维度都服从一个参数不同正态分布，每个维度都是独立的。
clear; clc; close all;

% Mu = [0 0];
% Sigma = [1 1];
% SIGMA must be a square matrix with size equal to the number of columns in MU, or a row vector with length equal to the number of columns in MU.
% 虽然系统这么提示，但是CovarianceMatrix直接用p*1维向量不行
% CovarianceMatrix = Sigma;
% CovarianceMatrix = diag(CovarianceMatrix);		% 转换为对角矩阵
N = 100;                            			% 样本点数目
X1 = rand(1,100);
X2 = rand(1,100);
% X = lhsnorm(Mu, CovarianceMatrix, N);			% 调用函数
z = rand(100,100);
figure
% plot3(X(:,1),X(:,2),z,'*')
plot3(X1,X2,z,'*')
grid on