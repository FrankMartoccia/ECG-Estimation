% Workspace cleaning
clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Network parameters
hiddenSizes = 60;
trainFcn = 'trainbr';

% Network creation
net = fitnet(hiddenSizes, trainFcn);
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.0; % trainbr doesn't need a validation set
net.divideParam.testRatio = 0.2;
% performFcn = 'mse' (by default)

[net,tr] = train(net, InputStdECG, TargetStdECG);

y = net(InputStdECG);

% Plot creation
figure, plotregression(TargetStdECG, y)
