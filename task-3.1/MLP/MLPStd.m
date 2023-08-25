% Workspace cleaning
clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Network parameters
hiddenSizes = 65;
trainFcn = 'trainbr';

% Network creation
net = fitnet(hiddenSizes, trainFcn);
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.2;
net.performFcn = 'mse'; 

[net,tr] = train(net, InputStdECG, TargetStdECG);

y = net(InputStdECG);

% Plot creation
figure, plotregression(TargetStdECG, y)
