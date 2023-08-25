% Workspace cleaning
clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Network parameters
hiddenSizes = 60;

% 'trainlm', 'trainbr', 'trainbfg', 'trainrp', 'trainscg', 'traincgb',
% 'traincgf', 'traincgp', 'trainoss', 'traingdx', 'traingd'	
trainFcn = 'trainbfg';

% Network creation
net = fitnet(hiddenSizes, trainFcn);
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.2;
net.performFcn = 'mse'; 

[net,tr] = train(net, InputMeanECG, TargetMeanECG);

y = net(InputMeanECG);

% Plot creation
figure, plotregression(TargetMeanECG, y)
