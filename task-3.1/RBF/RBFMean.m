% Workspace cleaning
clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Euclidean distance between pairs of observations
D = pdist(InputMeanECG');
fprintf('Min distance: %f\n', min(D));
fprintf('Max distance: %f\n', max(D));

% Parameters initialization
P = InputMeanECG;
T = TargetMeanECG;
goal = 0.0;
spread = 0.3;
MN = 300; % Maximum number of neurons

% Radial basis network creation
net = newrb(P, T, goal, spread, MN);
y = net(InputMeanECG);

% Plot creation
figure, plotregression(TargetMeanECG, y)

