% Workspace cleaning
clear
close all
clc

load("data/afterFeaturesSelection.mat");

% Euclidean distance between pairs of observations
D = pdist(InputStdECG');
fprintf('Min distance: %f\n', min(D));
fprintf('Max distance: %f\n', max(D));

% Parameters initialization
P = InputStdECG;
T = TargetStdECG;
goal = 0.0;
spread = 0.6;
MN = 300; % Maximum number of neurons

% Radial basis network creation
net = newrb(P, T, goal, spread, MN);
y = net(InputStdECG);

% Plot creation
figure, plotregression(TargetStdECG, y)