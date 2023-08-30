% Load data
clear; clc; close all;
load('data/inputFuzzySystem.mat');

t = vec2ind(TargetActivityClassesVec);

% Divide input in classes
run = InputActivity(:, t == 1);
sit = InputActivity(:, t == 2);
walk = InputActivity(:, t == 3);

nFeatures = size(InputActivity, 1);
nBins = 15;

% Plot feature distribution to define membership_function
figure;
for i = 1:nFeatures
    subplot(2, 2, i);
    plotHistogram(InputActivity(i, :), "All Activities", i, nBins);
end
sgtitle("Feature Distribution for All Activities");

% Plot histograms to define rules
figure;
for i = 1:nFeatures
    subplot(nFeatures, 3, 3 * (i - 1) + 1);
    plotHistogram(run(i, :), "RUN", i, nBins);
    
    subplot(nFeatures, 3, 3 * (i - 1) + 2);
    plotHistogram(sit(i, :), "SIT", i, nBins);
    
    subplot(nFeatures, 3, 3 * (i - 1) + 3);
    plotHistogram(walk(i, :), "WALK", i, nBins);
end
sgtitle("Feature Distribution by Activity");

% Adjust figure layout
%figureHandle = gcf;
%set(findall(figureHandle, 'Type', 'Axes'), 'YScale', 'log');

% Define function to plot histogram
function plotHistogram(data, classLabel, featureIndex, nBins)
    histogram(data, nBins, 'BinWidth', 0.05, 'BinLimits', [0, 1]);
    title("Histogram of feature " + num2str(featureIndex) + ": " + classLabel);
    xticks(linspace(0, 1, 11));
end
