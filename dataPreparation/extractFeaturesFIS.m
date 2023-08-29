% Workspace cleaning
clear
close all
clc

load('data/afterFeaturesSelection.mat')

% Feature selection options
options = statset('Display', 'iter','UseParallel',true);

% Running features selection to determine top 3 features for Fuzzy System
selectedFeatures = selectFeatures(InputActivity, TargetActivityClassesVec, options, 'Activity', 3);

% Save results
InputActivity = InputActivity(selectedFeatures, :);
save('data/inputFuzzySystem', InputActivity, TargetActivityClassesVec);

% Run feature selection and save the results
function [selectedFeatures] = selectFeatures(Features, Target, options, fileName, nFeatures)
    filePath = (strcat('results/dataPreparation/sf', fileName));
    diary(strcat(filePath, 'Log.txt'));
    selectedFeatures = sequentialfs(@selectionCriterionActivity, Features', Target', ...
        'options', options, 'nfeatures', nFeatures);
    diary('off');
end

