% Workspace cleaning
clear
close all
clc

load('data/afterFeaturesSelection.mat')

% Feature selection options
options = statset('Display', 'iter','UseParallel',true);

% Running features selection to determine top 3 features for Fuzzy System
selectFeatures(InputActivity, TargetActivityClassesVec, options, 'Activity', 3)

% Run feature selection and save the results
function selectFeatures(Features, Target, options, fileName, nFeatures)
    filePath = (strcat('results/dataPreparation/sf', fileName));
    diary(strcat(filePath, 'Log.txt'));
    selectedFeatures = sequentialfs(@selectionCriterionActivity, Features', Target', ...
        'options', options, 'nfeatures', nFeatures);
    diary('off');
    saveBestFeatures(filePath, selectedFeatures);
end

