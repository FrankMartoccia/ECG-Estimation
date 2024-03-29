clear; clc; close all;

% CONSTANTS
FOLDER_PATH = 'dataset';
N_MEASUREMENTS = 66;
WINDOW_SIZE = 3000;

Timeseries = cell(N_MEASUREMENTS, 1);
Target = cell(N_MEASUREMENTS, 1);

FileList = dir(fullfile(FOLDER_PATH, '*.csv'));

timeseriesCounter = 0;
targetCounter = 0;

% Timeseries and Target matrices are filled with all the values of the samples 
% by scanning the files in the dataset folder
for m = 1:length(FileList)
    fileName = FileList(m).name;
    disp(fileName)
    if (contains(fileName, 'timeseries'))
        timeseriesCounter = timeseriesCounter + 1;
        disp(['Timeseries nr. ' num2str(timeseriesCounter)]);
        TimeseriesTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:L');
        TimeseriesMatrix = table2array(TimeseriesTable);
        
        % Data normalization
        TimeseriesMatrix = normalize(TimeseriesMatrix', 'scale');

        Timeseries{timeseriesCounter} = TimeseriesMatrix;

    else
        targetCounter = targetCounter + 1;
        TargetTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:B');
        TargetECG = table2array(TargetTable); 
        Target{targetCounter} = TargetECG';
    end
end

% Deleted noisy data (subject 13 walking activity)
Timeseries(12) = [];
Target(12) = [];

nWindows = 0;

% Computation of the total number of windows
for i = 1 : N_MEASUREMENTS - 1
    nSamples = length(Timeseries{i});
    nWindows = nWindows + floor(nSamples / WINDOW_SIZE);
end

% Initialization of the final data structures
TimeseriesCNN = cell(nWindows, 1);
TargetCNN = zeros(nWindows, 1);

currentWindow = 1;

% TimeseriesCNN and TargetCNN are filled with samples in windows of width equal to WINDOW_SIZE
for measurement = 1 : N_MEASUREMENTS - 1
    nSamples = size(Timeseries{measurement}, 2);
    currentStartIdx = 1;
    currentEndIdx = WINDOW_SIZE;

    while currentEndIdx <= nSamples
        TimeseriesCNN{currentWindow} = Timeseries{measurement}(:, currentStartIdx : currentEndIdx);
        TargetCNN(currentWindow) = std(Target{measurement}(1, currentStartIdx : currentEndIdx));
        currentWindow = currentWindow + 1;
        currentStartIdx = currentStartIdx + WINDOW_SIZE;
        currentEndIdx = currentEndIdx + WINDOW_SIZE;
    end
end

% Outliers removal
[TargetCNN, to_remove] = rmoutliers(TargetCNN);
TimeseriesCNN = TimeseriesCNN(~to_remove);
fprintf("Removed %i outliers \n", sum(to_remove));

save('data/inputCNN.mat', 'TimeseriesCNN', 'TargetCNN')