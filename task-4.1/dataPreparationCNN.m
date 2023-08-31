clear; clc; close all;

% CONSTANTS
FOLDER_PATH = 'dataset';
N_MEASUREMENTS = 66;
WINDOW_SIZE = 5000;

Timeseries = cell(N_MEASUREMENTS, 1);
Target = cell(N_MEASUREMENTS, 1);

FileList = dir(fullfile(FOLDER_PATH, '*.csv'));

timeseriesCounter = 0;
targetCounter = 0;

for m = 1:length(FileList)
    fileName = FileList(m).name;
    
    if (contains(fileName, 'timeseries'))
        timeseriesCounter = timeseriesCounter + 1;
        disp(['Timeseries nr. ' num2str(timeseriesCounter)]);
        TimeseriesTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:L');
        TimeseriesMatrix = table2array(TimeseriesTable);
        
        % Remove outliers
        TimeseriesMatrix = filloutliers(TimeseriesMatrix, 'linear');
        Timeseries{timeseriesCounter} = TimeseriesMatrix';

    else
        targetCounter = targetCounter + 1;
        TargetTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:B');
        TargetECG = table2array(TargetTable); 
        Target{targetCounter} = TargetECG';
    end
end