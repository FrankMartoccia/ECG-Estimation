clear

% CONSTANTS
FOLDER_PATH = 'dataset';

WITHOUT_WIN_METHOD = "WITHOUT_WIN_METHOD";
CONTIGUOUS_WIN_METHOD = "CONTIGUOUS_WIN_METHOD";
OVERLAPPED_WIN_METHOD = "OVERLAPPED_WIN_METHOD";

N_FEATURES = 13;
N_FEATURES_MATRIX_ROWS = 66; % n. of subjects * n. of activities
N_FEATURES_MATRIX_COLUMNS = 11 * N_FEATURES; % n. of signals * n. of features
N_CONTIGUOUS_WIN_NUM = 4; % n. of contiguous windows
N_OVERLAPPED_WIN_NUM = 7; % n. of overlapped windows

FeaturesWithoutWin = zeros(N_FEATURES_MATRIX_ROWS, N_FEATURES_MATRIX_COLUMNS);

FeaturesContiguousWin = zeros(N_FEATURES_MATRIX_ROWS,...
    N_FEATURES_MATRIX_COLUMNS * N_CONTIGUOUS_WIN_NUM);

FeaturesOverlappedWin = zeros(N_FEATURES_MATRIX_ROWS,...
    N_FEATURES_MATRIX_COLUMNS * N_OVERLAPPED_WIN_NUM);

TargetMeanECG = zeros(N_FEATURES_MATRIX_ROWS, 1);

TargetStdECG = zeros(N_FEATURES_MATRIX_ROWS, 1);

TargetActivity = zeros(N_FEATURES_MATRIX_ROWS, 1);


FileList = dir(fullfile(FOLDER_PATH, '*.csv'));

timeseriesCounter = 0;
targetCounter = 0;

for m = 1:length(FileList)
    fileName = FileList(m).name;
    
    if (contains(fileName, 'timeseries'))
        timeseriesCounter = timeseriesCounter + 1;
        disp(timeseriesCounter);
        TimeseriesTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:L');
        TimeseriesMatrix = table2array(TimeseriesTable);
        
        % Check outliers
        if (max(isoutlier(TimeseriesMatrix)) == 1) 
            disp('Outlier detected')
        end
        % Remove outliers
        TimeseriesMatrix = filloutliers(TimeseriesMatrix, 'linear');

        % Feature extraction (without window)
        FeaturesWithoutWin(timeseriesCounter, :) = extractFeatures(TimeseriesMatrix, WITHOUT_WIN_METHOD); 
        
        % Feature extraction (with contiguous windows)
        FeaturesContiguousWin(timeseriesCounter, :) = extractFeatures(TimeseriesMatrix,...
            CONTIGUOUS_WIN_METHOD, N_CONTIGUOUS_WIN_NUM); 
        
        % Feature extraction (with overlapped windows)
        FeaturesOverlappedWin(timeseriesCounter, :) = extractFeatures(TimeseriesMatrix,...
            OVERLAPPED_WIN_METHOD, N_OVERLAPPED_WIN_NUM);         

    else
        targetCounter = targetCounter + 1;
        TargetTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:B');
        TargetECG = table2array(TargetTable);
        TargetMeanECG(targetCounter, 1) = mean(TargetECG);
        TargetStdECG(targetCounter, 1) = std(TargetECG);
        
        if contains(fileName, "run")
            TargetActivity(targetCounter, 1) = 1;
        elseif contains(fileName, "sit")
            TargetActivity(targetCounter, 1) = 2;
        else
            TargetActivity(targetCounter, 1) = 3;
        end
    end
    
end

save('data/BeforeNormalization', 'FeaturesWithoutWin', 'FeaturesContiguousWin', ...
    'FeaturesOverlappedWin', 'TargetMeanECG', 'TargetStdECG', 'TargetActivity');


function [normalizedFeatures] = normalizeFeatures(inputFeatures)
    minValues = min(inputFeatures);
    
    normalizedFeatures = (inputFeatures - minValues) ./ (max(inputFeatures) - minValues);
end


