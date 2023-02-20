clear

% CONSTANTS
FOLDER_PATH = 'dataset';

WITHOUT_WIN_METHOD = 'WITHOUT_WIN_METHOD';
CONTIGUOUS_WIN_METHOD = 'CONTIGUOUS_WIN_METHOD';
OVERLAPPED_WIN_METHOD = 'OVERLAPPED_WIN_METHOD';

FEATURE_NUM = 13;
FEATURE_MATRIX_ROWS_NUM = 66; % n. of subjects * n. of activities
FEATURE_MATRIX_COLUMNS_NUM = 11 * FEATURE_NUM; % n. of signals * n. of features
CONTIGUOUS_WIN_NUM = 4; % n. of contiguous windows
OVERLAPPED_WIN_NUM = 7; % n. of overlapped windows

FeaturesWithoutWin = zeros(FEATURE_MATRIX_ROWS_NUM, FEATURE_MATRIX_COLUMNS_NUM);

FeaturesContiguousWin = zeros(FEATURE_MATRIX_ROWS_NUM,...
    FEATURE_MATRIX_COLUMNS_NUM * CONTIGUOUS_WIN_NUM);

FeaturesOverlappedWin = zeros(FEATURE_MATRIX_ROWS_NUM,...
    FEATURE_MATRIX_COLUMNS_NUM * OVERLAPPED_WIN_NUM);


FileList = dir(fullfile(FOLDER_PATH, '*.csv'));

timeseriesCounter = 0;

for m = 1:length(FileList)
    fileName = FileList(m).name;
    
    if (contains(fileName, 'timeseries'))
        timeseriesCounter = timeseriesCounter + 1;
        disp(timeseriesCounter);
        TimeseriesTable = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:L');
        TimeseriesMatrix = table2array(TimeseriesTable);
        
        % Remove outliers
        if (max(isoutlier(TimeseriesMatrix)) == 1) 
            disp('Outlier detected')
        end
        
        TimeseriesMatrix = filloutliers(TimeseriesMatrix, 'linear');

        % Feature extraction (without window)
        FeaturesWithoutWin(timeseriesCounter, :) = extractFeatures(TimeseriesMatrix, WITHOUT_WIN_METHOD); 



    end
    
end