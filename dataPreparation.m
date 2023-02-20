clear

% CONSTANTS

FOLDER_PATH = 'dataset';

FEATURE_MATRIX_ROWS_NUM = 66; % n. of subjects * n. of activities
FEATURE_MATRIX_COLUMNS_NUM = 121; % n. of signals * n. of features
CONTIGUOUS_WIN_NUM = 4;
OVERLAPPED_WIN_NUM = 7;

Features_no_win = zeros(FEATURE_MATRIX_ROWS_NUM, FEATURE_MATRIX_COLUMNS_NUM);

Features_contiguous_win = zeros(FEATURE_MATRIX_ROWS_NUM,...
	FEATURE_MATRIX_COLUMNS_NUM * CONTIGUOUS_WIN_NUM);

Features_overlapped_win = zeros(FEATURE_MATRIX_ROWS_NUM,...
	FEATURE_MATRIX_COLUMNS_NUM * OVERLAPPED_WIN_NUM);


FileList = dir(fullfile(FOLDER_PATH, '*.csv'));


Timeseries = cell(FEATURE_MATRIX_ROWS_NUM, 1);

counter = 0;
for m = 1:length(FileList)
    fileName = FileList(m).name;
    
    if (contains(fileName, 'timeseries'))
    	counter = counter + 1;
    	disp(counter);
        Timeseries{counter} = readtable(fullfile(FOLDER_PATH, fileName), 'Range', 'B:L');
    end
    
end
