function [Features] = extractFeatures(TimeseriesMatrix, method, windowsNum)
    if method == 'WITHOUT_WIN_METHOD'
        Features = extractFeaturesWithoutWin(TimeseriesMatrix);
    end

    %if method == 'CONTIGUOUS_WIN_METHOD'

    %    Features = extractFeaturesWithWin
    %end

    %if method = 'OVERLAPPED_WIN_METHOD'
    %    Features = extractFeaturesWithWin
    %end 
end

function [Features] = extractFeaturesWithoutWin(TimeseriesMatrix)
    Features = computeFeatures(TimeseriesMatrix);
end

%function extractFeaturesWithWin(TimeseriesMatrix, windowsNum, )

%end

function [SignalFeatures] = computeFeatures(Timeseries)
    % Computes 14 features from a given signal

    % Feature 1: Mean
    mean_signal = mean(Timeseries);

    % Feature 2: Standard deviation
    std_signal = std(Timeseries);

    % Feature 3: Maximum amplitude
    max_amp = max(Timeseries);

    % Feature 4: Minimum amplitude
    min_amp = min(Timeseries);

    % Feature 5: Median
    median_signal = median(Timeseries);

    % Feature 6: Interquartile range
    IQR_signal = iqr(Timeseries);

    % Feature 7: Skewness
    skewness_signal = skewness(Timeseries);

    % Feature 8: Kurtosis
    kurtosis_signal = kurtosis(Timeseries);

    % Feature 9: Mean absolute deviation
    mad_signal = mad(Timeseries);

    % Feature 10: Root mean square
    rms_signal = rms(Timeseries);

    % Feature 11: Peak to peak amplitude
    peak_to_peak_amp = peak2peak(Timeseries);

    % Feature 12: Waveform length
    waveform_length = sum(abs(diff(Timeseries)));

    % Feature 13: Power spectral density
    [pxx, ~] = pwelch(Timeseries);
    psd = bandpower(pxx);

    % Store features in a matrix
    SignalFeatures = [mean_signal; std_signal; max_amp; min_amp; median_signal;...
                IQR_signal; skewness_signal; kurtosis_signal;...
                mad_signal; rms_signal; peak_to_peak_amp; waveform_length; psd];
    SignalFeatures = SignalFeatures(:)';
end

