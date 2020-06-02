% LBRES 2206
% Hydrologic time series diagnostic
% Alice Alonso
% alice.alonso@uclouvain.be
% March 2020

%% Preliminary Steps: Definition of user variables 
% Start fresh: clear any variables in the workspace, clear the command window, and close any open figures
clc; clear all; close all

% First define here the font size you want to have on your figures.  
% This is a useful trick to quickly adapt yr figures for oral presentation support
% (larger font needed), or written reports (smaller font needed)

fs = 12;

%% Load and Format the Data 
%
% Specify the current working directory 
cd '~/Dropbox/PROJECTS/HydroTSDiagnostic';

%% DATA PREPARATION
load Data/DataCongoRiver DataCongoRiver

% Reshape data from matrix to column
dMat = table2array(DataCongoRiver);

[m,n] = size(dMat);
dVec = [];
M = [];
D = [];
Y = [];

for i = 1:31:m
    
    % 31x12 matrices of discharge, day,  month and year
    tempd = dMat(i:i+30, 1:12);
    tempD = repmat([1:31]', 1,12);
    tempM = repmat([1:12], 31, 1);
    tempY = repmat(dMat(i, 14), 31, 12);
    
    % vectors of discharge, day,  month and year
    tempd = reshape(tempd,[],1);
    tempD = reshape(tempD,[],1);
    tempM = reshape(tempM,[],1);
    tempY = reshape(tempY,[],1);
    
    % Remove the 31st of Apr, Jun, Sep and Nov, and 29, 30 of Feb
    idx = [60, 61, 62, 31*4, 31*6, 31*9, 31*11];
    tempd (idx) = []; 
    tempD (idx) = []; 
    tempM (idx) = []; 
    tempY (idx) = []; 
    
    % Append in final vectors
    dVec = [dVec; reshape(tempd,[],1)];
    D = [D; reshape(tempD,[],1)];
    M = [M; reshape(tempM,[],1)];
    Y = [Y; reshape(tempY,[],1)];
end

tNum = datenum([Y,M,D]);
tStr = datestr(tNum);

% Create a table Qd and store the daily data
Qd = table;
Qd.tStr = tStr;
Qd.tNum = tNum;
Qd.Q = dVec;

save Data/DataCongoRiver Qd -append

%% Plot daily data
close
figure
plot(Qd.tNum, Qd.Q, 'k')
xlabel('Time', 'fontsize', fs)
ylabel('Discharge (m3/s)', 'fontsize', fs)
set(gca, 'fontsize', fs)
datetick
axis tight


%%
% <html><h3> Handle Missing Values</h3></html>
%
% Count missing values
idxMV = find(isnan(Qd.Q));
nbMV = length(idxMV);
pMV = nbMV/length(Qd.Q)*100; 
X = sprintf('There are %d missing values, which is equivalent to %d  percent of the total nber of data',nbMV,pMV);
disp(X)

% Display missing values
figure, hold on
plot(Qd.tNum, Qd.Q)
plot(Qd.tNum(idxMV,1), 80000*ones(nbMV,1) , 'r.','markersize', 10)
xlabel('Time', 'fontsize', fs)
ylabel('Discharge (m3/s)', 'fontsize', fs)
datetick
axis tight

% Fill in missing values
% First look at the lenght of time period during which data are missing
di = diff(idxMV);

% We see that MVs are mostly isolated, that there is no large data gap 
% (i.e. when consecutive occurences where diff(idx) =  1). 
% Therefore, it is reasonable to fill the missing values using
% interpolation techniques. Here, we use spline interpolation.

idxQ = find (~isnan(Qd.Q)); %index of all data but MVs
Qd.Qrec = interp1(Qd.tNum(idxQ), Qd.Q(idxQ), Qd.tNum, 'spline');

figure; hold on
plot(Qd.tNum, Qd.Qrec)
plot(Qd.tNum(idxMV), Qd.Qrec(idxMV), 'r.', 'markersize', 10) 
xlabel('Time', 'fontsize', fs)
ylabel('Discharge(m3/s)', 'fontsize', fs)
legend('Measured', 'Reconstructed')

% Save the data in the DataCongoRiver Matlab structure
save Data/DataCongoRiver Qd -append

%%
% <html><h3>Calculate annual statistics</h3></html>
%%
Qy = table;
dv = datevec(Qd.tNum);
Qy.Year = unique (dv(:,1));

for i = 1:length(Qy.Year)
    idx = find(dv(:,1) == Qy.Year(i));
    Qy.Mean(i) = nanmean(Qd.Q(idx));
    Qy.Std(i) = nanstd(Qd.Q(idx));
    Qy.p50(i) = median(Qd.Q(idx));
    Qy.p75(i) = prctile(Qd.Q(idx), 75);
    Qy.p25(i) = prctile(Qd.Q(idx), 25);
    Qy.p90(i) = prctile(Qd.Q(idx), 75);
    Qy.p10(i) = prctile(Qd.Q(idx), 25);
    Qy.Min(i) = min(Qd.Q(idx));
    Qy.Max(i) = max(Qd.Q(idx));
end
clear idx i dv 


%%
% <html><h3>Plot the time series data</h3></html>
%
% Daily Data
figure
plot(Qd.tNum, Qd.Qrec)
xlabel('Time', 'fontsize', fs)
ylabel('Discharge(m3/s)', 'fontsize', fs)
datetick
close 

% Repeat for time series of annual statistics
figure
subplot 311 ; hold on
plot(Qy.Year, Qy.Mean, 'k', 'linewidth', 2)
plot(Qy.Year, Qy.Mean + Qy.Std, 'k--')
plot(Qy.Year, Qy.Mean - Qy.Std, 'k--')
ylabel('Discharge (m^3/s)', 'fontsize', fs)
title('Mean - Std')
axis tight
subplot 312 ; hold on
plot(Qy.Year, Qy.p50, 'k', 'linewidth', 2)
plot(Qy.Year, Qy.Max, 'k--')
plot(Qy.Year, Qy.Min, 'k--')
ylabel('Discharge (m^3/s)', 'fontsize', fs)
title('Median - min/max')
axis tight
subplot 313 ; hold on %Inter quartile range
plot(Qy.Year, Qy.p50, 'k', 'linewidth', 2)
plot(Qy.Year, Qy.p75, 'k--')
plot(Qy.Year, Qy.p25, 'k--')
ylabel('Discharge (m^3/s)', 'fontsize', fs)
title('Median - p25/p75')
axis tight

%%
% <html><h3>Heat Map</h3></html>
%
% BASED ON CALENDAR YEAR
% Rearrange the daily Discharge data into a matrix with one raw per year`

dv = datevec(Qd.tNum);
years = unique (dv(:,1));
n = length(years);
Qyd = NaN(n, 366);

for i = 1:n; %For every year of the data record
    idx = find(dv(:,1) == years(i));
    if length(idx)<365 % Consider only the year for which we have the full data set
    Qyd(i,:) = NaN;
    else
    ddi = log10(Qd.Qrec(idx)); 
    Qyd(i,1:length(ddi)) =ddi';
    end
end

figure
pcolor(Qyd)
colormap('jet')
shading('flat')
shading interp

set(gca,'YTick',0:10:years(end)-years(1));
set(gca,'YTickLabel',years(1):10:years(end));
xlabel('DOY','FontSize', fs)
col = colorbar('location','southoutside','FontSize',fs);
set(get(col,'xlabel'),'string','Log10 Discharge (m^3/s)','FontSize',fs);
set(gca, 'FontSize',fs);

clear dv years n i idx ddi 
%%
% <html><h3>Descriptive Statistics</h3></html>
%
%% Plot the histogram of the data 
close all
figure 
subplot 121

H = histogram(Qd.Qrec, 50)

xlabel('Discharge(m^3/s)', 'fontsize', fs)
ylabel('Count', 'fontsize', fs)
set (gca, 'fontsize', fs)

subplot 122
H = histogram(Qd.Qrec, 10)

xlabel('Discharge(m^3/s)', 'fontsize', fs)
ylabel('Count', 'fontsize', fs)
set (gca, 'fontsize', fs)
%%
% Plot the boxplot of the data
figure 
subplot('position', [.1 .1 .15 .8])
boxplot(Qd.Qrec,'Labels', 'Annual')
ylabel('Discharge (m^3/s)', 'fontsize', fs)
set (gca, 'fontsize', fs)

% Plot the boxplot of data by month. 
% What is the water year? Is the variability of the discharge higher during some months more than
% others?
dv = datevec(Qd.tNum);
month_num = dv(:,2);
MonthLab = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
subplot('position', [.3 .1 .65 .8])
boxplot(Qd.Q, month_num, 'Labels', MonthLab);
ylabel('Discharge (m^3/s)', 'fontsize', fs)

%% Calculate the moments and percentiles
% 
% * mean and standart deviation
% * median
% * percentiles 25, 50 and 75
% * coefficient of skewness
% * coefficient of kurtosis
% 
% You can use either Matlab built-in functions, or code them yourself!
% In the discussion, make sure to discuss the values of those coefficients
% in light of the histograms and boxplots


%%
% <html><h3>Probability Distribution Function</h3></html>
%
% * Plot the probability density function, and cumulative density function

% Make sure to understand how the transformations between the
% histogram and the pdf and cdf are made.
% Based on these graphs, reply to the following questions:
% 
% * What is the probability to have a Discharge higher than 200m3/s?
% * What is the Discharge magnitude that is not exceeded more than 90% of
% the time?
% * How does the shape of the pdf change when you increase/ decrease the number of bins? (e.g. try with 50, 100, 200 bins) 

figure
% PDF
subplot 121
Qpdf = histogram(Qd.Qrec, 100, 'Normalization', 'pdf');

bar((Qpdf.BinEdges(1:end-1)+Qpdf.BinEdges(2:end))/2, Qpdf.Values, 'k')
xlim([min(Qd.Qrec), max(Qd.Qrec)])
xlabel('Discharge (m^3/s)', 'fontsize', fs)
ylabel('pdf', 'fontsize', fs)

%CDF:  use one of the two options ((un)comment one or the other). They both do equivalent job, but are there 
% to help undersand how such CDF is built
subplot 122
% Option 1: Using the histogram function with CDF normalization
Qcdf = histogram(Qd.Qrec, 100, 'Normalization', 'cdf');
plot((Qcdf.BinEdges(1:end-1)+Qcdf.BinEdges(2:end))/2, Qcdf.Values, 'k', 'marker', '.', 'markersize', 10, 'markeredgecolor', 'k')

% % Option 2: Using the cumulated sum of the pdf
% Qpdf = histogram(Qd.Qrec, 100, 'Normalization', 'pdf');
% plot((Qpdf.BinEdges(1:end-1)+Qpdf.BinEdges(2:end))/2, cumsum(Qpdf.Values)*Qpdf.BinWidth, 'k', 'linewidth', 2)
grid on
xlim([min(Qd.Qrec), max(Qd.Qrec)])
xlabel('Discharge(m3/s)', 'fontsize', fs)
ylabel('cdf', 'fontsize', fs)

%% Let's now test whether or not the data follow a normal distribution
% Let's first look at it visually: 
% Compare cdf with normal(cdf1), lognormal (cdf2) and exponential (cdf3)distributions. 
% Does one seem to fit best than the other?

figure
subplot 121
[pd1,pd2,pd3] = createFitPdf(Qd.Qrec);

subplot 122
[cd1,cd2,cd3] = createFitCdf(Qd.Qrec);

% Using the Kolomogorov-Smirnof test that test the null Hypothesis that the sample is drawn from a normal distribution. 
[H,p] = kstest(Qd.Qrec) 

%%
% <html><h3>Flow duration curve</h3></html>
%
% Answer the following questions: 
% * What is the probability to have a Discharge higher than 50,000m3/s?
% * What is the Discharge magnitude that is not exceeded more than 90% of
% the time?

clear Qpdf Qcdf pdf1 pdf2 pdf3

Qcdf = histogram(Qd.Qrec, 1000, 'Normalization', 'cdf');

figure; hold on
plot((1-Qcdf.Values)*100,(Qcdf.BinEdges(1:end-1)+Qcdf.BinEdges(2:end))/2, 'k')%, 'marker', '.', 'markersize', 10, 'markeredgecolor', 'k')
grid on
ylim([min(Qd.Qrec), max(Qd.Qrec)])
ylabel('Discharge(m3/s)', 'fontsize', fs)
xlabel('Time exceeded (%)', 'fontsize', fs)
set(gca, 'fontsize', fs)

%%
% <html><h3>Auto Correlation Function</h3></html>
%
p = 100 %  total number of lags to consider
close all
figure
ta = acf(Qd.Qrec, p)
xlabel('Time lag', 'fontsize', fs)
ylabel('Autocorrelation coeff (-)', 'fontsize', fs)
set(gca, 'fontsize', fs)
ylim([-1 1])

%%
% <html><h3>Trend analysis</h3></html>
%
% Remember that it is good practice to remove seasonnality when conducting trend
% analysis.
% To remedy, let's run the trend test on annual and monthly data.

%% ON ANNUAL DATA
% Let's run the non parametric and parametric trend analysis for the six annual statistics calculated
% above.
% Is there a significant trend? Upwards or downards?
% Save statistics in Trend_y table
Trend_y = table('size', [6,8],'VariableTypes', {'double','double','double','double','double','double','double','double'});
Trend_y.Properties.VariableNames = {'p', 'Z', 'Int', 'Slope', 'R2', 'Fstat', 'Pval', 'Sigma'};
Trend_y.Properties.RowNames = {'Mean', 'p10','p25', 'p50', 'p75','p90'}

fig = figure
set(fig , 'units' , 'centimeters' , 'position' , [10 , 10 , 25 , 40])
fig.Color = 'white';

    % Mean
    [H,p,Z] = Kendall(Qy.Mean, 0.05);
    [b,~,~, ~, stats] = regress (Qy.Mean, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_y(1,:) =  statTemp;
    
    subplot 611
    hold on
    plot(Qy.Year, Qy.Mean, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Mean')
    xlim([1900 2005])
    set(gca, 'xticklabels', '')


    % p10
    [H,p,Z] = Kendall(Qy.p10, 0.05);
    [b,~,~, ~, stats] = regress (Qy.p10, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_y(2,:) = statTemp;
    
    
    subplot 612
    hold on
    plot(Qy.Year, Qy.p10, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('p10')
    xlim([1900 2005])
    set(gca, 'xticklabels', '')
   
    % p25
    [H,p,Z] = Kendall(Qy.p25, 0.05);
    [b,~,~, ~, stats] = regress (Qy.p25, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_y(3,:) = statTemp;
    
        subplot 613
    hold on
    plot(Qy.Year, Qy.p25, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('p25')
    xlim([1900 2005])
    set(gca, 'xticklabels', '')
    ylabel('Monthly discharge (m^3/s)')
    
    % p50
    [H,p,Z] = Kendall(Qy.p50, 0.05);
    [b,~,~, ~, stats] = regress (Qy.p50, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_y(4,:) = statTemp;
    
            subplot 614
    hold on
    plot(Qy.Year, Qy.p50, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Median')
    xlim([1900 2005])
    set(gca, 'xticklabels', '')
      
        % p75
    [H,p,Z] = Kendall(Qy.p75, 0.05);
    [b,~,~, ~, stats] = regress (Qy.p75, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_y(5,:) = statTemp;
    
            subplot 615
    hold on
    [b,~,~, ~, stats] = regress (Qy.p75, [Qy.Year, ones(length(Qy.Year),1)]);
    plot(Qy.Year, Qy.p75, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('p75')
    xlim([1900 2005])
    set(gca, 'xticklabels', '')
       
        % p90
    [H,p,Z] = Kendall(Qy.p90, 0.05);
    [b,~,~, ~, stats] = regress (Qy.p90, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_y(6,:) = statTemp;
    
                subplot 616
    hold on
    [b,~,~, ~, stats] = regress (Qy.p90, [Qy.Year, ones(length(Qy.Year),1)]);
    plot(Qy.Year, Qy.p90, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('p90')
    xlim([1900 2005])
   
writetable(Trend_y, 'Trend_y.csv','WriteRowNames',true)  

%% ON MONTHLY DATA
% Calculate median discharge on a monthly basis, make plots and run Man Kendall test and regression analysis

fig = figure
set(fig , 'units' , 'centimeters' , 'position' , [10 , 10 , 25 , 40])
fig.Color = 'white';

month = {'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'};        
for m = 1:12
 
    mD_p50 = [];    
    
    % Build vectors with median discharge for each month 
for i = 1:length(Qy.Year)
    dv = datevec(Qd.tNum);
    idx = find(dv(:,1) == Qy.Year(i) & dv(:,2) == m);
    mdatai = Qd.Qrec(idx);
    mD_p50 = [mD_p50; prctile(mdatai, 50)]; 
end

[H,p,Z] = Kendall(mD_p50, 0.05);
[b,~,~, ~, stats] = regress (mD_p50, [Qy.Year, ones(length(Qy.Year),1)]);
Trend_m(m,1:2) = table(p,Z); 
Trend_m(m,3:8) = table(b(2), b(1), stats(1), stats(2), stats(3), stats(4));

    subplot(6,2,m)
    hold on
    plot(Qy.Year, mD_p50, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title(month{m})
    if m == 7
        ylabel('max monthly discharge (m^3/s)')
    end

end

Trend_m.Properties.VariableNames = {'p', 'Z', 'Int', 'Slope', 'R2', 'Fstat', 'Pval', 'Sigma'};
Trend_m.Properties.RowNames = month;
writetable(Trend_m, 'Trend_m.csv','WriteRowNames',true)  


%%
% <html><h3> Coefficient of variation of daily data with a user-defined moving window </h3></html>
%
% How the the variability change through time? Is what you observe with discharge data similar to what you
% observe with rainfall data? Discuss briefly.

Ws = 20*365; % Width of the moving window
l = length(Qd.Qrec);
CV = NaN(l,1);

for i = 1+Ws/2 : l-Ws/2
    d = Qd.Qrec(i-Ws/2 : i+Ws/2);
    CV(i) = nanstd(d)/nanmean(d);
end

figure 
plot(Qd.tNum, CV, 'k', 'linewidth', 2)
datetick
axis tight
ylabel('CV (Q)', 'fontsize', fs)
set(gca,  'fontsize', fs)

clear Ws l CV i d 

%%
% <html><h3>Spectral Analysis:Fourier Transform (FT)</h3></html>
%

% Remove mean
mu = nanmean(Qd.Qrec);
Qd.Qrec21 = Qd.Qrec-mu;

% Detrend (if any trend)
L = length(Qd.Qrec);
p = polyfit(1:L,Qd.Qrec21',1);
reg = polyval(p,1:L);
Qd.Qrec22 = Qd.Qrec21-reg';

% Use a window to taper the series and so reduce end effects (Gibbs phenomenon)
% Here, we use a Hanning window
n = [1:L];
w = 0.5*(1-cos(2*pi()*n./L)); % Hanning window
d = Qd.Qrec22.*w';

% Plot raw and prepared data for inspection
close
figure
subplot 411
plot(Qd.tNum, Qd.Qrec)
title('raw')
subplot 412; hold on
plot(Qd.tNum, Qd.Qrec21)
plot(Qd.tNum, reg)
title('mean removed')
subplot 413
plot(Qd.tNum, Qd.Qrec22)
title('trend removed')
subplot 414
plot(Qd.tNum, d)
title('padded with Hanning window')


%% Fast Fourrier Transform
% Note: best practice would be to perform a segmented averaged FFT with
% confidence interval

Fs = 365;          % Sampling frequency  (365 measurements/year)             
T = 1/Fs;          % Sampling period       
L = length(Qd.Qrec22);  % Length of signal
t = (0:L-1)*T;     % Time vector
f = Fs*(1:L)/L; % Frequency vector (cycles/year)


% Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
P2 = abs(fft(d)/L); % Conduct FFT and take real part
P2 = 8/3*P2; % Rescale. 8/3 = rescaling factor for Hanning window 
P1 = P2(1:L/2);
f = f(1:L/2);

figure
loglog(f,P1, 'k')
grid on
xlabel('Fréquence (cycle par an)', 'fontsize', fs)
ylabel('abs(FFT)', 'fontsize', fs)
xlim([.01, 100])
ylim([.1 10^5])
set(gca,  'fontsize', fs)
 
%%
% <html><h3>Wavelet Analysis</h3></html>
%
cwt(Qd.Qrec22,years(1/(365)));
