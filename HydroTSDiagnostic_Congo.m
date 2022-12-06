% LBRES 2206
% Hydrologic time series diagnostic
% Alice Alonso
% alice.alonso@uclouvain.be
% November 2021

%% Preliminary Steps: Prepare the workspace and define user inputs
% Start fresh: clear any variables in the workspace, clear the command window, and close any open figures
clc; clear all; close all

% First define here the font size you want to have on your figures.  
% This is a useful trick to quickly adapt yr figures for oral presentation support
% (larger font needed), or written reports (smaller font needed)

fs = 12; 

%%
% <html><h3> Specify the current working directory </h3></html>
%

cd '/Users/alonso/Library/CloudStorage/OneDrive-UCL/07_COURS/LBRES2206_HydroAvancee/TP/StudyCaseCongoRiver'

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
d = table;
d.tStr = tStr;
d.datenum = tNum;
d.Q = dVec;

save Data/DataCongoRiver d -append

%% Plot daily data

figure
plot(d.datenum, d.Q, 'k')
%xlabel('Time', 'fontsize', fs)
ylabel('Discharge (m3/s)', 'fontsize', fs)
set(gca, 'fontsize', fs)
datetick
axis tight

%%
% <html><h3> Handle Missing Values</h3></html>
%

% Start with removing the NaNs at the beginning of the time series, before
% the monitoring started
idxF = find(~isnan(d.Q), 1); % trouver la position de la premiere date pour laquelle on a une donnée de débit
d = d([idxF:end],:);

% Count missing values
idxMV = find(isnan(d.Q));
nbMV = length(idxMV);
pMV = nbMV/length(d.Q)*100; 
X = sprintf('There are %d missing values, which is equivalent to %d  percent of the total nber of data',nbMV,pMV);
disp(X)

% Display missing values
figure, hold on
plot(d.datenum, d.Q)
plot(d.datenum(idxMV,1), ones(nbMV,1) , 'r.','markersize', 10)
xlabel('Time', 'fontsize', fs)
ylabel('Discharge (m3/s)', 'fontsize', fs)
datetick
axis tight

% Decide on how to handle the missing values, if any

% Start by looking at the graph: how are the missing values distributed?

% If we see that MVs are mostly isolated, that there is no large data gap 
% (i.e. when consecutive occurences where diff(idx) =  1), it is reasonable 
% to fill the missing values using interpolation techniques. 
% Here, we propose to use spline interpolation (uncomment if relevant).
% 
idxQ = find (~isnan(d.Q)); %index of all data but MVs
d.Q0 = d.Q;% We do this to keep the original data in case
d.Q = interp1(d.datenum(idxQ), d.Q(idxQ), d.datenum, 'spline');
% 

% If there are large chunks of missing values, a linear interpolation is
% not acceptable.  For this exercise, if we have such segment of missing
% values, we are just going to leave it unchanged. However, we will need to
% take that into account when analysing and discussing the outputs of the
% analyses.

clear di idxF idxMV idxQ pMV X pgon

%% Exploratory analysis of the data and Summary Statistics
%%
% <html><h3>HEAT MAP</h3></html>
%
% Rearrange the daily Discharge data into a matrix with one raw per year

dv = datevec(d.datenum);
YY = unique (dv(:,1));
n = length(YY);
Qyd = NaN(n, 366);

for i = 1:n; %For every year of the data record
    idx = find(dv(:,1) == YY(i));
%     if length(idx)<365 % Consider only the year for which we have a data
%     %for each day
%     Qyd(i,:) = NaN;
%     else
    ddi = d.Q(idx); 
    
    Qyd(i,1:length(ddi)) =ddi';
%     end
end

% Visualise the data 
% Taking the logarithmic values will help better visualise the annual
% variability, while the natural values highlight the extreme positives.
% Try both!

close all
figure
image(log10(Qyd(2:end, :)),'CDataMapping', 'scaled')
%image(Qyd(2:end, :),'CDataMapping', 'scaled')
colormap(flipud(pink))
%colormap((autumn))

shading('flat')
%shading interp

set(gca,'YTick',0:10:YY(end)-YY(1));
set(gca,'YTickLabel',YY(1):10:YY(end));
xlabel('DOY','FontSize', fs)
col = colorbar('location','southoutside','FontSize',fs);
set(get(col,'xlabel'),'string','Log10 (Discharge (m^3/s))','FontSize',fs);
%set(get(col,'xlabel'),'string','Discharge (m^3/s)','FontSize',fs);

set(gca, 'FontSize',fs);

clear dv n i idx ddi Qyd col 
%%
% <html><h3>SUMMARY STATISTICS</h3></html>
%

%%
% <html><h3>Plot the histogram of the data</h3></html>
%

figure 
subplot 121
H = histogram(d.Q, 50);

xlabel('Discharge(m^3/s)', 'fontsize', fs)
ylabel('Count', 'fontsize', fs)
set (gca, 'fontsize', fs)

subplot 122
H = histogram(d.Q, 10);

xlabel('Discharge(m^3/s)', 'fontsize', fs)
ylabel('Count', 'fontsize', fs)
set (gca, 'fontsize', fs)

clear H

%%
% <html><h3>Boxplots</h3></html>
%

figure 
subplot('position', [.1 .1 .15 .8])
boxplot(d.Q)
ylabel('Discharge (m^3/s)', 'fontsize', fs)
set(gca, 'fontsize', fs)

% Plot the boxplot of data by month. 
% What is the water year? Is the variability of the discharge higher during some months more than
% others?
dv = datevec(d.datenum);
month_num = dv(:,2);
MonthLab = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
subplot('position', [.3 .1 .65 .8])
boxplot(d.Q, month_num, 'Labels', MonthLab);
%ylabel('Discharge (m^3/s)', 'fontsize', fs)

clear dv month_num MonthLab 

%%
% <html><h3>Summary statistics</h3></html>
%
 
l = length(d.Q);

timeStart = d.datenum(1); 
timeEnd = d.datenum(end);
mean = sum(d.Q)/l; 
stddev = sqrt((sum((d.Q-mean).^2))/(l-1)); 
skewness = l*sum((d.Q-mean).^3)/((l-1)*(l-2)*stddev^3);
kurtosis = l^2*sum((d.Q-mean).^4)/((l-1)*(l-2)*(l-2)*stddev^4);
p25 = prctile(d.Q, 25);
p50 = prctile(d.Q, 50);
p75 = prctile(d.Q, 75);
IQR = p75-p25;
sumstats = table(timeStart, timeEnd, mean, stddev, skewness, kurtosis, p25, p50, p75, IQR)

clear l timeStart timeEnd mean stddev skewness kurtosis p25 p50 p75 IQR
%%
% <html><h3>Probability Distribution Function</h3></html>
%
% * Plot the probability density function, and cumulative density function

% Make sure to understand how the relation between the histogramn the pdf,
% and the cdf.
% Based on these graphs, reply to the following questions:
% 
% * What is the probability to have a Discharge higher than 100m3/s?
% * What is the Discharge magnitude that is not exceeded more than 90% of
% the time?
% * How does the shape of the histogram change when you increase/ decrease the number of bins? (e.g. try with 50, 100, 200 bins) 

figure
% PDF
subplot 121
Qpdf = histogram(d.Q, 10, 'Normalization', 'pdf');

bar((Qpdf.BinEdges(1:end-1)+Qpdf.BinEdges(2:end))/2, Qpdf.Values, 'k')
xlim([min(d.Q), max(d.Q)])
xlabel('Discharge (m^3/s)', 'fontsize', fs)
ylabel('pdf', 'fontsize', fs)

%CDF:  use one of the two options ((un)comment one or the other). They both do equivalent job, but are there 
% to help undersand how such CDF is built
subplot 122
% Option 1: Using the histogram function with CDF normalization
Qcdf = histogram(d.Q, 100, 'Normalization', 'cdf');
plot((Qcdf.BinEdges(1:end-1)+Qcdf.BinEdges(2:end))/2, Qcdf.Values, 'k', 'marker', '.', 'markersize', 10, 'markeredgecolor', 'k')

% % Option 2: Using the cumulated sum of the pdf
% Qpdf = histogram(d.Q, 100, 'Normalization', 'pdf');
% plot((Qpdf.BinEdges(1:end-1)+Qpdf.BinEdges(2:end))/2, cumsum(Qpdf.Values)*Qpdf.BinWidth, 'k', 'linewidth', 2)
grid on
xlim([min(d.Q), max(d.Q)])
xlabel('Discharge(m3/s)', 'fontsize', fs)
ylabel('cdf', 'fontsize', fs)

clear Qpdf Qcdf 

% Let's test whether or not the data follow a 'classical' probability distribution
% Let's first look at it visually: 
% Compare cdf with normal(cdf1), lognormal (cdf2) and exponential (cdf3)distributions. 
% Does one seem to fit best than the other?

figure
subplot 121
[pd1,pd2,pd3] = createFitPdf(d.Q);

subplot 122
[cd1,cd2,cd3] = createFitCdf(d.Q);

% Using the Kolomogorov-Smirnof test that test the null Hypothesis that the sample is drawn from a normal distribution. 
% H indicates the result of the hypothesis test:
%        H = 0 => Do not reject the null hypothesis at the 5% significance
%        level. 
%        H = 1 => Reject the null hypothesis at the 5% significance
%        level.
[H,p] = kstest(d.Q) 

clear pd1 pd2 pd3 cd1 cd2 cd3 H p

%%
% <html><h3>Flow duration curve</h3></html>
%
% Answer the following questions: 
% * What is the probability to have a Discharge higher than 20m3/s?
% * What is the Discharge magnitude that is not exceeded more than 90% of
% the time?
% * How does the flow duration curve compare with the CDF?

Qcdf = histogram(d.Q, 100, 'Normalization', 'cdf');

figure; hold on
plot((1-Qcdf.Values)*100,(Qcdf.BinEdges(1:end-1)+Qcdf.BinEdges(2:end))/2, 'k')%, 'marker', '.', 'markersize', 10, 'markeredgecolor', 'k')
grid on
axis([0,100,min(d.Q), max(d.Q)])
ylabel('Discharge (m3/s)', 'fontsize', fs)
xlabel('Time exceeded (%)', 'fontsize', fs)
set(gca, 'fontsize', fs)

clear Qcdf

%%
% <html><h3>Auto Correlation Function</h3></html>
%
p = 100; %  maximum time lags to consider
ta = acf(d.Q, p);

figure
plot([1:p], ta, 'k'); hold on
plot([1 100], [0 0], 'k--')

xlabel('Time lag (days)', 'fontsize', fs)
ylabel('Autocorrelation coeff (-)', 'fontsize', fs)
set(gca, 'fontsize', fs)
ylim([-1 1])

%% Trend analyses on the IHAs

% IHA defined and explained in 
%%
% _% Richter, B. D., Baumgartner, J. V., Powell, J. & Braun, D. P.
% A Method for Assessing Hydrologic Alteration within Ecosystems. 
% Conserv. Biol. 10, 1163?1174 (1996)._

%%
% <html><h3>GROUP 1: Magnitude of monthly water conditions </h3></html>
%

%% 
% Here, we take the median value for each calendar month

% Define de percentile to consider
perc = 50;

fig = figure;
set(fig , 'units' , 'centimeters' , 'position' , [10 , 10 , 25 , 40])
fig.Color = 'white';

Qy = table;
dv = datevec(d.datenum);
Qy.Year = unique (dv(:,1));
month = {'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'}; 

for m = 1:12
 
    mD_p = [];    
    
    % Build vectors with median discharge for each month 
for i = 1:length(Qy.Year)
    dv = datevec(d.datenum);
    idx = find(dv(:,1) == Qy.Year(i) & dv(:,2) == m);
    mdatai = d.Q(idx);
    mD_p = [mD_p; prctile(mdatai, perc)]; 
end

[H,p,Z] = Kendall(mD_p, 0.05);
[b,~,~, ~, stats] = regress (mD_p, [Qy.Year, ones(length(Qy.Year),1)]);
Trend_mG1(m,1:2) = table(p,Z); 
Trend_mG1(m,3:8) = table(b(2), b(1), stats(1), stats(2), stats(3), stats(4));

    subplot(6,2,m)
    hold on
    plot(Qy.Year, mD_p, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title(month{m})
    if m == 7
        ylabel('max monthly discharge (m^3/s)')
    end

end

Trend_mG1.Properties.VariableNames = {'p', 'Z', 'Int', 'Slope', 'R2', 'Fstat', 'Pval', 'Sigma'};
Trend_mG1.Properties.RowNames = month;
Trend_mG1

clear b fig H i idx m mD_p mdatai month p perc stats Z

%%  
% <html><h3>GROUP 2: Magnitude and duration of annual extreme water conditions </h3></html>
%

% *Calculate annual statistics*


% First, let's generate time series with 3, 7, 30, and 90 days means in
% preparation for the calculation of the IHA
 
d.Q3 = movmean(d.Q, [1 1]); % Application of a moving average with  a window  of size 1+1+1 = 3
d.Q7 = movmean(d.Q, [3 3]); % Application of a moving average with  a window  of size 3+3+1 = 7
d.Q31 = movmean(d.Q, [15 15]); % Application of a moving average with  a window  of size 15+15+1 = 31
d.Q91 = movmean(d.Q, [45 45]); % Application of a moving average with  a window  of size 45+45+1 = 91

% Then, calculate the annual stats and the indicators from Group 2

for i = 1:length(Qy.Year)
    idx = find(dv(:,1) == Qy.Year(i));
    Qy.Mean(i) = nanmean(d.Q(idx));
    Qy.Std(i) = nanstd(d.Q(idx));
    Qy.p50(i) = median(d.Q(idx));
    Qy.p75(i) = prctile(d.Q(idx), 75);
    Qy.p25(i) = prctile(d.Q(idx), 25);
    Qy.p90(i) = prctile(d.Q(idx), 90);
    Qy.p10(i) = prctile(d.Q(idx), 10);
    Qy.G2Min1(i) = min(d.Q(idx)); % IHA: Annual minima 1-day mean
    Qy.G2Max1(i) = max(d.Q(idx)); % IHA: Annual maxima 1-day mean
    Qy.G2Min3(i) = min(d.Q3(idx)); % IHA: Annual minima 3-day mean
    Qy.G2Max3(i) = max(d.Q3(idx)); % IHA: Annual maxima 3-day mean
    Qy.G2Min7(i) = min(d.Q7(idx)); % IHA: Annual minima 7-day mean
    Qy.G2Max7(i) = max(d.Q7(idx)); % IHA: Annual maxima 7-day mean
    Qy.G2Min31(i) = min(d.Q31(idx)); % IHA: Annual minima 31-day mean
    Qy.G2Max31(i) = max(d.Q31(idx)); % IHA: Annual maxima 31-day mean
    Qy.G2Min91(i) = min(d.Q91(idx)); % IHA: Annual minima 91-day mean
    Qy.G2Max91(i) = max(d.Q91(idx)); % IHA: Annual maxima 91-day mean
end

clear idx i 

% * Plot the time series of some of the annual statistics*

% Notice the diferences in the y-axis !

figure
subplot 311 ; hold on
plot(Qy.Year, Qy.Mean, 'k.-', 'linewidth', 2, 'markersize', 10)
plot(Qy.Year, Qy.Mean + Qy.Std, 'k--')
plot(Qy.Year, Qy.Mean - Qy.Std, 'k--')
ylabel('Discharge (m^3/s)', 'fontsize', fs)
title('Annual mean - mean +/-std dev')
axis tight

subplot 312 ; hold on 
plot(Qy.Year, Qy.p50, 'k.-', 'linewidth', 2, 'markersize', 10)
plot(Qy.Year, Qy.p75, 'k--')
plot(Qy.Year, Qy.p25, 'k--')
ylabel('Discharge (m^3/s)', 'fontsize', fs)
title('Annual median - p25/p75')
axis tight

subplot 313 ; hold on
plot(Qy.Year, Qy.p50, 'k.-', 'linewidth', 2, 'markersize', 10)
plot(Qy.Year, Qy.G2Max1, 'k--')
plot(Qy.Year, Qy.G2Min1, 'k--')
ylabel('Discharge (m^3/s)', 'fontsize', fs)
title('Annual median - min/max')
axis tight

%*Trend analysis on the indicators*

% Let's run the non parametric and parametric trend analysis for the indicators calculated
% above.
% Is there a significant trend? Upwards or downards?
% Save statistics in Trend_y table

Trend_yG2 = table('size', [12,8],'VariableTypes', {'double','double','double','double','double','double','double','double'});
Trend_yG2.Properties.VariableNames = {'p', 'Z', 'Int', 'Slope', 'R2', 'Fstat', 'Pval', 'Sigma'};
Trend_yG2.Properties.RowNames = {'Mean', 'Std', 'Min1','Max1', 'Min3', 'Max3','Min7', 'Max7','Min31', 'Max31','Min90', 'Max90'};

%close all
fig = figure;
set(fig , 'units' , 'centimeters' , 'position' , [10 , 10 , 25 , 40])
fig.Color = 'white';

xS = Qy.Year(1);
xE = Qy.Year(end);

    % Mean
    [H,p,Z] = Kendall(Qy.Mean, 0.05);
    [b,~,~, ~, stats] = regress (Qy.Mean, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(1,:) =  statTemp;
    
    subplot(6,2,1)
    hold on
    plot(Qy.Year, Qy.Mean, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Mean')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
     % StdDev
    [H,p,Z] = Kendall(Qy.Std, 0.05);
    [b,~,~, ~, stats] = regress (Qy.Std, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(2,:) =  statTemp;
    
    subplot(6,2,2)
    hold on
    plot(Qy.Year, Qy.Std, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Std')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
    % Min1
    [H,p,Z] = Kendall(Qy.G2Min1, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Min1, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(3,:) =  statTemp;
    
    subplot (6,2,3)
    hold on
    plot(Qy.Year, Qy.G2Min1, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Min1')
    xlim([xS xE])
    set(gca, 'xticklabels', '')

    % Max1
    [H,p,Z] = Kendall(Qy.G2Max1, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Max1, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(4,:) = statTemp;
    
    subplot  (6,2,4)
    hold on
    plot(Qy.Year, Qy.G2Max1, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Max1')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
   
       % Min3 
    [H,p,Z] = Kendall(Qy.G2Min3, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Min3, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(5,:) = statTemp;
    
    
    subplot  (6,2,5)
    hold on
    plot(Qy.Year, Qy.G2Min3, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Min3')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
   
        %Max3
    [H,p,Z] = Kendall(Qy.G2Max3, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Max3, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(6,:) = statTemp;
    
    
    subplot(6,2,6)
    hold on
    plot(Qy.Year, Qy.G2Max3, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Max3')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
        %Min7
    [H,p,Z] = Kendall(Qy.G2Min7, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Min7, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(7,:) = statTemp;
    
    
    subplot(6,2,7)
    hold on
    plot(Qy.Year, Qy.G2Min7, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Min7')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
   
        %Max7
    [H,p,Z] = Kendall(Qy.G2Max7, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Max7, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(8,:) = statTemp;
    
    
    subplot(6,2,8)
    hold on
    plot(Qy.Year, Qy.G2Max7, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Max7')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
        %Min31
    [H,p,Z] = Kendall(Qy.G2Min31, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Min31, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(9,:) = statTemp;
    
    
    subplot(6,2,9)
    hold on
    plot(Qy.Year, Qy.G2Min31, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Min31')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
   
        %Max31
    [H,p,Z] = Kendall(Qy.G2Max31, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Max31, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(10,:) = statTemp;
    
    
    subplot(6,2,10)
    hold on
    plot(Qy.Year, Qy.G2Max31, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Max31')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
        %Min91
    [H,p,Z] = Kendall(Qy.G2Min91, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Min91, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(11,:) = statTemp;
    
    
    subplot(6,2,11)
    hold on
    plot(Qy.Year, Qy.G2Min91, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Min91')
    xlim([xS xE])
    
       %Max91
    [H,p,Z] = Kendall(Qy.G2Max91, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G2Max91, [Qy.Year, ones(length(Qy.Year),1)]);
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG2(12,:) = statTemp;
    
    
    subplot(6,2,12)
    hold on
    plot(Qy.Year, Qy.G2Max91, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Max91')
    xlim([xS xE])
    
clear a b fig H Z p stats statTemp xE xS z

%%  
% <html><h3> GROUP 4: Frequency and duration of high and low pulses </h3></html>
%

%*Calculate the indicators*

p25 = prctile(d.Q, 25);
p75 = prctile(d.Q, 75);

for i = 1:length(Qy.Year)
    idx = find(dv(:,1) == Qy.Year(i));
    
    dQy = d.Q(idx);
    
   % High pulse (>percentile 75)
    % Essentially need to pick out values which exceed threshold with the condition that the previous value  
    % needs to be below the threshold
    idxl = dQy>p75;
    idxl(1) = 0;
    idx2 = find(idxl);
    yest = dQy(idx2-1)<p75; 
    loc = idx2(yest);  % location of the pulses
    Qy.G4nHigh(i) = length(loc); % number of high pulses
     
    
    % Low pulse (<percentile 25)
    % Essentially need to pick out values which exceed threshold with the condition that the previous value  
    % needs to be below the threshold
    idxl = dQy<p25;
    idxl(1) = 0;
    idx3 = find(idxl);
    yest = dQy(idx3-1)<p25; 
    loc = idx(yest);  % location of the pulses
    Qy.G4nLow(i) = length(loc); % number of high pulses
end


%*Trend analysis on the indicators*

Trend_yG4 = table('size', [2,8],'VariableTypes', {'double','double','double','double','double','double','double','double'});
Trend_yG4.Properties.VariableNames = {'p', 'Z', 'Int', 'Slope', 'R2', 'Fstat', 'Pval', 'Sigma'};
Trend_yG4.Properties.RowNames = {'nHigh', 'nLow'};

close all
fig = figure;
set(fig , 'units' , 'centimeters' , 'position' , [10 , 10 , 25 , 10])
fig.Color = 'white';

xS = Qy.Year(1);
xE = Qy.Year(end);

    % Number of high pulses (>p75)
    [H,p,Z] = Kendall(Qy.G4nHigh, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G4nHigh, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG4(1,:) =  statTemp;
    
    subplot 211
    hold on
    plot(Qy.Year, Qy.G4nHigh, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Number of high pulses (>p75)')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
     % Number of low pulses (<p25)
    [H,p,Z] = Kendall(Qy.G4nLow, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G4nLow, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG4(2,:) =  statTemp;
    
    subplot 212
    hold on
    plot(Qy.Year, Qy.G4nLow, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Number of low pulses (<p25)')
    xlim([xS xE])

clear idxl idx loc p25 p75 H p Z a b stats statTemp fig


%%  
% <html><h3>Rate and frequency of water  Frequency change </h3></html>
%

for i = 1:length(Qy.Year)
    idx = find(dv(:,1) == Qy.Year(i));
    
    dQy = d.Q(idx);
    changes = diff(dQy);
    idxRises = find(changes>0);
    idxFalls = find(changes<0);
    
% Means of all positive differences between consecutive days
    Qy.G5meanPosDiff(i) = mean(changes(idxRises)); 
% Means of all negative differences between consecutive days
    Qy.G5meanNegDiff(i) = mean(changes(idxFalls));
% No. of rise    
    Qy.G5nRises(i) = length(idxRises);
% No. of falls
    Qy.G5nFalls(i) = length(idxFalls);
end


Trend_yG5 = table('size', [4,8],'VariableTypes', {'double','double','double','double','double','double','double','double'});
Trend_yG5.Properties.VariableNames = {'p', 'Z', 'Int', 'Slope', 'R2', 'Fstat', 'Pval', 'Sigma'};
Trend_yG5.Properties.RowNames = {'meanPosDiff', 'meanNegDiff', 'nRises', 'nFalls'};

close all
fig = figure;
set(fig , 'units' , 'centimeters' , 'position' , [10 , 10 , 25 , 15])
fig.Color = 'white';

xS = Qy.Year(1);
xE = Qy.Year(end);

    % Means of all positive differences between consecutive days
    [H,p,Z] = Kendall(Qy.G5meanPosDiff, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G5meanPosDiff, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG5(1,:) =  statTemp;
    
    subplot 411
    hold on
    plot(Qy.Year, Qy.G5meanPosDiff, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Mean Positive Differences (m^3/s)')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
        % Number of rises
    [H,p,Z] = Kendall(Qy.G5nRises, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G5nRises, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG5(2,:) =  statTemp;
    
    subplot 412
    hold on
    plot(Qy.Year, Qy.G5nRises, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Number of rises')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
    % Means of all negative differences between consecutive days
    [H,p,Z] = Kendall(Qy.G5meanNegDiff, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G5meanNegDiff, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG5(3,:) =  statTemp;
    
    subplot 413
    hold on
    plot(Qy.Year, Qy.G5meanNegDiff, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Mean Negative Differences (m^3/s)')
    xlim([xS xE])
    set(gca, 'xticklabels', '')
    
    
    % Number of falls
    [H,p,Z] = Kendall(Qy.G5nFalls, 0.05);
    [b,~,~, ~, stats] = regress (Qy.G5nFalls, [Qy.Year, ones(length(Qy.Year),1)]);    
    statTemp = table(p, Z, b(2), b(1), stats(1), stats(2), stats(3), stats(4));
    Trend_yG5(4,:) =  statTemp;
    
    subplot 414
    hold on
    plot(Qy.Year, Qy.G5nFalls, 'k', 'marker', '.', 'markersize', 10)
    plot(Qy.Year, b(1)*Qy.Year + b(2)*ones(length(Qy.Year),1))
    title('Number of falls')
    xlim([xS xE])


 clear dQy changes  idxRises idxFalls i figH p Z statTemp b stats
 
 %%
% <html><h3>Spectral Analysis:Fourier Transform (FT)</h3></html>
%

% Remove mean
mu = nanmean(d.Q);
d.Q21 = d.Q-mu;

% Detrend (if any trend)
L = length(d.Q);
p = polyfit(1:L,d.Q21',1);
reg = polyval(p,1:L);
d.Q22 = d.Q21-reg';

% Use a window to taper the series and so reduce end effects (Gibbs phenomenon)
% Here, we use a Hanning window
n = [1:L];
w = 0.5*(1-cos(2*pi()*n./L)); % Hanning window
d.Qpad = d.Q22.*w';

% Plot raw and prepared data for inspection
figure
subplot 411
plot(d.datenum, d.Q)
title('raw')
subplot 412; hold on
plot(d.datenum, d.Q21)
plot(d.datenum, reg)
title('mean removed')
subplot 413
plot(d.datenum, d.Q22)
title('trend removed')
subplot 414
plot(d.datenum, d.Qpad)
title('padded with Hanning window')


%% Fast Fourrier Transform
% Note: best practice would be to perform a segmented averaged FFT with
% confidence interval

Fs = 365;          % Sampling frequency  (365 measurements/year)             
T = 1/Fs;          % Sampling period       
L = length(d.Q22);  % Length of signal
t = (0:L-1)*T;     % Time vector
f = Fs*(1:L)/L; % Frequency vector (cycles/year)


% Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
P2 = abs(fft(d.Qpad)/L); % Conduct FFT and take real part
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
cwt(d.Q22, years(1/(365)));
