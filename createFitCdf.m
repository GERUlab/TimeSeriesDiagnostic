function [pd1,pd2,pd3] = createFitCdf(dataD)
%CREATEFIT    Create plot of datasets and fits
%   [PD1,PD2,PD3] = CREATEFIT(DATAD)
%   Creates a plot, similar to the plot in the main distribution fitting
%   window, using the data that you provide as input.  You can
%   apply this function to the same data you used with dfittool
%   or with different data.  You may want to edit the function to
%   customize the code and this help message.
%
%   Number of datasets:  1
%   Number of fits:  3
%
%   See also FITDIST.

% This function was automatically generated on 28-Feb-2014 16:19:35

% Output fitted probablility distributions: PD1,PD2,PD3

% Data from dataset "Data":
%    Y = dataD

% Force all inputs to be column vectors
dataD = dataD(:);

% Prepare figure
% clf;
hold on;
LegHandles = []; LegText = {};


% --- Plot data originally in dataset "Data"
[CdfY,CdfX] = ecdf(dataD,'Function','cdf');  % compute empirical function
hLine = stairs(CdfX,CdfY,'k.','markersize',7)%'Color','k','LineStyle','none', 'LineWidth',2);
xlabel('Data');
ylabel('Cumulative probability')
LegHandles(end+1) = hLine;
LegText{end+1} = 'Data';

% Create grid where function will be computed
XLim = get(gca,'XLim');
XLim = XLim + [-1 1] * 0.01 * diff(XLim);
XGrid = linspace(XLim(1),XLim(2),100);


% --- Create fit "Normal"

% Fit this distribution to get parameter values
% To use parameter estimates from the original fit:
%     pd1 = ProbDistUnivParam('normal',[ 25.92968971478, 53.29803139816])
pd1 = fitdist(dataD, 'normal');
YPlot = cdf(pd1,XGrid);
hLine = plot(XGrid,YPlot,'Color','r',...
    'LineStyle','-', 'LineWidth',1,...
    'Marker','none', 'MarkerSize',6);
LegHandles(end+1) = hLine;
LegText{end+1} = 'Normal';

% --- Create fit "Lognormal"

% Fit this distribution to get parameter values
% To use parameter estimates from the original fit:
%     pd2 = ProbDistUnivParam('lognormal',[ 2.815131598463, 0.796145646599])
pd2 = fitdist(dataD, 'lognormal');
YPlot = cdf(pd2,XGrid);
hLine = plot(XGrid,YPlot,'Color','b',...
    'LineStyle','-', 'LineWidth',1,...
    'Marker','none', 'MarkerSize',6);
LegHandles(end+1) = hLine;
LegText{end+1} = 'Log';

% --- Create fit "Exponential"

% Fit this distribution to get parameter values
% To use parameter estimates from the original fit:
%     pd3 = ProbDistUnivParam('exponential',[ 25.92968971478])
pd3 = fitdist(dataD, 'exponential');
YPlot = cdf(pd3,XGrid);
hLine = plot(XGrid,YPlot,'Color','g',...
    'LineStyle','-', 'LineWidth',1,...
    'Marker','none', 'MarkerSize',6);
LegHandles(end+1) = hLine;
LegText{end+1} = 'Exp';

% Adjust figure
box on;
hold off;

% Create legend from accumulated handles and labels
hLegend = legend(LegHandles,LegText,'Orientation', 'vertical', 'Location', 'NorthEast');
set(hLegend,'Interpreter','none');
xlim([0 max(dataD)])
