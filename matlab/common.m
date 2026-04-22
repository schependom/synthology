function style = common(FS)
% COMMON Extracts and applies shared styling and configurations for charts.
%
% Usage:
%   style = common(24); 
%   (Where 24 is your desired FontSize)

    if nargin < 1
        FS = 24; % Default to your standard font size
    end

    % Global LaTeX defaults
    set(groot, 'defaultTextInterpreter',          'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter',        'latex');

    % Apply font size to global defaults
    set(groot, 'defaultAxesFontSize',   FS);
    set(groot, 'defaultLegendFontSize', FS - 2);
    set(groot, 'defaultTextFontSize',   FS - 2);

    % Return style struct containing font size and KU Leuven colors
    style.FS = FS;
    style.C = kulcolors();
end