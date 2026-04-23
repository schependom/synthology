function render_barchart(floatData, xTicks, xLabelText, yLabelText, titleText, outFile)
% RENDER_FLOAT_BARCHART Generates a styled bar chart for a list of floats.
%
% Parameters:
%   floatData  - Array of numeric floats to plot.
%   xTicks     - Cell array of strings/chars corresponding to each bar.
%   xLabelText - Label for the X-axis.
%   yLabelText - Label for the Y-axis.
%   titleText  - Title of the chart.
%   outFile    - Complete path where the PDF will be saved.

    % 1. Apply common styling config
    style = common(24); 
    FS = style.FS;
    C = style.C;

    % 2. Initialize figure (using same dimensions/background as exp23)
    fig = figure('Position', [100, 100, 1000, 500], 'Color', 'w');

    % 3. Plot the bar chart
    % Using KULcorporate color and 0.88 transparency to match your style
    b = bar(1:length(floatData), floatData);
    b.FaceColor = C.KULcorporate;
    b.FaceAlpha = 0.88;

    % ---- ADD VALUES ----
    % Get the coordinates for the top center of each bar
    xLocs = b.XEndPoints;
    yLocs = b.YEndPoints;

    % Loop through each bar and add the text
    for i = 1:length(floatData)
        % Format the text (e.g., '12.3 s'). Change %.1f to %.2f for two decimals
        labelStr = sprintf('%.1f s', floatData(i)); 
        
        % Place the text slightly above the bar
        text(xLocs(i), yLocs(i), labelStr, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', FS - 2, ...         % Slightly smaller than axis font
            'Color', 'k');                  % Black text
    end
    % -----------------------------------

    % 4. Format axes and text
    set(gca, 'XTick', 1:length(floatData), 'XTickLabel', xTicks, 'FontSize', FS);

    xlabel(xLabelText, 'FontSize', FS, 'FontWeight', 'bold');
    ylabel(yLabelText, 'FontSize', FS, 'FontWeight', 'bold');
    title(titleText,   'FontSize', FS + 2, 'FontWeight', 'bold');

    % 5. Apply shared grid styling
    box off;
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);

    % 6. Export and save the result
    if isfile(outFile)
        delete(outFile);
    end

    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved float bar chart: %s\n', outFile);
    close(fig);
end