% --- Experiment 2/3: Hop Distribution Bar Chart ---
% Shared KUL color palette
C = kulcolors();

% Figure size (pixels): tweak these for your preferred output dimensions.
figWidth = 800;
figHeight = 300;
figure('Position', [100, 100, figWidth, figHeight]);

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 20);   
set(groot, 'defaultLegendFontSize', 18); 
set(groot, 'defaultTextFontSize', 18);  

% X-axis categories
categories = {'1 Hop', '2 Hops', '3 Hops', '\geq 4 Hops'};

% Dummy Data: [UDM Counts; Synthology Counts]
% Rows = Hop Categories (1, 2, 3, 4+), Cols = Methods (UDM, Synthology)
% Replace with your actual counts. This reflects that UDM dominates 1-hop 
% while Synthology has a broader depth profile.
data = [
    8500,  1200;  % 1 Hop
     900,  3500;  % 2 Hops
      50,  2800;  % 3 Hops
       5,  1500   % 4+ Hops
];

b = bar(data, 'grouped');

% Styling for publication
b(1).FaceColor = C.KULijsblauw;  % UDM
b(2).FaceColor = C.KULcorporate; % Synthology

% Axes and labels
set(gca, 'XTickLabel', categories, 'FontSize', 11);
ylabel('Number of Inferred Facts', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Logical Proof Depth (Hops)', 'FontSize', 12, 'FontWeight', 'bold');

% Legend
legend('UDM Baseline', 'Synthology', 'Location', 'northeast', ...
    'FontSize', 11, 'Interpreter', 'latex');

% Clean up aesthetics
box off;
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6);

% Optional: Use a log scale on the Y-axis if the 1-hop disparity is massive
% set(gca, 'YScale', 'log');

% Save the figure to pdf
exportgraphics(gcf, 'out/exp2_hop_distr.pdf', 'ContentType', 'vector');
