function c = kulcolors()
%KULCOLORS Returns KU Leuven colors for plots.
% Field names follow KULkleuren.sty (e.g., KULijsblauw, KULcorporate).
% Colors are normalized RGB triples in a struct.

% Names aligned with KULkleuren.sty
c.KULijsblauw  = [220, 231, 240] / 255;
c.KULwit       = [255, 255, 255] / 255;
c.KULcorporate = [0, 64, 122] / 255;
c.KULpetrol    = [20, 127, 161] / 255;

c.KULzwart     = [0, 29, 65] / 255;
c.KULrood      = [240, 119, 110] / 255;
c.KULoranje    = [251, 176, 55] / 255;
c.KULgeel      = [228, 218, 62] / 255;
c.KULgroen     = [135, 192, 189] / 255;
c.KULblauw     = [82, 189, 236] / 255;
c.KULpaars     = [199, 147, 174] / 255;

% Backward-compatible aliases for earlier script versions.
c.blue      = c.KULcorporate;
c.lightBlue = c.KULblauw;
c.darkBlue  = c.KULzwart;
c.cyan      = c.KULpetrol;
c.green     = c.KULgroen;
c.orange    = c.KULoranje;
c.red       = c.KULrood;
c.gray      = [130, 130, 130] / 255;
c.lightGray = c.KULijsblauw;
c.black     = c.KULzwart;
c.white     = c.KULwit;

% Convenient default order for color cycling.
c.palette = [
    c.KULcorporate;
    c.KULblauw;
    c.KULpetrol;
    c.KULgroen;
    c.KULoranje;
    c.KULrood;
    c.KULpaars;
    c.KULzwart
];
end
