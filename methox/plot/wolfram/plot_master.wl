data = Import["_FILENAME_", "Table"];
zz=data[[All,3]]-Min[data[[All,3]]];
xy=data[[All,{2, 1}]];
xyz = MapThread[Append, {xy, zz}];
label = "iteration _ITER_";
img = ListContourPlot[xyz ,
PlotLabel -> label,
Frame -> True,
FrameLabel -> {{"P-Oattack (Å)"},{"P-Olg (Å)"}},
Contours-> Table[i,{i,0,50,2}],
ContourLabels -> None,
ColorFunction -> "BrightBands",
PlotLegends -> BarLegend[Automatic, LegendMarkerSize -> 300],
PlotRange -> {{1.5,4.0}, {1.5,4.0}, {0,50}},
PlotRangeClipping -> True];
Export["_IMGNAME_", img, "PNG", ImageResolution->200];
