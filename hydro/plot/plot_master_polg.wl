data = Import["_FILENAME_", "Table"];
zz=data[[All,3]]-Min[data[[All,3]]];
xy=data[[All,{2, 1}]];
xyz = MapThread[Append, {xy, zz}];
label = "iteration _ITER_ P-Olg = " <> ToString@data [[1, 4]];
img = ListContourPlot[xyz ,
PlotLabel -> label,
Frame -> True,
FrameLabel -> {{"P-Oattack (Å)"},{"Asymmetric Stretch (Å)"}},
Contours-> Table[i,{i,0,50,2}],
ContourLabels -> None,
ColorFunction -> "BrightBands",
PlotLegends -> BarLegend[Automatic, LegendMarkerSize -> 300],
PlotRange -> {{-2.0,2.0}, {1.5,4.0}, {0,50}},
PlotRangeClipping -> True];
Export["_IMGNAME_", img, "PNG", ImageResolution->200];
