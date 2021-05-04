function RGBValuesForClusters = GetRGBValues(name, ColorsToUse)
RGBValuesForClusters = zeros(3,size(ColorsToUse,1));
for thisColorIndex = 1:size(ColorsToUse,2)
    thisColor = ColorsToUse(thisColorIndex);
    RGBValuesForClusters(:,thisColorIndex) = rgb(char(name{thisColor}));
    
end
RGBValuesForClusters = RGBValuesForClusters';