
function [Correspondences, GreedyMaxes, GreedyMaxValues, ...
    ClusterOffsetDeviation,...
    GoodLatentCoords, CorrespondingExemplarCoords] = fitAbsurdist(ListOfLatents, ListOfExemplars, externalSim, ...
    imagePairName, YString,...
    bigImage, latentVerticalShift, ExemplarOffset, ExemplarVerticalShift, ...
    plottingLineWidthThreshold, ListOfTranslatedLatents, numberOfAbsurdistIternations,...
    actuallyFitAbsurdist, assignUsingMatchedPairs, ActuallyDisplayImages,...
    normalizeTransitionMatrix, ZeroWeakCorrespondences)
%compute normalized distances between points for external
%similarity
% if 0
%     for thisLatentFixation = 1:size(ListOfLatents,2)
%         for thisExemplarFixation = 1:size(ListOfExemplars,2)
%             latentPoint = ListOfLatents(:, thisLatentFixation);
%             ExemplarPoint = ListOfExemplars(:, thisExemplarFixation);
%             oneExternalSim = sqrt(((latentPoint(1, 1)-twoCenters(1,1))-(ExemplarPoint(1,1)-twoCenters(1,2)))^2 + ...
%                 ((latentPoint(2, 1)-twoCenters(2,1))-(ExemplarPoint(2,1)-twoCenters(2,2)))^2);
%             externalSim(thisLatentFixation, thisExemplarFixation) =oneExternalSim;
%         end
%     end
%     externalSim =  (externalSim- min(externalSim(:)))/( max(externalSim(:)) -  min(externalSim(:)));
%     %externalSim = exp(-externalSim);
% end
if actuallyFitAbsurdist
    Correspondences = findCorrespondences(ListOfLatents, ListOfExemplars, externalSim, numberOfAbsurdistIternations);
    
else
    Correspondences = externalSim;
    if ~isempty(Correspondences)
        if ( max(externalSim(:)) -  min(externalSim(:))) == 0 && ~normalizeTransitionMatrix
            fprintf('Correspondences were all equal\n');
        else
            if ~normalizeTransitionMatrix
                Correspondences = (Correspondences - min(Correspondences(:)))/(max(Correspondences(:))-min(Correspondences(:)));
            else%normalize Correspondence matrix by normalzing row, normalizing column, and taking average
                
                if ZeroWeakCorrespondences
                    Correspondences(Correspondences<2)=0;
                end
                rowNormed=Correspondences./sum(Correspondences,2);
                colNormed=(Correspondences'./sum(Correspondences,1)')';
                
                %set nans to 0 for now; shouldn't have?
                rowNormed(isnan(rowNormed())) = 0;
                colNormed(isnan(colNormed())) = 0;
                
                Correspondences = (rowNormed + colNormed)/2;
                
            end
        end
    end
    
end
if isempty(Correspondences)
    GreedyMaxes = [];
    GreedyMaxValues = [];
    ClusterOffsetDeviation= [];
    GoodLatentCoords = [];
    CorrespondingExemplarCoords = [];
    return
end
%Correspondences = abs(Correspondences);
%for every fixation on the latent, return the index and the
%value into the examplar list
if assignUsingMatchedPairs
    %try version from matchpairs
    GreedyMaxes = nan(1, size(Correspondences,1));
    GreedyMaxValues = nan(1, size(Correspondences,1));
    
    if sum(isnan(Correspondences(:)))== 0
        [M,uR,uC] = matchpairs(Correspondences,0, 'max');
        %assign a corresponding exemplar to every
        %latent cluster
        %M is an assignment of every latent (first
        %column to an exemplar cluster
        
        for assignedLatentIndex = 1:size(M,1)
            GreedyMaxes(1, M(assignedLatentIndex,1)) = M(assignedLatentIndex,2);
            GreedyMaxValues(1, M(assignedLatentIndex,1)) = Correspondences(M(assignedLatentIndex,1), M(assignedLatentIndex,2));
            %[GreedyMaxes, GreedyMaxValues]
            
        end
        [oldGreedyMaxes, oldGreedyMaxValues] = getGreedyMaxes(Correspondences);
        
    else
        
    end
else
    [GreedyMaxes, GreedyMaxValues, externalSimValues] = getGreedyMaxes(Correspondences, externalSim);
end

%[GreedyMaxes, GreedyMaxValues] = getGreedyMaxes(Correspondences);


allNonNaNValues = GreedyMaxValues(find(GreedyMaxValues==GreedyMaxValues));
denom = max(allNonNaNValues)-min(allNonNaNValues);

%SaveName = sprintf('%s_%s_%04d_First', imagePairName, YString, thisFixationIndex);

%draw groundtruth on


if ActuallyDisplayImages
    drawnow
end
%PrintFigure(gcf, printResolution, folderForImagesFromAbsurdist, sprintf('%s.png', SaveName));


%now plot the max correspondence for each left one to each
%right one
ClusterOffsetDeviation = nan(1,size(ListOfLatents,2));
GoodLatentCoords = nan(2,size(ListOfLatents,2));%coordinates of left cluster
CorrespondingExemplarCoords = nan(2,size(ListOfLatents,2));%coordinates of corresponding right cluster

for thisLatentFixation = 1:size(ListOfLatents,2)
    if ~isnan(GreedyMaxes(thisLatentFixation)) && ~isnan(GreedyMaxes(thisLatentFixation))
        LatentCoord = ListOfLatents(:, thisLatentFixation);
        %IndexOfMax = find(Correspondences(thisLatentFixation, :) == max(Correspondences(thisLatentFixation, :)));
        ExemplarCoord = ListOfExemplars(:, GreedyMaxes(thisLatentFixation));
        
        GoodLatentCoords(:,thisLatentFixation) = LatentCoord;
        CorrespondingExemplarCoords(:,thisLatentFixation) = ExemplarCoord; %pass these back so we can do things like compute angles as summary score
        
        if denom > 0
            lineWidth = (GreedyMaxValues(thisLatentFixation)-min(allNonNaNValues(:)))/denom * 4 + 1;
        else
            lineWidth = 1;
        end
        if ActuallyDisplayImages
            hold on
            
            if (GreedyMaxValues(thisLatentFixation) > plottingLineWidthThreshold)
            %if (GreedyMaxValues(thisLatentFixation)-min(allNonNaNValues(:)))/denom > plottingLineWidthThreshold%
                plot([LatentCoord(1) ExemplarCoord(1)+ExemplarOffset], [LatentCoord(2)+latentVerticalShift ExemplarCoord(2)+ExemplarVerticalShift], 'w-', 'LineWidth', lineWidth, 'Color', [1 1 .999]) %add offsets
                text(((ExemplarCoord(1)+ExemplarOffset)+LatentCoord(1))/2, ...
                    ((ExemplarCoord(2)+ExemplarVerticalShift)+(LatentCoord(2)+latentVerticalShift))/2,...
                    sprintf('%3.2f (%d)', GreedyMaxValues(thisLatentFixation),externalSimValues(thisLatentFixation)), 'Color', 'red', 'FontSize', 24);
                if GreedyMaxValues(thisLatentFixation)>1.0
                    fred = 1;
                end
            else
                plot([LatentCoord(1) ExemplarCoord(1)+ExemplarOffset], [LatentCoord(2)+latentVerticalShift ExemplarCoord(2)+ExemplarVerticalShift], 'w-', 'LineWidth', lineWidth, 'Color', [.5 .5 .5])%plot weak lines in gray
%                 if (GreedyMaxValues(thisLatentFixation)-min(allNonNaNValues(:)))/denom > .5
%                     text(((ExemplarCoord(1)+ExemplarOffset)+LatentCoord(1))/2, ...
%                         ((ExemplarCoord(2)+ExemplarVerticalShift)+(LatentCoord(2)+latentVerticalShift))/2,...
%                         sprintf('%3.2f (%d)', (GreedyMaxValues(thisLatentFixation)-min(allNonNaNValues(:)))/denom,externalSimValues(thisLatentFixation)), 'Color', 'blue', 'FontSize', 12);
%                 end
            end
        end
        %now plot ground truth if available
        if ~isempty(ListOfTranslatedLatents) && sum(~isnan(ListOfTranslatedLatents(:)))>0
            TranslatedLatentCoord = ListOfTranslatedLatents(:,thisLatentFixation);
            if ActuallyDisplayImages
                
                if (GreedyMaxValues(thisLatentFixation)-min(allNonNaNValues(:)))/denom > plottingLineWidthThreshold
                    
                    plot(TranslatedLatentCoord(1)+ExemplarOffset, TranslatedLatentCoord(2)+ExemplarVerticalShift, 'o', 'Color', [1 1 .999]) %add offsets
                    plot([ ExemplarCoord(1)+ExemplarOffset TranslatedLatentCoord(1)+ExemplarOffset], [ExemplarCoord(2)+ExemplarVerticalShift TranslatedLatentCoord(2)+ExemplarVerticalShift], 'w-', 'LineWidth', lineWidth, 'Color', [0, 0, 1])%plot weak lines in gray
                else
                    plot(TranslatedLatentCoord(1)+ExemplarOffset, TranslatedLatentCoord(2)+ExemplarVerticalShift, 'o', 'Color', [.5 .5 .5]) %add offsets
                    plot([ ExemplarCoord(1)+ExemplarOffset TranslatedLatentCoord(1)+ExemplarOffset], [ExemplarCoord(2)+ExemplarVerticalShift TranslatedLatentCoord(2)+ExemplarVerticalShift], 'w-', 'LineWidth', lineWidth, 'Color', [0, 0, .5])%plot weak lines in gray
                    
                end
            end
            %now get error distance
            ClusterOffsetDeviation(thisLatentFixation) = sqrt(((ExemplarCoord(1)+ExemplarOffset)-(TranslatedLatentCoord(1)+ExemplarOffset))^2+...
                ((ExemplarCoord(2)+ExemplarVerticalShift)-(TranslatedLatentCoord(2)+ExemplarVerticalShift))^2);
            
        end
    end
    
end

%if DFT images, plot ground truth on exemplar
if ActuallyDisplayImages
    if  strfind(imagePairName, 'DFT')
        DFTNumber = str2num(imagePairName(4));
        DFTTrialType = imagePairName(5);
        rightCoords = DFTCoordinates{DFTNumber}.ExemplarCoords;
        plot([rightCoords(1)+ExemplarOffset rightCoords(3)+ExemplarOffset], ...
            [rightCoords(2)+ExemplarVerticalShift rightCoords(2)+ExemplarVerticalShift], 'r-');
        plot([rightCoords(1)+ExemplarOffset rightCoords(3)+ExemplarOffset], ...
            [rightCoords(4)+ExemplarVerticalShift rightCoords(4)+ExemplarVerticalShift], 'r-');
        plot([rightCoords(1)+ExemplarOffset rightCoords(1)+ExemplarOffset], ...
            [rightCoords(2)+ExemplarVerticalShift rightCoords(4)+ExemplarVerticalShift], 'r-');
        plot([rightCoords(3)+ExemplarOffset rightCoords(3)+ExemplarOffset], ...
            [rightCoords(2)+ExemplarVerticalShift rightCoords(4)+ExemplarVerticalShift], 'r-');
        
    end
    
end
if 0
    [xloc, yloc] = ginput(1);
    
    PredPoints = ApplyCalibration([(xloc-1) (yloc-1)]', forwardDim1coeff, forwardDim2coeff, NumCoeffsMainFit);
    plot(xloc, yloc, 'r*');
    plot(PredPoints(1), PredPoints(2), 'g*');
    
    pause
end
if ActuallyDisplayImages
    hold off
    
    %SaveName = sprintf('%s_%s_%04d_Second', imagePairName, YString, thisFixationIndex);
    drawnow
end
%PrintFigure(gcf, printResolution, folderForImagesFromAbsurdist, sprintf('%s.png', SaveName));
%pause
