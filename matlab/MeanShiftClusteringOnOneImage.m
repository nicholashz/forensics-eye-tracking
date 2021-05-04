function [FinalClusterAssignmentForEachRow, FinalClusterCentersForEachRow, FinalClusterSizeForEachRow, ...
    ColorsUsedInPlotting, FinalMasterRowAssignment]...
    = MeanShiftClusteringOnOneImage(AllAnalysisFixationsLeft, AllAnalysisFixationsRight, AllComparisonFixationsLeft,...
    AllComparisonFixationsRight, bigImage, latentVerticalShift, ExemplarOffset, ExemplarVerticalShift, TotalRows, bandwidth, NumPointsInClusterToBeABadCluster, RGBValuesForClusters, ActuallyDisplayImages)
%offset fixation data for plotting first thing:

% AllAnalysisFixations(:,1:2) = [AllAnalysisFixations(:,1) AllAnalysisFixations(:,2) + latentVerticalShift];
% AllComparisonFixationsLeft(:,1:2) = [AllComparisonFixationsLeft(:,1)  AllComparisonFixationsLeft(:,2) + latentVerticalShift];
% AllComparisonFixationsRight(:,1:2) = [AllComparisonFixationsRight(:,1) + ExemplarOffset AllComparisonFixationsRight(:,2) + ExemplarVerticalShift];

ClusterAssignmentForEachRow = ones(TotalRows, 1) * (-1);%-1 is unassigned (probably because on exemplar during comparison)
ClusterCentersForEachRow = zeros(TotalRows, 2);
ClusterSizeForEachRow = zeros(TotalRows, 1);

spacing=5;
sizeOfCircle = 20;
theAlpha = .25;
%doCombined = 0;
RedoMatFiles = 0; %not used?


whichExperimentalStage = 2; %0 = analysis, 1 = comparison, 2 = both

%parameter for mean shift clustering:
%bandwidth = 30;
%bandwidth = 40;
%bandwidth = 20;

%NumPointsInClusterToBeABadCluster = 1;
%NumPointsInClusterToBeABadCluster = 20; %across all subjects; only for plotting purposes
%NumPointsInClusterToBeABadCluster = 0

%addpath('MeanShift/');

useLog = 1;

%bandwidth = 60
% while doCombined <0 || doCombined > 1
%    doCombined = input('Do you want to do individual subjects or all subects combined (0=individual, 1 =Combined)?');
%end

if ActuallyDisplayImages
    %now plot the data over the image
    figure(1)
    imshow(bigImage)
    drawnow
    hold on
    if size(AllAnalysisFixationsLeft,1)>0
        plot(AllAnalysisFixationsLeft(:,1), AllAnalysisFixationsLeft(:,2)+latentVerticalShift, 'r.', 'MarkerSize', 10)
        plot(AllAnalysisFixationsRight(:,1) + ExemplarOffset, AllAnalysisFixationsRight(:,2)+ExemplarVerticalShift, 'b.', 'MarkerSize', 10)
    end
    plot(AllComparisonFixationsLeft(:,1), AllComparisonFixationsLeft(:,2) + latentVerticalShift, 'g.', 'MarkerSize', 10)
    plot(AllComparisonFixationsRight(:,1) + ExemplarOffset, AllComparisonFixationsRight(:,2) + ExemplarVerticalShift, 'c.', 'MarkerSize', 10)
    
    hold off
    drawnow
end
%now do clustering



cVec = 'bgrcmybgrcmybgrcmybgrcmy';%,


a=[1 1 0;
    0 0 1;
    0 1 0;
    0 1 1;
    1 0 0;
    1 0 1;
    1 1 1;
    240/255 120/255 0;
    64/255 0 128/255;
    128/255 64/255 0;
    0 64/255 0;
    128/255 128/255 128/255;
    128/255 128/255 1;
    0 128/255 128/255;
    128/255 0 0;
    1 128/255 128/255];


%store combined left and right results
FinalClusterAssignmentForEachRow = [];
FinalClusterCentersForEachRow = [];
FinalClusterSizeForEachRow = [];
FinalMasterRowAssignment = [];

for thisAnalysis = 2:3
    if thisAnalysis == 1
        x = AllAnalysisFixationsLeft(:,1:2)';%apparently have to transpose this
        RowsForFixationsInHugeArray = AllAnalysisFixationsLeft(:,3);
        
        PointsXOffset = 0;
        PointsYOffset = latentVerticalShift;
        %add in the offsets here
        if ActuallyDisplayImages
            figure(10),clf,
            imshow(bigImage)
            drawnow
            title(sprintf('Analysis Fixations Bandwidth = %d, Prune = %d', bandwidth, NumPointsInClusterToBeABadCluster));
            
            hold on
        end
    end
    if thisAnalysis == 2
        if ActuallyDisplayImages
            hold off
        end
        x = AllComparisonFixationsLeft(:,1:2)';%apparently have to transpose this
        RowsForFixationsInHugeArray = AllComparisonFixationsLeft(:,4);
        MasterRowNumbersInHugeArray = AllComparisonFixationsLeft(:,3);
        
        PointsXOffset = 0;
        PointsYOffset = latentVerticalShift;
        if ActuallyDisplayImages
            figure(11),clf,
            imshow(bigImage)
            drawnow
            title(sprintf('Comparison Fixations Bandwidth = %d, Prune = %d', bandwidth, NumPointsInClusterToBeABadCluster));
            hold on
        end
    end
    if thisAnalysis == 3
        x = AllComparisonFixationsRight(:,1:2)';%apparently have to transpose this
        RowsForFixationsInHugeArray = AllComparisonFixationsRight(:,4);
        MasterRowNumbersInHugeArray = AllComparisonFixationsRight(:,3);
        
        PointsXOffset = ExemplarOffset;
        PointsYOffset = ExemplarVerticalShift;
        
        %figure(11)
        %hold on
    end
    
    
    
    tic
    [clustCent,point2cluster,clustMembsCell] = MeanShiftCluster(x,bandwidth, false);
    toc
    
    numClust = length(clustMembsCell);
    
    
    [goodClusters, badClusters] = plotPointsOnFigures(x, clustMembsCell, clustCent, ...
        NumPointsInClusterToBeABadCluster, ActuallyDisplayImages,...
        PointsXOffset, PointsYOffset, cVec, numClust, a);
    
    %now go through all the good clusters and assign the cluster number to
    %the corresponding row in
    ClusterAssignmentForEachRow = [];
    ClusterCentersForEachRow = [];
    ClusterSizeForEachRow = [];
    MasterRowAssignment = [];
    
    for ClusterNumber = 1:numClust
        myClustCen = clustCent(:,ClusterNumber);
        numPointsInThisCluster = size(clustMembsCell{ClusterNumber,1}, 2);
        
        RowsInXWithThisCluster = find(point2cluster==ClusterNumber);
        MasterRowsWithThisCluster = MasterRowNumbersInHugeArray(RowsInXWithThisCluster);
        
        %now we need to convert these points to the big list according to
        %the mapping in RowsForFixationsInHugeArray
        RowsToChangeInHugeArray = RowsForFixationsInHugeArray(RowsInXWithThisCluster);
        ClusterAssignmentForEachRow(RowsToChangeInHugeArray) = ClusterNumber;
        ClusterCentersForEachRow(RowsToChangeInHugeArray, 1) = myClustCen(1,1);
        ClusterCentersForEachRow(RowsToChangeInHugeArray, 2) = myClustCen(2,1);
        ClusterSizeForEachRow(RowsToChangeInHugeArray, 1) = numPointsInThisCluster;
        MasterRowAssignment(RowsToChangeInHugeArray, 1) = MasterRowsWithThisCluster; %#ok<AGROW>
        
    end
    
    
    
    %axis([-4 4 -4 4]);
    
    numPointsPerCluster = zeros(1,size(clustMembsCell,1));
    for thisCluster = 1:size(clustMembsCell,1)
        numPointsPerCluster(thisCluster) = size(clustMembsCell{thisCluster},2);
    end
    sort(numPointsPerCluster, 2, 'descend');
    %title(['numClust:' int2str(numClust)]);
    
    %now prune bad clusters
    %them'
    %eventually reassign them
    if 0
        for k = badClusters
            badIndicies = find(point2cluster==k);
            %now delete these entries
            allIndecies = 1:size(point2cluster,2);
            goodIndicies = setdiff(allIndecies, badIndicies);
            x = x(:, goodIndicies);
            point2cluster = point2cluster(:,goodIndicies);
            %        AllSubNumbers = AllSubNumbers(1, goodIndicies);
            %        AllWhichExperimentalStage = AllWhichExperimentalStage(1, goodIndicies);
        end
    end
    
    
    %now write out:
    %x, y, xcenter, ycenter, clusternumber
    if 0
        if thisSub == 0
            
            saveFile = sprintf('ClusteringDataOutput/Cluster_AllSubs_Trial%dBandwidth%dPart%d.txt', trial, bandwidth, whichExperimentalStage);
            clusterSaveFile = sprintf('ClusteringDataOutput/ClusterCenters_AllSubs_Trial%dBandwidth%dPart%d.txt', trial, bandwidth, whichExperimentalStage);
            figureSaveName = sprintf('ClusteringDataOutput/Cluster_AllSubs_Trial%dBandwidth%dPart%d.jpg', trial, bandwidth, whichExperimentalStage);
            
        else
            saveFile = sprintf('ClusteringDataOutput/Cluster_%s_Trial%dBandwidth%dPart%d.txt', char(SubjectList{thisSub}), trial, bandwidth, whichExperimentalStage);
            clusterSaveFile = sprintf('ClusteringDataOutput/ClusterCenters_%s_Trial%dBandwidth%dPart%d.txt', char(SubjectList{thisSub}), trial,bandwidth, whichExperimentalStage);
            figureSaveName = sprintf('ClusteringDataOutput/Cluster_%s_Trial%dBandwidth%dPart%d.jpg', char(SubjectList{thisSub}), trial, bandwidth, whichExperimentalStage);
            
        end
        
        
        print(gcf, '-dpng', strcat(figureSaveName));
        
        fid = fopen (saveFile, 'w');
        fprintf(fid, 'subnum\tx\ty\tclustername\tExptPart\n');
        for thisgaze = 1:size(x,2)
            fprintf(fid, '%d\t%6.3f\t%6.3f\t%d\t%d\n', AllSubNumbers(1,thisgaze), x(1,thisgaze), x(2, thisgaze), point2cluster(thisgaze), AllWhichExperimentalStage(1,thisgaze)); %x is actually AllSubsLeft...
        end
        fclose(fid);
        
        fid = fopen (clusterSaveFile, 'w');
        fprintf(fid, 'subnum\tx\ty\tclustername\n');
        for thisCluster = goodClusters
            myClustCen = clustCent(:,thisCluster);
            
            fprintf(fid, '%d\t%6.3f\t%6.3f\t%d\n',  AllSubNumbers(1,thisgaze), myClustCen(1,1), myClustCen(2,1), thisCluster);
        end
        fclose(fid);
    end
    
    %now combine left and right into one big array
    FinalClusterAssignmentForEachRow{thisAnalysis-1} = ClusterAssignmentForEachRow;
    
    FinalClusterCentersForEachRow{thisAnalysis-1} =  ClusterCentersForEachRow;
    FinalClusterSizeForEachRow{thisAnalysis-1} =  ClusterSizeForEachRow;
    FinalMasterRowAssignment{thisAnalysis-1} = MasterRowAssignment;
end
ColorsUsedInPlotting = a;

%now flatten these

%ClusterAssignmentForEachRow = [ClusterAssignmentForEachRow(2,:) ClusterAssignmentForEachRow(3,:)];
%       ClusterCentersForEachRow(thisAnalysis, RowsToChangeInHugeArray, 1) = myClustCen(1,1);
%      ClusterCentersForEachRow(thisAnalysis, RowsToChangeInHugeArray, 2) = myClustCen(2,1);
%     ClusterSizeForEachRow(thisAnalysis, RowsToChangeInHugeArray, 1) = numPointsInThisCluster;
%    MasterRowAssignment(thisAnalysis, RowsToChangeInHugeArray, 1) = MasterRowsWithThisCluster;