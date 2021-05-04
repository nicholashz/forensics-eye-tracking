







function [Correspondences, GreedyMaxes, GreedyMaxValues, ListOfLatentIdentities, ...
    ListOfExemplarIdentities,...
    ListOfLatents,...
    ListOfExemplars,  ListOfTranslatedLatents, ...
    ClusterOffsetDeviation,...
    GoodLatentCoords, CorrespondingExemplarCoords,...
    externalSims] = ...
    findAbsurdistCorrespondences(LeftSideData, RightSideData,...
    bigImage, latentVerticalShift, ExemplarOffset, ExemplarVerticalShift,...
    thisImageName, examinerName, Transitions, plottingLineWidthThreshold,...
    thresholdForClusterInclusion, useOnlyUniqueFixations,...
    TranslatedClusterCenters, numberOfAbsurdistIternations,...
    actuallyFitAbsurdist, assignUsingMatchedPairs, ...
    ActuallyDisplayImages, randomizeHintClusters,...
    normalizeTransitionMatrix,...
    ZeroWeakCorrespondences)


%                        ClusteringData{imageSide, bandwidthIndex}.ClusterAssignmentForEachRow = ClusterAssignmentForEachRow{imageSide}';
%                       ClusteringData{imageSide, bandwidthIndex}.ClusterCentersForEachRow = ClusterCentersForEachRow{imageSide};
%                      ClusteringData{imageSide, bandwidthIndex}.ClusterSizeForEachRow = ClusterSizeForEachRow{imageSide};
%                     ClusteringData{imageSide, bandwidthIndex}.MasterRowAssignment = MasterRowAssignment{imageSide};


UniqueLeftSideClusters = unique(LeftSideData.ClusterAssignmentForEachRow);
ListOfLatents = [];
ListOfLatentIdentities = [];
ListOfTranslatedLatents = [];
for thisLeftClusterIndex = 1:size(UniqueLeftSideClusters,1)
    %find an instance of this cluster so we can get the cluster center
    %coords
    %could prune clusters here and not include it in the list if cluster
    %membership is less than like 3 fixations
    ClusterIdentity =  UniqueLeftSideClusters(thisLeftClusterIndex);
    InstancesOfThisCluster = find(LeftSideData.ClusterAssignmentForEachRow == ClusterIdentity);
    InstancesOfThisCluster = InstancesOfThisCluster(1);%take first one
    
    SizeOfThisCluster =  LeftSideData.ClusterSizeForEachRow(InstancesOfThisCluster);
    if SizeOfThisCluster > thresholdForClusterInclusion
        ListOfLatents = [ListOfLatents LeftSideData.ClusterCentersForEachRow(InstancesOfThisCluster,:)']; %also store name so we can map names to indecides when we compute externalSims
        ListOfLatentIdentities = [ListOfLatentIdentities UniqueLeftSideClusters(thisLeftClusterIndex)];
        if ~isempty(TranslatedClusterCenters)
            %match identity
            for thisClusterIndex = 1:size(TranslatedClusterCenters,2)
                if TranslatedClusterCenters(thisClusterIndex).ClusterIdentity == ClusterIdentity
                    thisTranslatedClusterCenter =  TranslatedClusterCenters(thisClusterIndex).TranslatedCenter;
                end
            end
            ListOfTranslatedLatents = [ListOfTranslatedLatents thisTranslatedClusterCenter'];
        else
            ListOfTranslatedLatents = [];
        end
    else
        fprintf('Cluster %d had only %d fixations and was excluded\n', ...
            UniqueLeftSideClusters(thisLeftClusterIndex),...
            SizeOfThisCluster);
    end
end

UniqueRightSideClusters = unique(RightSideData.ClusterAssignmentForEachRow);
ListOfExemplars = [];
ListOfExemplarIdentities = [];

for thisRightClusterIndex = 1:size(UniqueRightSideClusters,1)
    %find an instance of this cluster so we can get the cluster center
    %coords
    
    
    %could prune clusters here and not include it in the list if cluster
    %membership is less than like 3 fixations
    
    InstancesOfThisCluster = find(RightSideData.ClusterAssignmentForEachRow == UniqueRightSideClusters(thisRightClusterIndex));
    InstancesOfThisCluster = InstancesOfThisCluster(1);%take first one
    
    SizeOfThisCluster =  RightSideData.ClusterSizeForEachRow(InstancesOfThisCluster);
    if SizeOfThisCluster > thresholdForClusterInclusion
        ListOfExemplars = [ListOfExemplars RightSideData.ClusterCentersForEachRow(InstancesOfThisCluster,:)'];
        ListOfExemplarIdentities = [ListOfExemplarIdentities UniqueRightSideClusters(thisRightClusterIndex)];
    else
        fprintf('Cluster %d had only %d fixations and was excluded\n', ...
            UniqueRightSideClusters(thisRightClusterIndex),...
            SizeOfThisCluster);
    end
end

%ListOfLatents = LeftSideData.ClusterCentersForEachRow';
%ListOfExemplars = RightSideData.ClusterCentersForEachRow';

externalSims = zeros(size(ListOfLatents,2), size(ListOfExemplars,2));
numTotalAdded = 0;
%now populate externalSims with transitions
for thisTransition = 1:size(Transitions,2)
    LeftClusters = Transitions{thisTransition}.PriorClusters;
    LeftClusters';
    RightClusters = Transitions{thisTransition}.CurrentClusters;
    
    if useOnlyUniqueFixations == 2 %using sum model, not product model
        %go through each combination, increase externalSims by counts of
        %fixations
        LeftClustersUnique = unique(LeftClusters);
        RightClustersUnique = unique(RightClusters);
        
        %go through all combinations, increase externalSims by sum of
        %fixations
        for thisLeftClustersUniqueIndex = 1:size(LeftClustersUnique,1)
            thisLeftClustersUnique = LeftClustersUnique(thisLeftClustersUniqueIndex);
            %now find the number of this unique cluster appears in original
            %list
            
            
            
            numberOfThisUniqueLeft = sum(thisLeftClustersUnique==LeftClusters);
            
            if randomizeHintClusters %pick random cluster for this transition for this cluster; done with replacement
                if ~isempty(UniqueLeftSideClusters) %we may have pruned everything
                    thisLeftClustersUnique = randsample(LeftSideData.ClusterAssignmentForEachRow, 1);%pick a new one
                else
                    %leave the same; will get pruned below anyway
                end
                
            end
            
            leftIndexIntoSims = find(ListOfLatentIdentities==thisLeftClustersUnique);%if we pruned above, will get empty here, which will be a good check.
            
            for thisRightClustersUniqueIndex = 1:size(RightClustersUnique,1)
                %now find the number of this unique cluster
                thisRightClustersUnique = RightClustersUnique(thisRightClustersUniqueIndex);
                %now find the number of this unique cluster appears in original
                %list
                
                numberOfThisUniqueRight = sum(thisRightClustersUnique==RightClusters);
                
                if randomizeHintClusters %pick random cluster for this transition for this cluster; done with replacement
                    if ~isempty(UniqueRightSideClusters) %we may have pruned everything
                        thisRightClustersUnique = randsample(RightSideData.ClusterAssignmentForEachRow, 1);%pick a new one
                    else
                        %leave the same; will get pruned below anyway
                    end
                    
                end
                
                rightIndexIntoSims = find(ListOfExemplarIdentities==thisRightClustersUnique);
                if ~isempty(leftIndexIntoSims) && ~isempty(rightIndexIntoSims)
                    externalSims(leftIndexIntoSims,rightIndexIntoSims) = ...
                        externalSims(leftIndexIntoSims,rightIndexIntoSims) + numberOfThisUniqueLeft + numberOfThisUniqueRight;
                    numTotalAdded = numTotalAdded+ numberOfThisUniqueLeft + numberOfThisUniqueRight;
                end
            end
        end
    else %multiplicative model
        for thisLeftClusterIndex = 1:size(LeftClusters,1)
            thisLeftCluster = LeftClusters(thisLeftClusterIndex);%problem- not all of the cluster names in LeftClusters will have made it into the ListOfLatentIdentities
            for thisRightClusterIndex = 1:size(RightClusters,1)
                thisRightCluster=RightClusters(thisRightClusterIndex);
                
                
                leftIndexIntoSims = find(ListOfLatentIdentities==thisLeftCluster);%if we pruned above, will get empty here, which will be a good check.
                rightIndexIntoSims = find(ListOfExemplarIdentities==thisRightCluster);
                if ~isempty(leftIndexIntoSims) && ~isempty(rightIndexIntoSims)
                    externalSims(leftIndexIntoSims,rightIndexIntoSims) = ...
                        externalSims(leftIndexIntoSims,rightIndexIntoSims) + 1;
                end
            end
        end
    end
end

 %if max(size(externalSims))<15 && max(size(externalSims))>10
 %    a=1
 %end

if actuallyFitAbsurdist
    if ~useOnlyUniqueFixations
        externalSims = log(externalSims + 1); %take log of fixation count to de-emphasize big fixation counts
    end
end

[Correspondences, GreedyMaxes, GreedyMaxValues, ClusterOffsetDeviation,...
    GoodLatentCoords, CorrespondingExemplarCoords] = fitAbsurdist(ListOfLatents, ListOfExemplars, externalSims,...
    thisImageName, ...
    examinerName,...
    bigImage, latentVerticalShift, ExemplarOffset, ExemplarVerticalShift, ...
    plottingLineWidthThreshold, ListOfTranslatedLatents, numberOfAbsurdistIternations,...
    actuallyFitAbsurdist, assignUsingMatchedPairs, ActuallyDisplayImages,...
    normalizeTransitionMatrix,ZeroWeakCorrespondences);


