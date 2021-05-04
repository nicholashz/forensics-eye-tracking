close all

%clear

%strategy:Find all fixations in detailedBF_88, e.g.
%FixationsByExaminerAndImage{20,6}.DetailedBF_88

%group by left and right sides
%do clustering with a bandwidth of 30 or 45, assign a cluster number to
%each fixation, along with a cluster center
%send cluster center or centroid of fixations in that cluster to absurdist
%weight each item by number of fixations
%run absurdist, record number of high-quality correspondences



%Match model with Nick's:
%22 or 66 bandwith
%no pruning of fixations based on cluster size
%normalize rows and columns
%still do greedy max instead of hungarian
%threshold late- do in Export
%throw out correspondences with one temporal sequence only

close all
rng(47408);%fix random number generator

ActuallySaveImages = 0;

ActuallyDisplayImages = 0;

plottingImages = 0; %if 1, plot images, which unfortuantely can't be released with the source code


modelparams.numberOfAbsurdistIternations = 1000;

%BandwidthsToAnalyze = [30 45 60 90 120];%also change below

%other additions
%Code right-left transition hints
%restrict hints to just first 1/3 or last 1/3 of trial
%use only last cluster of first sequence and first cluster of subsequent
%sequence

%modelparams.actuallyFitAbsurdist = input('Actually fit absurdist? (1=yes, 0=no) ');
modelparams.actuallyFitAbsurdist = 0;% if 0, only use temporal sequence information to find correspondences

if modelparams.actuallyFitAbsurdist == 1
    BandwidthsToAnalyze = [22];%bandwidth for clustering; %run only one at a time
    modelparams.thresholdForClusterInclusion = 2; %need at least 3 fixations in a cluster to be included
    
else
    BandwidthsToAnalyze = [66];%bandwidth for clustering;
    %modelparams.thresholdForClusterInclusion = 1; %need at least 2 fixations in a cluster to be included
    modelparams.thresholdForClusterInclusion = 0;%no prune
end

if modelparams.actuallyFitAbsurdist
    plottingLineWidthThreshold = .7; %based on eye; 2 might also work. %this doesn't affect actual model fits. Just for plotting
else
    plottingLineWidthThreshold = .3; %based on eye; 2 might also work.
end


%modelparams.UseDetailedBF88Only = input('Restrict to DetailedBF88 only (2 = detail only, 1=yes, 0=no) ');
modelparams.UseDetailedBF88Only = 2;%final model used detail fixations
%modelparams.UseDetailedBF88Only = 0;

%modelparams.UseOnlyLastAndFirstCluster = input('Restrict to last cluster of prior sequence and first cluster of current sequence? (1-yes, 0=no) ');
%modelparams.UseOnlyLastAndFirstCluster = input('Restrict to last cluster of prior sequence and first cluster of current sequence? (1-yes, 0=no) ');

%modelparams.useOnlyUniqueFixations = input('Use unique clusters only? (2 = use sum model, 1=yes, 0=no (multiplicative model) ');
modelparams.useOnlyUniqueFixations= 1;%if there are multiple times (fixations) that a cluster number enters into a transition, only mark it once if 1. Technically uses multiplicative model but multiplies by 1...
%modelparams.assignUsingMatchedPairs = 1
%setForassignUsingMatchedPairs = [0 1];
%setForassignUsingMatchedPairs = input('Use MatchedPairs assignment? (1=yes, 0=no) ');
setForassignUsingMatchedPairs = 0;
%setForIncludeRightToLeftTransitions = input('Include Right-to-Left Transitions? (1-yes, 0=no) ');
setForIncludeRightToLeftTransitions = [0];

%setForIncludeRightToLeftTransitions = [0 1];
setForUseOnlyLastAndFirstCluster = [0];
%setForUseOnlyLastAndFirstCluster = input('Include Only Last and First cluster? (1-yes, 0=no) ');
setForRestrictToFirstThirdOfFixations = [0];%0 to do all, 1 to restrict to first third, 2 to restrict to last third

modelparams.randomizeHintClusters = [0];

modelparams.normalizeTransitionMatrix = 1;

modelparams.ZeroWeakCorrespondences = 0;%if 1, any correspondence less than 2 in transition matrix gets set to 0. Took out because not great justification

%modelparams.IncludeRightToLeftTransitions = input('Include Right-Left Transitions (1=yes, 0=no) ');%if 1, include both left-to-right (default) and right-to-left transitions
for assignUsingMatchedPairs = setForassignUsingMatchedPairs
    modelparams.assignUsingMatchedPairs = assignUsingMatchedPairs;
    
    for IncludeRightToLeftTransitions = setForIncludeRightToLeftTransitions
        modelparams.IncludeRightToLeftTransitions = IncludeRightToLeftTransitions;
        
        for UseOnlyLastAndFirstCluster = setForUseOnlyLastAndFirstCluster
            modelparams.UseOnlyLastAndFirstCluster = UseOnlyLastAndFirstCluster;
            %modelparams.UseOnlyLastAndFirstCluster = input('Restrict to last cluster of prior sequence and first cluster of current sequence? (1-yes, 0=no) ');
            for RestrictToFirstThirdOfFixations = setForRestrictToFirstThirdOfFixations
                modelparams.RestrictToFirstThirdOfFixations = RestrictToFirstThirdOfFixations;
                
                addpath('rgb');
                
                [hex, name] = GetListOfColors();
                
                ColorsToUse = fliplr([1:4:size(name,1)-4]);
                RGBValuesForClusters = GetRGBValues(name, ColorsToUse);
                
                %FolderWithDriftCorrectedData = '/Volumes/TimeMachineBackups/WBData/CombinedDataDriftCorrected/';
                if ~exist('FixationsByExaminerAndImage', 'var')
                    FileWithFixationData = 'AllFixationsByImageAndExaminer.mat';
                    load(FileWithFixationData);
                end
                
                %FolderWithDriftCorrectedData = [pwd '/DFT1BFiles/'];
                
                %FolderToSaveAbsurdistImages = '/Volumes/TimeMachineBackups/WBData/2020Analyses/AbsurdistFits';
                
                if modelparams.actuallyFitAbsurdist
                    FolderToSaveAbsurdistImages = '/AbsurdistFits';
                else
                    FolderToSaveAbsurdistImages = 'TempSeqFits';
                end
                
                
                if modelparams.randomizeHintClusters == 1 && modelparams.useOnlyUniqueFixations == 2 %only set up for sum model
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_RandomHints'];
                end
                
                %FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages sprintf('BW%d',
                if modelparams.UseDetailedBF88Only == 2
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_Detail'];
                elseif modelparams.UseDetailedBF88Only == 1
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_DetailBF88'];
                end
                if modelparams.useOnlyUniqueFixations == 1
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_JstUnq'];
                end
                
                if modelparams.useOnlyUniqueFixations == 2
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_SumModel'];
                end
                
                if modelparams.IncludeRightToLeftTransitions
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_IncRLTrans'];
                end
                
                if modelparams.UseOnlyLastAndFirstCluster
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_UseLstFrstClust'];
                end
                
                if modelparams.RestrictToFirstThirdOfFixations==1
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_JustFirstThird'];
                end
                
                if modelparams.RestrictToFirstThirdOfFixations==2
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_JustLastThird'];
                end
                
                if modelparams.normalizeTransitionMatrix
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_NormTransMat'];
                    
                end
                if modelparams.thresholdForClusterInclusion == 0
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_NoClusterPrune'];
                    
                end
                if modelparams.ZeroWeakCorrespondences == 1
                    FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '_KillWeakLinks'];
                end
                
                
                %append BW
                FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages sprintf('_BW%d', BandwidthsToAnalyze(1))];
                
                [~,SaveName] = fileparts(FolderToSaveAbsurdistImages);
                
                Suffix = SaveName;
                SaveName = [SaveName '.mat'];
                
                
                FolderToSaveAbsurdistImages = [FolderToSaveAbsurdistImages '/']; %#ok<*AGROW>
                
                if ActuallySaveImages
                    if ~exist(FolderToSaveAbsurdistImages, 'dir')
                        system(sprintf('mkdir %s', FolderToSaveAbsurdistImages));
                    end
                    
                    system(sprintf('rm %s*', FolderToSaveAbsurdistImages));
                end
                
                %FoldernameForClusteredFixations = '/Volumes/TimeMachineBackups/WBData/2020Analyses/ClusteredFixationFiles/';
                
                %FoldernameForImageBasedFiles = '/Volumes/TimeMachineBackups/WBData/2020Analyses/ClusteredFixationFilesByImage/';
                
                NumPointsInClusterToBeABadCluster = 0;
                %NumPointsInClusterToBeABadCluster = 0;
                NumBadFixations = 0; %not doing pruning on this data
                
                AbsurdistCorrespondences = cell(size(ImageList,1), numExaminers, size(BandwidthsToAnalyze,2));
                
                %for thisImageToAnalyze = 1:size(ImageList,1)
                for thisImageToAnalyze =1:size(ImageList,1)
                %          for thisImageToAnalyze = 3
                    
                    
                    thisImageName = char(ImageList{thisImageToAnalyze,1});
                    
                    %if contains(thisImageName, 'CW') || contains(thisImageName, 'CE')
                    fprintf('Working on image %s\n', thisImageName);
                    
                    %for thisSubject = 49
                    %for thisSubject = 6
                    for thisSubject = 1:numExaminers
                             %for thisSubject = 14
                        AllLeftFixations = [];
                        AllRightFixations = [];
                        DataForThisSubjectOnThisImage = FixationsByExaminerAndImage{thisImageToAnalyze, thisSubject};
                        
                        
                        if ~isempty(DataForThisSubjectOnThisImage)
                            %pull off ground truth and decision
                            
                            %estXPoints = DataForThisSubjectOnThisImage.estX_OtherImg;
                            %estYPoints = DataForThisSubjectOnThisImage.estY_OtherImg;
                            Outcome = DataForThisSubjectOnThisImage.Outcome(1);
                            Conclusion = DataForThisSubjectOnThisImage.Conclusion(1);
                            ConclusionsDetail = DataForThisSubjectOnThisImage.ConclusionDetail(1);
                            GroundTruth = DataForThisSubjectOnThisImage.Mating(1);
                            Borderline = DataForThisSubjectOnThisImage.Borderline(1);
                            Difficulty = DataForThisSubjectOnThisImage.Difficulty(1);
                            
                            %AllAnalysisFixations =  DataForThisSubjectOnThisImage.Ph
                            LeftFixations = find(DataForThisSubjectOnThisImage.Image=='Left');
                            RightFixations = find(DataForThisSubjectOnThisImage.Image=='Right');
                            
                            AllCombined = [LeftFixations; RightFixations];
                            
                            DetailBF88 = find(DataForThisSubjectOnThisImage.DetailedBF_88 > 0);
                            Detail = find(DataForThisSubjectOnThisImage.Subphase=='Deciding');%was using more complex; re-run
                            
                            if modelparams.UseDetailedBF88Only == 2
                                LeftDetailBF88Fixations = intersect(LeftFixations,Detail);
                                RightDetailBF88Fixations = intersect(RightFixations,Detail);
                                AllCombinedDetailBF88Fixations = intersect(AllCombined, Detail);
                            elseif modelparams.UseDetailedBF88Only == 1
                                LeftDetailBF88Fixations = intersect(LeftFixations,DetailBF88);
                                RightDetailBF88Fixations = intersect(RightFixations,DetailBF88);
                                AllCombinedDetailBF88Fixations = intersect(AllCombined, DetailBF88);
                            else
                                
                                LeftDetailBF88Fixations = LeftFixations;
                                RightDetailBF88Fixations = RightFixations;
                                AllCombinedDetailBF88Fixations = AllCombined;
                            end
                            
                            AllLeftFixations = [DataForThisSubjectOnThisImage.FixX(LeftDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.FixY(LeftDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.Row(LeftDetailBF88Fixations,:)];
                            
                            AllLeftFixationsTranslatedToRightOnMated = [DataForThisSubjectOnThisImage.estX_OtherImg(LeftDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.estY_OtherImg(LeftDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.Row(LeftDetailBF88Fixations,:)];
                            
                            AllRightFixations = [DataForThisSubjectOnThisImage.FixX(RightDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.FixY(RightDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.Row(RightDetailBF88Fixations,:)];
                            %          AllComparisonFixationsLeft = [AllComparisonFixationsLeft; AllLeftFixations]; %#ok<*AGROW>
                            %         AllComparisonFixationsRight = [AllComparisonFixationsRight; AllRightFixations];
                            
                            ImageOnRight = double(DataForThisSubjectOnThisImage.Image(AllCombinedDetailBF88Fixations,:)=='Right');
                            
                            AllCombinedFixations = [DataForThisSubjectOnThisImage.FixX(AllCombinedDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.FixY(AllCombinedDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.Row(AllCombinedDetailBF88Fixations,:),...
                                DataForThisSubjectOnThisImage.SeqID(AllCombinedDetailBF88Fixations,:),...
                                ImageOnRight];
                            
                            %now sort so we can figure out transitions
                            AllCombinedFixations = sortrows(AllCombinedFixations,3);
                            %add additional columns for cluster number eventually
                            ClusterNumberForEachFixation = zeros(size(AllCombinedFixations,1), size(BandwidthsToAnalyze,2));
                            ClusterSizeForEachFixation = zeros(size(AllCombinedFixations,1), size(BandwidthsToAnalyze,2));
                            MasterRowForEachFixation = zeros(size(AllCombinedFixations,1), size(BandwidthsToAnalyze,2));
                            
                            
                            %now add index numbers in 4th column
                            AllLeftFixations = [AllLeftFixations [1:size(AllLeftFixations,1)]'];
                            AllRightFixations = [AllRightFixations [1:size(AllRightFixations,1)]'];
                            
                            %now encode temporal information. Go through each sequence
                            %and code it on the left side or right side from Image
                            %column.
                            
                            
                            
                            
                            if size(AllLeftFixations,1) > 0 && size(AllRightFixations,1)>0 %may not have fixations for this image
                                TotalRows = 0;%it grows anyway
                                
                                if plottingImages
                                %load the iamge
                                [bigImage, latentVerticalShift, ExemplarOffset, ExemplarVerticalShift] =loadImageAndComputeOffsets(thisImageName);
                                else
                                    bigImage = [];
                                    latentVerticalShift= 0;
                                    ExemplarOffset=0;
                                    ExemplarVerticalShift = 0;
                                end
                                for bandwidthIndex = 1:size(BandwidthsToAnalyze,2)
                                    bandwidth = BandwidthsToAnalyze(bandwidthIndex);
                                    
                                    [ClusterAssignmentForEachRow, ClusterCentersForEachRow, ClusterSizeForEachRow,...
                                        ColorsUsedInPlotting, MasterRowAssignment] = ...
                                        MeanShiftClusteringOnOneImage([], [], ...
                                        AllLeftFixations,...
                                        AllRightFixations, bigImage, latentVerticalShift, ExemplarOffset, ...
                                        ExemplarVerticalShift, TotalRows, bandwidth, NumPointsInClusterToBeABadCluster,...
                                        RGBValuesForClusters, ActuallyDisplayImages );
                                    drawnow
                                    
                                    
                                    for imageSide = 1:2
                                        
                                        ClusteringData{imageSide, bandwidthIndex}.ClusterAssignmentForEachRow = ClusterAssignmentForEachRow{imageSide}'; %#ok<*SAGROW>
                                        ClusteringData{imageSide, bandwidthIndex}.ClusterCentersForEachRow = ClusterCentersForEachRow{imageSide};
                                        ClusteringData{imageSide, bandwidthIndex}.ClusterSizeForEachRow = ClusterSizeForEachRow{imageSide};
                                        ClusteringData{imageSide, bandwidthIndex}.MasterRowAssignment = MasterRowAssignment{imageSide};
                                    end
                                    
                                    %add translated points if exist
                                    imageSide = 1;%leave open possibility of both sides later on
                                    if ~isempty(AllLeftFixationsTranslatedToRightOnMated(:,imageSide))
                                        ClusteringData{1, bandwidthIndex}.estXTranslatedToRightSide =AllLeftFixationsTranslatedToRightOnMated(:,1);
                                        ClusteringData{1, bandwidthIndex}.estYTranslatedToRightSide =AllLeftFixationsTranslatedToRightOnMated(:,2);
                                        
                                        %now go through and identify the closest fixation to
                                        %each cluster center. We will use that to determine the
                                        %correct translated point in the right image for each
                                        %cluster
                                        allClusters = unique( ClusterAssignmentForEachRow{imageSide});
                                        for thisClusterIndex = 1:size(allClusters,2)
                                            thisCluster = allClusters(thisClusterIndex);
                                            fixationRowsWithThisCluster = find(ClusterAssignmentForEachRow{imageSide} == thisCluster);
                                            %now find distance of each point in this cluster
                                            %to the cluster center
                                            clusterCenterForThisCluster = ClusterCentersForEachRow{imageSide}(fixationRowsWithThisCluster(1),:);
                                            LocationsOfFixationsInThisCluster = AllLeftFixations(fixationRowsWithThisCluster,:);
                                            %now find closest point
                                            Dist = sqrt(sum((clusterCenterForThisCluster - LocationsOfFixationsInThisCluster(:,1:2)) .^ 2, 2));
                                            rowOfClosest = find(min(Dist)==Dist);
                                            rowOfClosest = rowOfClosest(1);%deal with ties
                                            
                                            masterRowOfClosest = LocationsOfFixationsInThisCluster(rowOfClosest,3);
                                            originalRowToGetEstXFrom = find(AllLeftFixationsTranslatedToRightOnMated(:,3)==masterRowOfClosest);
                                            TranslatedClusterCenters(thisClusterIndex).Center = clusterCenterForThisCluster;%for sanity
                                            TranslatedClusterCenters(thisClusterIndex).TranslatedCenter = AllLeftFixationsTranslatedToRightOnMated(originalRowToGetEstXFrom,1:2);%translated to right side
                                            TranslatedClusterCenters(thisClusterIndex).ClusterIdentity = thisCluster;%for sanity
                                            
                                        end
                                    else
                                        TranslatedClusterCenters = [];
                                    end
                                    
                                    
                                    %go back and assign a cluster number to each
                                    %fixation and then create a list of cluster
                                    %transitions
                                    for imageSide = 1:2
                                        
                                        %strategy: go through each row for this subject, search in
                                        %each clustering data record for that master row
                                        %assignment. When find it, copy over the cluster data into
                                        %new columns
                                        for  thisRow = 1:size(AllCombinedFixations,1)
                                            MasterRowNumber = AllCombinedFixations(thisRow,3);
                                            %now try to find that master row number in
                                            %ClusteringData
                                            RowNumberInClusteringData = find(ClusteringData{imageSide, bandwidthIndex}.MasterRowAssignment==MasterRowNumber);
                                            if size(RowNumberInClusteringData,1)==0%check to see if it is on this side, otherwise check other side
                                                %every row in this subject's data should be in the clustering data; throw error if not
                                                OtherRowNumberInClusteringData = find(ClusteringData{3-imageSide, bandwidthIndex}.MasterRowAssignment==MasterRowNumber);
                                                if size(OtherRowNumberInClusteringData,1)==0
                                                    error('subject''s row number not found in either side clustering data\n');
                                                    
                                                end
                                            else
                                                ClusterNumberForEachFixation(thisRow, bandwidthIndex)=ClusteringData{imageSide, bandwidthIndex}.ClusterAssignmentForEachRow(RowNumberInClusteringData,1);
                                                ClusterSizeForEachFixation(thisRow, bandwidthIndex)=ClusteringData{imageSide, bandwidthIndex}.ClusterSizeForEachRow(RowNumberInClusteringData,1);
                                                MasterRowForEachFixation(thisRow, bandwidthIndex)=ClusteringData{imageSide, bandwidthIndex}.MasterRowAssignment(RowNumberInClusteringData,1);
                                                
                                            end
                                        end
                                        
                                    end
                                    
                                    
                                    
                                    %at this point we have independent clustering solutions
                                    %on the left and right images, stored in ClusterNumberForEachFixation
                                    %now find all left sequence followed immediately by
                                    %a right sequence. Store transitions as pairs of
                                    %left side and right side linked clusters
                                    currentTransitionNumber = 0;
                                    Transitions = {};
                                    
                                    if modelparams.RestrictToFirstThirdOfFixations==1
                                        allUniqueSequences = unique(AllCombinedFixations(1:floor(size(AllCombinedFixations,1)/3),4));%this logic only works because sequences are already sorted
                                    elseif modelparams.RestrictToFirstThirdOfFixations==2
                                        allUniqueSequences = unique(AllCombinedFixations(floor(size(AllCombinedFixations,1)*2/3):end,4));%this logic only works because sequences are already sorted
                                    else
                                        allUniqueSequences = unique(AllCombinedFixations(:,4));%this logic only works because sequences are already sorted
                                    end
                                    for thisUniqueIndex = 1:size(allUniqueSequences,1)
                                        %find first instance of it
                                        %V1 = find(V, 1, 'first')
                                        
                                        thisFirstIndex = find(AllCombinedFixations(:,4)== allUniqueSequences(thisUniqueIndex), 1);
                                        %first check to see if it is on the right
                                        firstFixOfSequenceIsOnRight = AllCombinedFixations(thisFirstIndex,5);
                                        if firstFixOfSequenceIsOnRight %was on right, so look to see if previous fixations sequence was consecutive
                                            if thisUniqueIndex > 1 && AllCombinedFixations(thisFirstIndex-1,4) == AllCombinedFixations(thisFirstIndex,4)-1
                                                firstFixOfPreviousSequenceIsOnLeft =  AllCombinedFixations(thisFirstIndex-1,5) == 0;
                                                if firstFixOfPreviousSequenceIsOnLeft
                                                    %found a consecutive left-right sequence, so collect all left fixation and
                                                    %right fixations associated with the two sequences and assign them to the same transition number
                                                    currentTransitionNumber = currentTransitionNumber + 1;
                                                    rowsForPriorSequence = find(AllCombinedFixations(:,4) == AllCombinedFixations(thisFirstIndex-1,4));
                                                    rowsForCurrentSequence = find(AllCombinedFixations(:,4) == AllCombinedFixations(thisFirstIndex,4));
                                                    if modelparams.UseOnlyLastAndFirstCluster %only pull off last cluster of previous sequence and first of next sequence
                                                        lastClusterInPriorSequence =  ClusterNumberForEachFixation(rowsForPriorSequence(end));
                                                        allClustersInPriorSequence = ClusterNumberForEachFixation(rowsForPriorSequence);%just for debugging
                                                        matchingPriorClustersRows = find(ClusterNumberForEachFixation == lastClusterInPriorSequence);
                                                        rowsForPriorSequence = intersect(rowsForPriorSequence, matchingPriorClustersRows);
                                                        
                                                        firstClusterInCurrentSequence =  ClusterNumberForEachFixation(rowsForCurrentSequence(1));
                                                        allClustersInCurrentSequence = ClusterNumberForEachFixation(rowsForCurrentSequence);%just for debugging
                                                        matchingCurrentClustersRows = find(ClusterNumberForEachFixation == firstClusterInCurrentSequence);
                                                        rowsForCurrentSequence = intersect(rowsForCurrentSequence, matchingCurrentClustersRows);
                                                        
                                                    end
                                                    if modelparams.useOnlyUniqueFixations == 0 || modelparams.useOnlyUniqueFixations == 2
                                                        Transitions{currentTransitionNumber}.PriorClusters = ClusterNumberForEachFixation(rowsForPriorSequence); %pull off cluster numbers for these transitions
                                                        Transitions{currentTransitionNumber}.CurrentClusters = ClusterNumberForEachFixation(rowsForCurrentSequence);%pull off cluster numbers for these transitions
                                                    else
                                                        Transitions{currentTransitionNumber}.PriorClusters = unique(ClusterNumberForEachFixation(rowsForPriorSequence)); %pull off cluster numbers for these transitions
                                                        Transitions{currentTransitionNumber}.CurrentClusters = unique(ClusterNumberForEachFixation(rowsForCurrentSequence));%pull off cluster numbers for these transitions
                                                    end
                                                end
                                            end
                                            
                                        end
                                    end
                                    
                                    if modelparams.IncludeRightToLeftTransitions %also include right-to-left transitions in hints
                                        for thisUniqueIndex = 1:size(allUniqueSequences,1)
                                            %find first instance of it
                                            %V1 = find(V, 1, 'first')
                                            
                                            thisFirstIndex = find(AllCombinedFixations(:,4)== allUniqueSequences(thisUniqueIndex), 1);
                                            %first check to see if it is on the right
                                            firstFixOfSequenceIsOnLeft = AllCombinedFixations(thisFirstIndex,5)==0;
                                            if firstFixOfSequenceIsOnLeft %was on right, so look to see if previous fixations sequence was consecutive
                                                if thisUniqueIndex > 1 && AllCombinedFixations(thisFirstIndex-1,4) == AllCombinedFixations(thisFirstIndex,4)-1
                                                    firstFixOfPreviousSequenceIsOnRight =  AllCombinedFixations(thisFirstIndex-1,5) == 1;
                                                    if firstFixOfPreviousSequenceIsOnRight
                                                        
                                                        %found a consecutive right-left sequence, so collect all right fixation and
                                                        %left fixations associated with the two sequences and assign them to the same transition number
                                                        currentTransitionNumber = currentTransitionNumber + 1;
                                                        rowsForPriorSequence = find(AllCombinedFixations(:,4) == AllCombinedFixations(thisFirstIndex-1,4));
                                                        rowsForCurrentSequence = find(AllCombinedFixations(:,4) == AllCombinedFixations(thisFirstIndex,4));
                                                        if modelparams.UseOnlyLastAndFirstCluster %only pull off last cluster of previous sequence and first of next sequence
                                                            lastClusterInPriorSequence =  ClusterNumberForEachFixation(rowsForPriorSequence(end));
                                                            allClustersInPriorSequence = ClusterNumberForEachFixation(rowsForPriorSequence);%just for debugging
                                                            matchingPriorClustersRows = find(ClusterNumberForEachFixation == lastClusterInPriorSequence);
                                                            rowsForPriorSequence = intersect(rowsForPriorSequence, matchingPriorClustersRows);
                                                            
                                                            firstClusterInCurrentSequence =  ClusterNumberForEachFixation(rowsForCurrentSequence(1));
                                                            allClustersInCurrentSequence = ClusterNumberForEachFixation(rowsForCurrentSequence);%just for debugging
                                                            matchingCurrentClustersRows = find(ClusterNumberForEachFixation == firstClusterInCurrentSequence);
                                                            rowsForCurrentSequence = intersect(rowsForCurrentSequence, matchingCurrentClustersRows);
                                                            
                                                        end
                                                        if ~modelparams.useOnlyUniqueFixations
                                                            %reverse these because doing
                                                            %right-left transitions and
                                                            %transition table is set up for
                                                            %left-right transitions
                                                            Transitions{currentTransitionNumber}.CurrentClusters = ClusterNumberForEachFixation(rowsForPriorSequence); %pull off cluster numbers for these transitions
                                                            Transitions{currentTransitionNumber}.PriorClusters = ClusterNumberForEachFixation(rowsForCurrentSequence);%pull off cluster numbers for these transitions
                                                        else
                                                            Transitions{currentTransitionNumber}.CurrentClusters = unique(ClusterNumberForEachFixation(rowsForPriorSequence)); %pull off cluster numbers for these transitions
                                                            Transitions{currentTransitionNumber}.PriorClusters = unique(ClusterNumberForEachFixation(rowsForCurrentSequence));%pull off cluster numbers for these transitions
                                                        end
                                                    end
                                                end
                                                
                                            end
                                        end
                                    end
                                    
                                    
                                    [Correspondences, GreedyMaxes, GreedyMaxValues, ListOfLatentIdentities, ...
                                        ListOfExemplarIdentities,...
                                        ListOfLatents,...
                                        ListOfExemplars, ListOfTranslatedLatents, ...
                                        ClusterOffsetDeviation,...
                                        GoodLatentCoords, CorrespondingExemplarCoords,...
                                        externalSims] = findAbsurdistCorrespondences( ClusteringData{1, bandwidthIndex},  ...%what is clustering
                                        ClusteringData{2, bandwidthIndex}, ...
                                        bigImage, latentVerticalShift, ExemplarOffset, ExemplarVerticalShift,...
                                        thisImageName,ExaminerList{thisSubject}, Transitions,...
                                        plottingLineWidthThreshold, modelparams.thresholdForClusterInclusion,...
                                        modelparams.useOnlyUniqueFixations, TranslatedClusterCenters,...
                                        modelparams.numberOfAbsurdistIternations,...
                                        modelparams.actuallyFitAbsurdist,...
                                        modelparams.assignUsingMatchedPairs,...
                                        ActuallyDisplayImages,...
                                        modelparams.randomizeHintClusters,modelparams.normalizeTransitionMatrix,...
                                        modelparams.ZeroWeakCorrespondences);
                                    
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.NotEnoughFixations = 0;
                                    
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Correspondences = Correspondences;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.GreedyMaxes = GreedyMaxes;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.GreedyMaxValues = GreedyMaxValues;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.externalSims = externalSims;
                                    
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ClusteringData = ClusteringData;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ListOfLatentIdentities = ListOfLatentIdentities;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ListOfExemplarIdentities = ListOfExemplarIdentities;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ListOfLatents = ListOfLatents;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ListOfExemplars = ListOfExemplars;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ListOfTranslatedLatents = ListOfTranslatedLatents;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ClusterOffsetDeviation = ClusterOffsetDeviation;
                                    
                                    
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Transitions = Transitions;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Image = thisImageName;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Examiner = ExaminerList{thisSubject};
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Outcome = Outcome;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Conclusion = Conclusion;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.ConclusionsDetail = ConclusionsDetail;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.GroundTruth = GroundTruth;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Borderline = Borderline;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Difficult = Difficulty;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.SaveName = SaveName;
                                    
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.GoodLatentCoords = GoodLatentCoords;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.CorrespondingExemplarCoords = CorrespondingExemplarCoords;
                                    %now compute angle for each
                                    
                                    numGoodAngles = 0;
                                    listOfGoodAngles = [];
                                    for thisLatent = 1:size(GoodLatentCoords,2)
                                        if ~isnan(GoodLatentCoords(1,thisLatent)) && ~isnan(GoodLatentCoords(2,thisLatent))
                                            x1 = GoodLatentCoords(1,thisLatent);
                                            x2 = CorrespondingExemplarCoords(1,thisLatent)+ExemplarOffset;
                                            y1 = GoodLatentCoords(2,thisLatent)+latentVerticalShift;
                                            y2 = CorrespondingExemplarCoords(2,thisLatent)+ExemplarVerticalShift;
                                            %AngleOfCorrespondence = atan2(norm(cross([x1; y1],[x2; y2])), dot([x1; y1],[x2; y2]));
                                            AngleOfCorrespondence =getAngle([x1-x1 y1-y1],[x2-x1 y2-y1]);
                                            %if y2<y1
                                            % fprintf('%3.0f %3.0f %3.0f %3.0f Angle = %3.3f\n', x1, y1, x2, y2, AngleOfCorrespondence);
                                            %end
                                            listOfGoodAngles = [listOfGoodAngles AngleOfCorrespondence];
                                            
                                        end
                                    end
                                    CosVals = cos(deg2rad(listOfGoodAngles));
                                    
                                    SinVals = sin(deg2rad(listOfGoodAngles));
                                    [Tcirc, CritTcirc, ProbOfOurTcirc] = ComputeTcirc(CosVals, SinVals);
                                    fprintf('Tcirc = %3.3f\n', Tcirc);
                                    if isnan(Tcirc)
                                        AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.notEnoughClustersToComputeTCirc = 1;
                                    else
                                        AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.notEnoughClustersToComputeTCirc = 0;
                                    end
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.listOfGoodAngles = listOfGoodAngles;
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.Tcirc = Tcirc;
                                    
                                    %now compute measure of stress
                                    OverallStress = ComputeOverallStress(Correspondences, GreedyMaxes, GreedyMaxValues);
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.OverallStress = OverallStress;
                                    
                                    if modelparams.actuallyFitAbsurdist
                                        modelName = 'Absurdist';
                                    else
                                        modelName = 'TempSeq';
                                    end
                                    if ActuallyDisplayImages
                                        title(sprintf('Image %s Subject %s %s %s %s %s %s BW%d Tcirc = %3.1f %s', ...
                                            thisImageName, ExaminerList{thisSubject}, GroundTruth, Conclusion, Difficulty, ...
                                            Borderline, modelName, BandwidthsToAnalyze(bandwidthIndex),Tcirc, SaveName),'Interpreter', 'none');
                                    end
                                    %pause
                                    AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, bandwidthIndex}.SaveName = SaveName;
                                    
                                    if ActuallySaveImages
                                        %writeName = sprintf('%s%sFixations.png', FolderToSaveClusteredImages, thisImageName);
                                        %figure(1);
                                        %print(gcf, '-dpng', strcat(writeName));
                                        
                                        
                                        
                                        %writeName = sprintf('%s%s_%d_AnalysisClusteredFixations.png', FolderToSaveClusteredImages, thisImageName, bandwidth);
                                        
                                        %figure(10);
                                        %print(gcf, '-dpng', strcat(writeName));
                                        
                                        writeName = sprintf('%s%s_%s_%s_BW%d.png', FolderToSaveAbsurdistImages, thisImageName, ExaminerList{thisSubject}, Suffix,...
                                            BandwidthsToAnalyze(bandwidthIndex));
                                        
                                        
                                        figure(11);
                                        print(gcf, '-dpng', strcat(writeName));
                                    end
                                end
                            else
                                fprintf('Not enough fixations found for subject %d (%s), image pair %s\n', thisSubject, ExaminerList{thisSubject},thisImageName);
                                AbsurdistCorrespondences{thisImageToAnalyze, thisSubject, 1}.NotEnoughFixations = 1;
                            end
                            %end
                            %        pause
                            
                        else
                            fprintf('No fixations found for subject %d, imagepair %s\n', thisSubject, thisImageName);
                        end
                    end
                    
                end
                if modelparams.actuallyFitAbsurdist
                    save(['AbsurdistFits/' SaveName], 'AbsurdistCorrespondences', 'BandwidthsToAnalyze', 'FolderToSaveAbsurdistImages', 'ImageList', 'ExaminerList', 'numExaminers', 'SaveName', 'modelparams');
                else
                    save(['TempSeqFits/' SaveName], 'AbsurdistCorrespondences', 'BandwidthsToAnalyze', 'FolderToSaveAbsurdistImages', 'ImageList', 'ExaminerList', 'numExaminers', 'SaveName', 'modelparams');
                end
            end
        end
    end
end