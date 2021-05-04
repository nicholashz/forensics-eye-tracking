function   [goodClusters, badClusters] = plotPointsOnFigures(x, clustMembsCell, clustCent, ...
        NumPointsInClusterToBeABadCluster, ActuallyDisplayImages,...
        PointsXOffset, PointsYOffset, cVec, numClust, a)
        
    badClusters = [];
    goodClusters = [];
    for k = 1:numClust
        myMembers = clustMembsCell{k};
        myClustCen = clustCent(:,k);
        if (length(myMembers)>NumPointsInClusterToBeABadCluster) %based on scatterplot
            if ActuallyDisplayImages
                theHandlePoints = plot(x(1,myMembers)+PointsXOffset,x(2,myMembers)+PointsYOffset,[cVec(mod(k-1, length(cVec))+1) 'o'], 'MarkerSize', 6);
                theHandle = plot(myClustCen(1)+PointsXOffset,myClustCen(2)+PointsYOffset,'o','MarkerEdgeColor','r','MarkerFaceColor',cVec(mod(k-1, length(cVec))+1), 'MarkerSize',8);
                set(theHandlePoints(1),'color',a(mod(k-1,size(a,1))+1,:));
                set(theHandlePoints(1),'MarkerFaceColor',a(mod(k-1,size(a,1))+1,:));
                set(theHandle(1),'MarkerFaceColor',a(mod(k-1,size(a,1))+1,:));
            end
            goodClusters = [goodClusters k];
        else
            badClusters = [badClusters k];
        end
    end
    