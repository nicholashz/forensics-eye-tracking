function OverallStress = ComputeOverallStress(Correspondences, GreedyMaxes, GreedyMaxValues)

%strategy- compute mean activation for identified correspondences, mean
%activation for non-identified correspondences, compute ratio

sumOfAssigned = 0;
numAssigned = 0;
sumOfUnassigned = 0;
numUnassigned = 0;

for thisLeftCluster = 1:size(Correspondences,1)
    for thisRightCluster = 1:size(Correspondences,2)
        thisCorrespondence = Correspondences(thisLeftCluster,thisRightCluster);
        %see if this was assigned 
        if GreedyMaxes(thisLeftCluster) == thisRightCluster
            sumOfAssigned = sumOfAssigned + thisCorrespondence;
            numAssigned = numAssigned + 1;
        else
            sumOfUnassigned = sumOfUnassigned + thisCorrespondence;
            numUnassigned = numUnassigned + 1;
        end
    end
end

OverallStress = ((sumOfAssigned/numAssigned)/((sumOfAssigned/numAssigned)+(sumOfUnassigned/numUnassigned)));

