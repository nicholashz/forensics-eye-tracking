%for every fixation on the latent, return the index and the
%value into the examplar list

function   [GreedyMaxes, GreedyMaxValues, externalSimValues] = getGreedyMaxes(Correspondences, externalSim)
[NumLatents, NumExemplars] = size(Correspondences);

GreedyMaxes = nan(1,NumLatents);
GreedyMaxValues = nan(1,NumLatents);
externalSimValues = nan(1,NumLatents);

while max(Correspondences(:))>0
    
    indexOfGlobalMax = find(max(Correspondences(:))==Correspondences);
    %check for multiples here
    if size(indexOfGlobalMax,1)>1
        indexOfGlobalMax = indexOfGlobalMax(1);
    end
    
    if size(indexOfGlobalMax,2)>1
        indexOfGlobalMax = indexOfGlobalMax(1);
    end
    
    [rowOfMax, colOfMax] = ind2sub(size(Correspondences),indexOfGlobalMax);
    
    GreedyMaxes(1, rowOfMax) = colOfMax;
    
    GreedyMaxValues(1, rowOfMax) = Correspondences(rowOfMax, colOfMax);
    if exist('externalSim', 'var')
       externalSimValues(1,rowOfMax) = externalSim(rowOfMax, colOfMax); 
    end
    
    %now zero out those rows and cols
    Correspondences(rowOfMax,:) = 0;
    Correspondences(:, colOfMax) = 0;
    
    
end