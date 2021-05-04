

function [bigImage, latentVerticalShift, ExemplarOffset, ExemplarVerticalShift] =loadImageAndComputeOffsets(ImagePairNameFromFile)

%split code based on mac or pc
tf = ispc;
nist_str_left = [ImagePairNameFromFile '_Left.png'];
nist_str_right = [ImagePairNameFromFile '_Right.png'];
LeftImageName = nist_str_left;
%now check to see if is DT
if ~tf
    
    %show the image, plot the fixations over the image
    if exist(['../../WBeyeDataset105/' nist_str_left], 'file')
        leftImage = imread(['../../WBeyeDataset105/' nist_str_left]);
        
    else
        leftImage = imread(['../../ABC18/' nist_str_left]);
    end
    if ndims(leftImage)<3
        leftImage(:,:,1) = leftImage(:,:);
        leftImage(:,:,2) = leftImage(:,:,1);
        leftImage(:,:,3) = leftImage(:,:,1);
    end
    
    %show the image, plot the fixations over the image
    if exist(['../../WBeyeDataset105/' nist_str_right], 'file')
        rightImage = imread(['../../WBeyeDataset105/' nist_str_right]);
    else
        rightImage = imread(['../../ABC18/' nist_str_right]);
    end
    
    if ndims(rightImage)<3
        rightImage(:,:,1) = rightImage(:,:);
        rightImage(:,:,2) = rightImage(:,:,1);
        rightImage(:,:,3) = rightImage(:,:,1);
    end
else
    
    
    %show the image, plot the fixations over the image
    if exist(['..\..\WBeyeDataset105\' nist_str_left], 'file')
        leftImage = imread(['..\..\WBeyeDataset105\' nist_str_left]);
        
    else
        leftImage = imread(['..\..\ABC18\' nist_str_left]);
    end
    if ndims(leftImage)<3
        leftImage(:,:,1) = leftImage(:,:);
        leftImage(:,:,2) = leftImage(:,:,1);
        leftImage(:,:,3) = leftImage(:,:,1);
    end
    
    %show the image, plot the fixations over the image
    if exist(['..\..\WBeyeDataset105\' nist_str_right], 'file')
        rightImage = imread(['..\..\WBeyeDataset105\' nist_str_right]);
    else
        rightImage = imread(['..\..\ABC18\' nist_str_right]);
    end
    
    if ndims(rightImage)<3
        rightImage(:,:,1) = rightImage(:,:);
        rightImage(:,:,2) = rightImage(:,:,1);
        rightImage(:,:,3) = rightImage(:,:,1);
    end
    
    
    
end

%now merge the two images
mostRows = max(size(leftImage,1), size(rightImage,1));
bigImage = uint8(ones(mostRows, size(leftImage,2)+size(rightImage,2), 3)*255);

%center both images
latentVerticalShift = floor((mostRows-size(leftImage,1))/2);
ExemplarVerticalShift = floor((mostRows-size(rightImage,1))/2);


bigImage(1+ latentVerticalShift:size(leftImage,1)+latentVerticalShift, 1:size(leftImage,2),:) = leftImage;


bigImage(1+ExemplarVerticalShift:size(rightImage,1)+ExemplarVerticalShift, size(leftImage,2) + 1 :size(leftImage,2)+size(rightImage,2),:) = rightImage;

ExemplarOffset = size(leftImage,2);

