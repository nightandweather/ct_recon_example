clear all; close all;

xd = 128;

figure(1);
true_object = abs(phantom(xd));

subplot(2,3,1), imshow(true_object, [0 max(true_object(:))]);

azi_angles = [1:1:180];

meas_data = radon(true_object,azi_angles);

subplot(2,3,2), imshow(meas_data', [0 max(meas_data(:))]);

rec = ones(size(true_object)); % initial reconstruction image



sinogram_ones = ones(size(meas_data));
sens = iradon(sinogram_ones,azi_angles,'none',xd);

    subplot(2,3,4), imshow(rec, [0 max(rec(:))]);
   
pause
for it = 1:10
    fp = radon(rec,azi_angles);
        subplot(2,3,5), imshow(fp', [0 max(fp(:))]);
    ratio = meas_data ./ (fp + 0.0001);
    subplot(2,3,3), imshow(ratio', [0 max(ratio(:))]);
    bp_ratio = iradon(ratio,azi_angles,'none',xd);
    
        subplot(2,3,6), imshow(bp_ratio, [0 max(rec(:))]);
    rec = rec .* bp_ratio

    pause
    subplot(2,3,4), imshow(rec, [0 max(rec(:))]);


end    
