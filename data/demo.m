%% demo
clc;clear;close all;
 load Indian_pines_corrected.mat;
 load Indian_pines_gt.mat;
 
%标签图像
 figure;
 imshow(indian_pines_gt,[]);
 title('类别标签')
 
 %一些波段图像
 figure;
 subplot(221)
 imshow(indian_pines_corrected(:,:,2),[]);
 title('band 2')
  subplot(222)
 imshow(indian_pines_corrected(:,:,20),[]);
 title('band 20')
  subplot(223)
 imshow(indian_pines_corrected(:,:,100),[]);
 title('band 100')
  subplot(224)
 imshow(indian_pines_corrected(:,:,150),[]);
 title('band 150')
 
 %三维显示
 disp3DBandByBand(indian_pines_corrected,50,10);%显示