%% demo
clc;clear;close all;
 load Indian_pines_corrected.mat;
 load Indian_pines_gt.mat;
 
%��ǩͼ��
 figure;
 imshow(indian_pines_gt,[]);
 title('����ǩ')
 
 %һЩ����ͼ��
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
 
 %��ά��ʾ
 disp3DBandByBand(indian_pines_corrected,50,10);%��ʾ