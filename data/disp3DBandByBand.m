function disp3DBandByBand(A,dispbands,gap)
% 使用层析的方法显示三维的图像
% A为三维的高光谱数据，归一化到0-1之间
% dispbands最多显示多少层图像,如果比sz大，则显示sz层，如果比sz小，则显示dispbands层
% gap为rgb三个相邻波段相差的波段数，大于等于1，小于sz，建议为10

[ A,maxvalue,minvalue ]=hyImageToUintBandByBandIn( A );%每个波段缩放到0-1之间

[sx,sy,sz]=size(A);

[x,y]=meshgrid(1:1:sx,1:1:sy);

if(sz>dispbands)
    sz=dispbands;
end

rgb = zeros(size(A, 1), size(A, 2), 3);

% gap=10;

figure;hold on;
for i=1:sz
    
    redband=mod(sz-i+1-gap,sz)+1;
    greenband=mod(sz-i+1,sz)+1;
    blueband=mod(sz-i+1+gap,sz)+1;
    red = A(:,:,redband);
    green=A(:,:,greenband);
    blue = A(:,:,blueband);
    
    % 进行三个颜色合成
    rgb(:,:,1) = flipud(adapthisteq(red));%flipud只是进行一个垂直的翻转，以匹配与imshow的显示方向问题
    rgb(:,:,2) = flipud(adapthisteq(green));
    rgb(:,:,3) = flipud(adapthisteq(blue));

    h=surf(x+sz-i,y+sz-i,ones(sx,sy)*1,rgb,'LineStyle','none');
% h=surf(x+sz-i,y+sz-i,ones(sx,sy)*200,reshape(A(:,:,sz-i+1),sx,sy),'LineStyle','none');
end
axis([1 (sx+sz) 1 (sy+sz)]);
axis off;

end

function [ HyImUint,maxvalue,minvalue ] = hyImageToUintBandByBandIn( hyIm )
%HYIMAGETOUINTBANDBYBAND 此处显示有关此函数的摘要
%   此处显示详细说明

HyImUint=hyIm;

[X,Y,B]=size(hyIm);

maxvalue=max(max(hyIm));
minvalue=min(min(hyIm));
maxvalue=reshape(maxvalue,1,B);
minvalue=reshape(minvalue,1,B);

for i=1:B
    HyImUint(:,:,i)=(hyIm(:,:,i)-minvalue(i))./(maxvalue(i)-minvalue(i));
end

end