function disp3DBandByBand(A,dispbands,gap)
% ʹ�ò����ķ�����ʾ��ά��ͼ��
% AΪ��ά�ĸ߹������ݣ���һ����0-1֮��
% dispbands�����ʾ���ٲ�ͼ��,�����sz������ʾsz�㣬�����szС������ʾdispbands��
% gapΪrgb�������ڲ������Ĳ����������ڵ���1��С��sz������Ϊ10

[ A,maxvalue,minvalue ]=hyImageToUintBandByBandIn( A );%ÿ���������ŵ�0-1֮��

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
    
    % ����������ɫ�ϳ�
    rgb(:,:,1) = flipud(adapthisteq(red));%flipudֻ�ǽ���һ����ֱ�ķ�ת����ƥ����imshow����ʾ��������
    rgb(:,:,2) = flipud(adapthisteq(green));
    rgb(:,:,3) = flipud(adapthisteq(blue));

    h=surf(x+sz-i,y+sz-i,ones(sx,sy)*1,rgb,'LineStyle','none');
% h=surf(x+sz-i,y+sz-i,ones(sx,sy)*200,reshape(A(:,:,sz-i+1),sx,sy),'LineStyle','none');
end
axis([1 (sx+sz) 1 (sy+sz)]);
axis off;

end

function [ HyImUint,maxvalue,minvalue ] = hyImageToUintBandByBandIn( hyIm )
%HYIMAGETOUINTBANDBYBAND �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

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