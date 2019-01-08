function [img_org, fvec, kpts] = sift(img_file, sigma, S)
%% SIFT �㷨ʵ��
% function [fvec, kpts] = sift(img_file, sigma, S)
% sigma  ��ʼ�߶� ��1.6
% (S+2) Ϊÿһ��octave��ֵͼ�����  (S+3)Ϊÿһ��octave�ڸ�˹ģ��ͼ�����
% by Granvallen

% keypoints ��¼   octave, interval, r, c, scale, mainang

%% ͼ��Ԥ����
img_org = imread(img_file); % ԭʼͼ��
% figure
% imshow(img_org)

% ���ɻҶ�ͼ
if (size(img_org, 3) == 3 || size(img_org, 3) == 4) 
    img_org = img_org(:, :, 1 : 3);
    img = rgb2gray(img_org);
else
    img = img_org;
end
[r, c] = size(img);
 
% ��ֵ�Ŵ�ͼ������ ������˫���԰� �ȽϿ�
img = imresize(img, 2, 'bicubic'); % 'bicubic' ˫���β�ֵ �����Ĭ�ϲ��� 'bilinear'
% �Բ�ֵ���ͼ����и�˹�˲�
sigma_init = sqrt(sigma^2 - 0.5^2 * 4); % Ԥ�����˹�˲�sigma
img = imgaussfilt(img, sigma_init);%, 'FilterSize', (2*round(3*sigma_init) + 1) * [1, 1]); % ע�Ᵽ֤�˲��˵ı߳�Ϊ����

% ������ֵת��Ϊdouble����֮����
img = double(img);


%% ͼ���������DoG
disp('����������')
octave = round(log2(min(r, c)) - log2(8) + 1); % ��Ĭ�Ͻ��������ϲ�ͼ�񳤿��СֵΪ8
K = 2 ^ (1 / S); % kΪ�߶ȿռ���
img = imgaussfilt(img, sigma); %, 'FilterSize', (2*round(3*sigma) + 1) * [1, 1]); 

% ���������� �õ�������ÿ��ͼ�� �� ÿ��DoGͼ�� pyramid
pyr_DoG = cell(octave, 1);
pyr_G = cell(octave, 1);
img_i = img; % ��ʼ����������
for oct_i = 1 : octave
    pyr_G_i = zeros([size(img_i) S+3]); % Ԥ����ռ� ��Ϊĳһ��oct�����и�˹ģ��ͼ�� ����ά����
    pyr_G_i(:, :, 1) = img_i; % ��һ��ͼƬ����ģ������
    for interval = 2 : S + 3 % ��һ��octave�ڵ�ͼ������˲�����(���˵�һ��)
        % �ĳ�openCV��SIFT�ļ��㷽��
        sigma_prev = K^(interval - 2) * sigma;
        sigma_i = sqrt((K*sigma_prev)^2 - sigma_prev^2);
        img_i = imgaussfilt(img_i, sigma_i);%, 'FilterSize', (2*round(3*sigma_i) + 1) * [1, 1]); 
        pyr_G_i(:, :, interval) = img_i;
    end
    pyr_G{oct_i} = pyr_G_i;
    pyr_DoG{oct_i} = pyr_G_i(:, :, 1 : end - 1) - pyr_G_i(:, :, 2 : end); % ��ÿһ��oct��DoGͼ��
    img_i = imresize(pyr_G_i(:, :, end - 2), 0.5, 'bicubic'); % ������ ������һoctave�ĵ�һ�Ÿ�˹ģ��ͼ��
    % ��һ��octave�ĵ�һ�Ÿ�˹ģ��ͼ���sigmaΪ��һ��octave���������Ÿ�˹ģ��ͼ���sigma
end
clear img_i pyr_G_i sigma_i interval oct_i


%% Ѱ�Ҽ�ֵ��
disp('Ѱ�Ҽ�ֵ��')
extrema = []; % ��ż�ֵ������
num = 0;  % ��ֵ�ܸ���

% Ѱ�Ҽ�ֵ(����ֵ + ��Сֵ)
tic
for oct_i = 1 : size(pyr_DoG, 1)
    DoG = pyr_DoG{oct_i};
    [DoG_r, DoG_c, interval] = size(DoG);

    for DoG_r_i = 2 : DoG_r - 1
        for DoG_c_i = 2 : DoG_c - 1
            
            % ȡ����ʱҪ�Ƚϵ�����
            DoG_px = DoG(DoG_r_i - 1 : DoG_r_i + 1, DoG_c_i - 1 : DoG_c_i + 1, :);
            DoG_px = reshape(DoG_px, 9, interval);
            % ����DoG_pxΪ 9 x interval ���� ÿһ��Ϊһintervalȡ�������� �����ų�һ��

            % �Ƚ�����
            for interval_i = 2 : interval - 1
                px_current = DoG_px(:, interval_i);
                px = px_current(5);
                px_current(5) = [];
                if (abs(px) >= floor(0.5*0.04 / S * 255) && ...
                        ((px >= max(DoG_px(:, interval_i - 1)) && ...
                        px >= max(px_current) && ...
                        px >= max(DoG_px(:, interval_i + 1))) || ...
                        (px <= min(DoG_px(:, interval_i - 1)) && ...
                        px <= min(px_current) && ...
                        px <= min(DoG_px(:, interval_i + 1))) ))
                    num = num + 1;
                    sigma_i = K^(interval_i - 1) * sigma; % ����ü�ֵ��߶�sigma
                    extrema(num, :) = [oct_i, interval_i, DoG_r_i, DoG_c_i, sigma_i, 0]; % ���һλ��Ԥ���������Ƕ�
                end
            end

        end
    end
end
toc
clear sigma_i interval_i DoG DoG_r DoG_c interval DoG_px px_current px oct_i


%% ��ֵ��ɸѡ  keypoints
disp('��ֵ��ɸѡ')
keypoints = [];
d1 = 0;
d2 = 0;
d3 = 0;
tic
for kpt_i = 1 : num
%     kpt_i
    % ��ȡ��ǰ��ֵ����Ϣ
    kpt = num2cell(extrema(kpt_i, :));
    [oct, interval, kpt_r, kpt_c, ~, ~] = deal(kpt{:});

    % ɸѡ���� 1 ȥ���뾫ȷ��ֵƫ�ƽϴ�ĵ�
	% ������ֵ����ֵ��ƫ���Ƿ�����Ҫ��
    isdrop = true; % �Ƿ�ɾ����ʶ
    for try_i = 1 : 5 % 5�β�ֵ�ƽ���ʵ��ֵ(����֤)
        
        % ȡ�����DoGͼ��  ע�� interval ���ܻ�仯 ����ÿ�ζ�Ҫ����ȡDoGͼ��
        DoG = pyr_DoG{oct}(:, :, interval) ./ 255;
        DoG_prev = pyr_DoG{oct}(:, :, interval - 1) ./ 255;
        DoG_next = pyr_DoG{oct}(:, :, interval + 1) ./ 255;
        
        % ����һ�׵� 
        dD = [DoG(kpt_r, kpt_c + 1) - DoG(kpt_r, kpt_c - 1);
                   DoG(kpt_r + 1, kpt_c) - DoG(kpt_r - 1, kpt_c);
                   DoG_prev(kpt_r, kpt_c) - DoG_next(kpt_r, kpt_c)] ./ 2;
        % ������׵�(Hessian����)
        dxx = (DoG(kpt_r, kpt_c + 1) + DoG(kpt_r, kpt_c - 1) - 2*DoG(kpt_r, kpt_c)) / 1; % ע���ĸΪ1
        dyy = (DoG(kpt_r + 1, kpt_c) + DoG(kpt_r - 1, kpt_c) - 2*DoG(kpt_r, kpt_c)) / 1;
        dss = (DoG_next(kpt_r, kpt_c) + DoG_prev(kpt_r, kpt_c) - 2*DoG(kpt_r, kpt_c)) / 1;
        % ��ϵ� ע���ĸΪ4
        dxy = (DoG(kpt_r + 1, kpt_c + 1) - DoG(kpt_r + 1, kpt_c - 1) ...
            - DoG(kpt_r - 1, kpt_c + 1) + DoG(kpt_r - 1, kpt_c - 1)) / 4;
        dxs = (DoG_next(kpt_r, kpt_c + 1) - DoG_next(kpt_r, kpt_c - 1) ...
            - DoG_prev(kpt_r, kpt_c + 1) + DoG_prev(kpt_r, kpt_c - 1)) / 4;
        dys = (DoG_next(kpt_r + 1, kpt_c) - DoG_next(kpt_r - 1, kpt_c) ...
            - DoG_prev(kpt_r + 1, kpt_c) + DoG_prev(kpt_r - 1, kpt_c)) / 4;
        % �ɶ��׵��ϳ�Hessian����
        H = [dxx, dxy, dxs;
                 dxy, dyy, dys;
                 dxs, dys, dss];
        
        % ���ƫ��ֵ
        % ���Hessian���󲻿��� ȥ����ֵ��
        if (abs(dxx * dyy - dxy * dxy) < 10^-16)
            break; 
        end
        x_hat = -H \ dD;
        
        
        
        % ������ֵ������ �Ա������һ�β�ֵ����
        kpt_c = kpt_c + round(x_hat(1));
        kpt_r = kpt_r + round(x_hat(2)); 
        interval = interval + round(x_hat(3));
        
         
        % �ж�ƫ�Ƴ̶�  ע��sigma��ƫ��ҲҪ����  ����ֱ��ͨ��ɸѡ
        if (all(abs(x_hat) < 0.5))  % ��ֵΪ 0.5
            isdrop = false;
            break;
        end
       
        % �ж��µ�����������Ƿ񳬹��߽�   ��������  �����ٴ���������ƫ��
        if (kpt_r <= 1 || kpt_r >= size(DoG, 1) || ...
                kpt_c <= 1 || kpt_c >= size(DoG, 2) || ...
                interval < 2 || interval > S + 1) % interval ����ȡ��һ�������һ��
            break; 
        end

    end
    
    % ����õ���ɾ�� ������һ����ֵ��
    if (isdrop)
        d1 = d1 +1;
        continue;
    end
   
    % ɸѡ���� 2 ȥ����Ӧ��С�ļ�ֵ�� ��ֵԽ��Ź��ĵ�Խ��
    D_hat = DoG(kpt_r, kpt_c) + dD' * x_hat / 2;
   	if (abs(D_hat) * S < 0.04) % ��opencv��SIFTʵ������ "* s"  ��ԭ����???
        d2 = d2 + 1;
        continue;
    end
    
    % ɸѡ���� 3 ȥ����Ե�ؼ���
    % ����Hessian����ļ�������ʽ
    trH = dxx + dyy;
    detH = dxx * dyy - dxy * dxy;
    edgeThreshold = 10; % edgeThreshold ԽСɸѡԽ�ϸ� ������
    if (detH <= 0 || trH^2 * edgeThreshold >= (edgeThreshold + 1)^2 * detH)
        d3 = d3 + 1;
        continue;
    end
    
    % ������ ��ֵ����Ա�����
    sigma_temp = K^(interval - 1) * sigma; % ����ü�ֵ��߶�sigma
    kpt_temp = [oct, interval, kpt_r, kpt_c, sigma_temp, 0];
    keypoints = [keypoints; kpt_temp];
end
toc
num = size(keypoints, 1);


%% �������������� kpts
disp('��������������')
kpts = []; 
tic
for kpt_i = 1 : num

    % ȡ����������Ϣ
    kpt = keypoints(kpt_i, :);
    oct_i = kpt(1); 
    interval = kpt(2);
    kpt_r = kpt(3);
    kpt_c = kpt(4);
    scale = kpt(5);
     
    radius = round(3 * 1.5 * scale); % ���������������뾶
    pyr_G_i = pyr_G{oct_i}(:, :, interval);
    % ������������������� (radius*2+1)*(radius*2+1)
    img_r = 0; img_c = 0; % ���������صľ������� (img_r, img_c)
    histtemp = zeros(1, 36 + 4); % ���߸������2����λ���ڲ�ֵ����      1   2    3 ~ 38    39  40
    
    for i = -radius : radius % ��������  ����
        img_c = kpt_c + i;
        if (img_c <= 1 || img_c >= size(pyr_G_i, 2)) % ���곬��ͼ�� �Ҳ�����ͼ�����Ե����
            continue;
        end
        for j = -radius : radius % ���ϵ���  ����
            img_r = kpt_r + j;
            if (img_r <= 1 || img_r >= size(pyr_G_i, 1))
                continue;
            end
            
            % �����ݶ�
            dx = pyr_G_i(img_r, img_c + 1) - pyr_G_i(img_r, img_c - 1);
            dy = pyr_G_i(img_r - 1, img_c) - pyr_G_i(img_r + 1, img_c);
            % �����ݶ� ��ֵ �� ����
            mag = sqrt(dx^2 + dy^2);
            % ����atan2������ ������Ķ���Ϊ -pi ~ pi ����Ҫ����һ��ת��
            if (dx >= 0 && dy >= 0 || dx <= 0 && dy >= 0)
                ang = atan2(dy, dx);
            else
                ang = atan2(dy, dx) + 2*pi;
            end
            ang = ang * 360/(2 * pi); % ת��Ϊ�Ƕ���
            
            % ���������ֱ��ͼ����
            bin = round(36 / 360 * ang);
            if( bin >= 36 )
                bin = bin - 36;
            elseif( bin < 0 ) 
                bin = bin + 36;
            end
            
            % �����˹��Ȩ�� 
            w_G = exp(- (i^2 + j^2) / (2 * (1.5 * scale)^2));
            
            % ����histtemp   1   2    3 ~ 38    39  40
            histtemp(bin + 3) = histtemp(bin + 3) + mag * w_G;
            
        end
    end
 
    % ��ֱ��ͼƽ������ ���histtemp�Ŀ�λ  1   2    3 ~ 38    39  40
    hist = zeros(1, 36); % ���һ���������ֱ��ͼ
    histtemp(1 : 2) = histtemp(37 : 38);
    histtemp(39 : 40) = histtemp(3 : 4);
    for k = 3 : 38   % ��Ȩ�ƶ�ƽ��  ����Ϊ5
        hist(k - 2) = (histtemp(k - 2) + histtemp(k + 2)) * (1/16) + ...
            (histtemp(k - 1) + histtemp(k + 1)) * (4/16) + ...
            histtemp(k) * (6/16);
    end 
    
    % ����ֱ��ͼ �� ������ �� ������
    hist_threshold = 0.8 * max(hist); % ���������ֵ

    for k = 1 : 36
        
        if (k == 1) % klΪ��k�����һ�������� ��ͬ
            kl = 36;
        else
            kl = k - 1;
        end
        
        if (k == 36)
            kr = 1;
        else
            kr = k + 1;
        end
        
        % ���ﲻ��Ҫ��ֱ��ͼ�����߳�����ֵ ��Ҫ����������ڵ�������
        if (hist(k) > hist(kl) && hist(k) > hist(kr) && hist(k) >= hist_threshold)
            % ͨ�� �����߲�ֵ ���㾫ȷ���ǹ�ʽ  ��ȷֵ��Χ  bin 0 ~ 35...  |36
            bin = k + 0.5 * (hist(kl) - hist(kr)) / (hist(kl) - 2*hist(k) + hist(kr));
            if (bin < 0) % binԽ�紦��
                bin = bin + 36;
            elseif (bin >= 36)
                bin = bin - 36;
            end
            % ���㾫ȷ����
            ang =  bin * (360/36); % angΪ�Ƕ���
            if (abs(ang - 360) < 10^-16)
                ang = 0;
            end
            
            % ����keypoints
            kpt_temp = [kpt(1 : 5), ang];
            kpts = [kpts; kpt_temp];
        end
        
    end

end
toc


%% ������������
disp('����������')
d = 4; % ��������Ŀ ���� 4 * 4 * 8 �е�4
n = 8; % �ݶ�ֱ��ͼ����Ŀ 8����������
fvec = [];
tic
% �ٴα���������  ÿ�����������128ά��������
for kpt_i = 1 : size(kpts, 1)
    kpt = kpts(kpt_i, :);
    oct_i = kpt(1);
    interval = kpt(2);
    kpt_r = kpt(3);
    kpt_c = kpt(4);
    scale = kpt(5);
    mainang = kpt(6); % ������
    % ȡ����Ӧ�߶ȵĸ�˹ģ��ͼ��
    pyr_G_i = pyr_G{oct_i}(:, :, interval);
    [pyr_G_i_r, pyr_G_i_c] = size(pyr_G_i);
    % ������ת�������
    cos_t = cos(mainang * pi / 180); % cos���ܵ��ǻ�����
    sin_t = sin(mainang * pi / 180);
    
    hist_width = 3 * scale; % ÿ��������߳�  ����
    radius = round(hist_width * sqrt(2) * (d + 1) * 0.5); % �������в��������������뾶   Բ�� 
    % �ж��¼�����İ뾶��ͼ��Խ��߳� ���뾶ȡС��
    % ���뾶��ͼ��Խ���һ����ʱ��һ������ �ٴ󲻹�Ҳ�Ǳ���ͼ���������� ����û������
    radius = min(radius, floor(sqrt(pyr_G_i_r^2 + pyr_G_i_c^2))); 
    % ����ת����������������߶Ƚ��й�һ��  �Ա���֮����������ת�������(r_rot, c_rot)������������߶ȵ�
    cos_t = cos_t / hist_width;
    sin_t = sin_t / hist_width;
    
    % ��ʼ���洢ֱ��ͼ���� ����һ����ά���� (d+2)*(d+2)*(n+2)
    % ����ά�ȷֱ��� �������������� �������������� ֱ��ͼ������
    % ����ά�ȶ��ֱ������2����λ ��֮��Ĳ�ֵʱ�ᳬ�� 4*4*8�ķ�Χ
    hist = zeros(d + 2, d + 2, n + 2);
    
    % ���ɱ����������������ص� ����ֱ��ͼ
    for i = -radius : radius  % ���ϵ���  ����
        for j = -radius : radius % ��������ɨ��  ����
            
            % ������ת���������� (ע��������������߶ȵ�)
            c_rot = j * cos_t - i * sin_t;
            r_rot = j * sin_t + i * cos_t;
            % ������������������������ (rbin, cbin)
            % |_*_|_*_|_*_|_*_|
            % |_*_|_*_|_*_|_*_|
            % |_*_|_*_|_*_|_*_|
            % |_*_|_*_|_*_|_*_|
            rbin = r_rot + d/2 - 0.5; %  ��������������߶��µľ������� ������0.5��ƽ��
            cbin = c_rot + d/2 - 0.5; % ȡֵ��Χ  -1 ~ 3... |4
            % 0.5��ƽ��ʹ�������������ύ�㶼�������������Ͻ� ����֮���ֵ ����һ�����ص����Χ�ĸ��������ֱ��ͼ����
            % ��������ͼ���������(img_r, img_c)
            img_r = kpt_r + i;
            img_c = kpt_c + j;
            
            if (-1 < rbin && rbin < d && -1 < cbin && cbin < d && ...
                   1 < img_r && img_r < pyr_G_i_r && 1 < img_c && img_c < pyr_G_i_c)
               
                % �����ݶ� ע�� dy
                dx = pyr_G_i(img_r, img_c + 1) - pyr_G_i(img_r, img_c - 1);
                dy = pyr_G_i(img_r - 1, img_c) - pyr_G_i(img_r + 1, img_c);
                
                % �����ݶ� ��ֵ �� ����
                mag = sqrt(dx^2 + dy^2);
                % ����atan2������ ������Ķ���Ϊ -pi ~ pi ����Ҫ����һ��ת��
                if (dx >= 0 && dy >= 0 || dx <= 0 && dy >= 0)
                    ang = atan2(dy, dx);
                else
                    ang = atan2(dy, dx) + 2*pi;
                end
                ang = ang * 360/(2 * pi); % ת��Ϊ�Ƕ���
                
                % �жϷ��������ĸ�ֱ��ͼ������
                obin = (ang - mainang) * (n / 360); % ȡֵ 0 ~ 7...  |8
                % �����˹��Ȩ��ķ�ֵ          
                w_G = exp(- (r_rot^2 + c_rot^2) / (0.5 * d^2)); % �����˹��Ȩ��   -1/((d/2)^2 * 2)   ->    -1/(d^2 * 0.5)
                mag = mag * w_G;
                  
                r0 = floor(rbin); % -1 0 1 2 3
                c0 = floor(cbin);
                o0 = floor(obin);
              
                % �൱���������ڵ�����
                rbin = rbin - r0;
                cbin = cbin - c0;
                obin = obin - o0;
                
                % ����o0Խ��ѭ��
                if (o0 < 0)
                    o0 = o0 + n;
                elseif (o0 >= n)
                    o0 = o0 - n;
                end
                 
                % �����Բ�ֵ ���hist������ �����ص��ֵ���׵���Χ�ĸ��������ֱ��ͼ��ȥ
                % ����8������ֵ    0 < rbin cbin obin < 1
                % �ȼ��㹱��Ȩֵ 
                v_rco000 = rbin * cbin * obin;
                v_rco001 = rbin * cbin * (1 - obin);
                
                v_rco010 = rbin * (1 - cbin) * obin;
                v_rco011 = rbin * (1 - cbin) * (1 - obin);
                
                v_rco100 = (1 - rbin) * cbin * obin;
                v_rco101 = (1 - rbin) * cbin * (1 - obin);                
                
                v_rco110 = (1 - rbin) * (1 - cbin) * obin;
                v_rco111 = (1 - rbin) * (1 - cbin) * (1 - obin);

                % rbin     1 ~ 6          r0  -1 ~ 3
                % cbin     1 ~ 6          c0  -1 ~ 3
                % obin    1 ~ 8    |   9 10   ��ֵ�ᵽ9     o0   0 ~ 7
                temp = zeros(2, 2, 2);
                temp(:, :, 1) = [v_rco000, v_rco010; v_rco100, v_rco110];
                temp(:, :, 2) = [v_rco001, v_rco011; v_rco101, v_rco111];
                hist(r0+2 : r0+3, c0+2 : c0+3, o0+1 : o0+2) = hist(r0+2 : r0+3, c0+2 : c0+3, o0+1 : o0+2) + mag .* temp;
            end 
            
        end
    end
    
    % ����������  ��histֱ��ͼ�е�����������
    fvec_i = [];
    for i = 1 : d   % �м� 4*4*8 �Ĳ���
        for j = 1 : d
            hist(i + 1, j + 1, 1) = hist(i + 1, j + 1, 1) + hist(i + 1, j + 1, 9);
            hist(i + 1, j + 1, 2) = hist(i + 1, j + 1, 2) + hist(i + 1, j + 1, 10);
            fvec_i = [fvec_i; reshape(hist(i + 1, j + 1, 1 : 8), 8, 1)];
        end
    end
    
    % ������������ֵ��ķ������д���  ���Ʒ����Թ��ոı�Ӱ��
    fvec_norm = norm(fvec_i, 2); % ��������ģ��
    fvec_threshold = 0.2 * fvec_norm;
    fvec_i(fvec_i > fvec_threshold) = fvec_threshold; % ������ֵ�ķ�������Ϊ��ֵ
    
    % ��һ����������
    fvec_norm = norm(fvec_i, 2);
    fvec_i = fvec_i ./ fvec_norm;
    fvec = [fvec, fvec_i];
    
end
toc



%% ��DoGͼ��
% close all
% oct_i = 4;
% for i = 1 : s+2
%     figure
%     imshow(pyr_DoG{oct_i}(:, :, i)*25)
% end

%% ��ʾ���йؼ���
% close all
% figure
% imshow(imresize(img_org, 2, 'bicubic'))
% hold on
% for i = 1 : size(keypoints, 1)
%     plot(keypoints(i, 4), keypoints(i, 3), 'r.', 'MarkerSize', 15);
% end
% hold off

%% ��ʾɸѡǰ��ֵ��
% for o = 1 : octave
%     figure
%     subplot(1,2,1); imshow(uint8(img))
%     title('ɸѡǰ')
%     hold on
%     for i = 1 : size(extrema, 1)
%         if (extrema(i, 1) == o)
%             plot(extrema(i, 4), extrema(i, 3), 'r.', 'MarkerSize', 15);
%         end
%     end
%     hold off
%     
%     subplot(1,2,2); imshow(uint8(img))
%     title('ɸѡ��')
%     hold on
%     for i = 1 : size(keypoints, 1)
%         if (keypoints(i, 1) == o)
%             plot(keypoints(i, 4), keypoints(i, 3), 'r.', 'MarkerSize', 15);
%         end
%     end
%     hold off
% end

%% ��ʾ��������ؼ���

% ת������߶�
for kpt_i = 1 : size(kpts, 1)
    kpts(kpt_i, 3 : 4) = kpts(kpt_i, 3 : 4) .* 2^(kpts(kpt_i, 1) - 2);  
end

figure
imshow(img_org)
hold on
for kpt_i = 1 : size(kpts, 1)
    kpt = kpts(kpt_i, :);
    oct_i = kpt(1);
    kpt_r = kpt(3) * 2^(oct_i - 1);
    kpt_c = kpt(4) * 2^(oct_i - 1);
    scale = kpt(5);
    mainang = kpt(6);
    
    radius = scale*2^(oct_i - 1);
    color = [rand(), rand(), rand()];
    rectangle('position', [kpt_c - radius, kpt_r - radius, radius*2, radius*2], 'curvature', [1, 1], ...
        'EdgeColor', color, 'LineWidth', 1);
    
    kpt_c2 = kpt_c + radius  * cos(mainang * pi / 180);
    kpt_r2 = kpt_r - radius * sin(mainang * pi / 180);
    plot([kpt_c, kpt_c2], [kpt_r, kpt_r2], 'Color', color, 'LineWidth', 1);
%     plot(kpt_c, kpt_r, 'r.', 'MarkerSize', 20, 'LineWidth', 1);
end
hold off




end





