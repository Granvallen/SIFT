function [img_org, fvec, kpts] = sift(img_file, sigma, S)
%% SIFT 算法实现
% function [fvec, kpts] = sift(img_file, sigma, S)
% sigma  初始尺度 如1.6
% (S+2) 为每一个octave差值图像个数  (S+3)为每一个octave内高斯模糊图像个数
% by Granvallen

% keypoints 记录   octave, interval, r, c, scale, mainang

%% 图像预处理
img_org = imread(img_file); % 原始图像
% figure
% imshow(img_org)

% 生成灰度图
if (size(img_org, 3) == 3 || size(img_org, 3) == 4) 
    img_org = img_org(:, :, 1 : 3);
    img = rgb2gray(img_org);
else
    img = img_org;
end
[r, c] = size(img);
 
% 插值放大图像两倍 还是用双线性吧 比较快
img = imresize(img, 2, 'bicubic'); % 'bicubic' 双三次插值 这个是默认参数 'bilinear'
% 对插值后的图像进行高斯滤波
sigma_init = sqrt(sigma^2 - 0.5^2 * 4); % 预处理高斯滤波sigma
img = imgaussfilt(img, sigma_init);%, 'FilterSize', (2*round(3*sigma_init) + 1) * [1, 1]); % 注意保证滤波核的边长为奇数

% 将像素值转化为double便于之后处理
img = double(img);


%% 图像金子塔与DoG
disp('建立金字塔')
octave = round(log2(min(r, c)) - log2(8) + 1); % 即默认金子塔最上层图像长宽较小值为8
K = 2 ^ (1 / S); % k为尺度空间间隔
img = imgaussfilt(img, sigma); %, 'FilterSize', (2*round(3*sigma) + 1) * [1, 1]); 

% 建立金字塔 得到金子塔每层图像 及 每层DoG图像 pyramid
pyr_DoG = cell(octave, 1);
pyr_G = cell(octave, 1);
img_i = img; % 初始化迭代变量
for oct_i = 1 : octave
    pyr_G_i = zeros([size(img_i) S+3]); % 预分配空间 此为某一个oct中所有高斯模糊图像 是三维数组
    pyr_G_i(:, :, 1) = img_i; % 第一张图片不用模糊处理
    for interval = 2 : S + 3 % 对一个octave内的图像进行滤波处理(除了第一张)
        % 改成openCV中SIFT的计算方法
        sigma_prev = K^(interval - 2) * sigma;
        sigma_i = sqrt((K*sigma_prev)^2 - sigma_prev^2);
        img_i = imgaussfilt(img_i, sigma_i);%, 'FilterSize', (2*round(3*sigma_i) + 1) * [1, 1]); 
        pyr_G_i(:, :, interval) = img_i;
    end
    pyr_G{oct_i} = pyr_G_i;
    pyr_DoG{oct_i} = pyr_G_i(:, :, 1 : end - 1) - pyr_G_i(:, :, 2 : end); % 求每一个oct的DoG图像
    img_i = imresize(pyr_G_i(:, :, end - 2), 0.5, 'bicubic'); % 降采样 生成下一octave的第一张高斯模糊图像
    % 下一个octave的第一张高斯模糊图像的sigma为上一个octave倒数第三张高斯模糊图像的sigma
end
clear img_i pyr_G_i sigma_i interval oct_i


%% 寻找极值点
disp('寻找极值点')
extrema = []; % 存放极值点容器
num = 0;  % 极值总个数

% 寻找极值(极大值 + 极小值)
tic
for oct_i = 1 : size(pyr_DoG, 1)
    DoG = pyr_DoG{oct_i};
    [DoG_r, DoG_c, interval] = size(DoG);

    for DoG_r_i = 2 : DoG_r - 1
        for DoG_c_i = 2 : DoG_c - 1
            
            % 取出此时要比较的像素
            DoG_px = DoG(DoG_r_i - 1 : DoG_r_i + 1, DoG_c_i - 1 : DoG_c_i + 1, :);
            DoG_px = reshape(DoG_px, 9, interval);
            % 这里DoG_px为 9 x interval 数组 每一列为一interval取出的像素 按列排成一列

            % 比较像素
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
                    sigma_i = K^(interval_i - 1) * sigma; % 计算该极值点尺度sigma
                    extrema(num, :) = [oct_i, interval_i, DoG_r_i, DoG_c_i, sigma_i, 0]; % 最后一位是预存的主方向角度
                end
            end

        end
    end
end
toc
clear sigma_i interval_i DoG DoG_r DoG_c interval DoG_px px_current px oct_i


%% 极值点筛选  keypoints
disp('极值点筛选')
keypoints = [];
d1 = 0;
d2 = 0;
d3 = 0;
tic
for kpt_i = 1 : num
%     kpt_i
    % 获取当前极值点信息
    kpt = num2cell(extrema(kpt_i, :));
    [oct, interval, kpt_r, kpt_c, ~, ~] = deal(kpt{:});

    % 筛选步骤 1 去除与精确极值偏移较大的点
	% 经过插值看极值点偏差是否满足要求
    isdrop = true; % 是否删除标识
    for try_i = 1 : 5 % 5次插值逼近真实极值(待考证)
        
        % 取出相关DoG图像  注意 interval 可能会变化 所以每次都要重新取DoG图像
        DoG = pyr_DoG{oct}(:, :, interval) ./ 255;
        DoG_prev = pyr_DoG{oct}(:, :, interval - 1) ./ 255;
        DoG_next = pyr_DoG{oct}(:, :, interval + 1) ./ 255;
        
        % 计算一阶导 
        dD = [DoG(kpt_r, kpt_c + 1) - DoG(kpt_r, kpt_c - 1);
                   DoG(kpt_r + 1, kpt_c) - DoG(kpt_r - 1, kpt_c);
                   DoG_prev(kpt_r, kpt_c) - DoG_next(kpt_r, kpt_c)] ./ 2;
        % 计算二阶导(Hessian矩阵)
        dxx = (DoG(kpt_r, kpt_c + 1) + DoG(kpt_r, kpt_c - 1) - 2*DoG(kpt_r, kpt_c)) / 1; % 注意分母为1
        dyy = (DoG(kpt_r + 1, kpt_c) + DoG(kpt_r - 1, kpt_c) - 2*DoG(kpt_r, kpt_c)) / 1;
        dss = (DoG_next(kpt_r, kpt_c) + DoG_prev(kpt_r, kpt_c) - 2*DoG(kpt_r, kpt_c)) / 1;
        % 混合导 注意分母为4
        dxy = (DoG(kpt_r + 1, kpt_c + 1) - DoG(kpt_r + 1, kpt_c - 1) ...
            - DoG(kpt_r - 1, kpt_c + 1) + DoG(kpt_r - 1, kpt_c - 1)) / 4;
        dxs = (DoG_next(kpt_r, kpt_c + 1) - DoG_next(kpt_r, kpt_c - 1) ...
            - DoG_prev(kpt_r, kpt_c + 1) + DoG_prev(kpt_r, kpt_c - 1)) / 4;
        dys = (DoG_next(kpt_r + 1, kpt_c) - DoG_next(kpt_r - 1, kpt_c) ...
            - DoG_prev(kpt_r + 1, kpt_c) + DoG_prev(kpt_r - 1, kpt_c)) / 4;
        % 由二阶导合成Hessian矩阵
        H = [dxx, dxy, dxs;
                 dxy, dyy, dys;
                 dxs, dys, dss];
        
        % 求解偏差值
        % 如果Hessian矩阵不可逆 去除极值点
        if (abs(dxx * dyy - dxy * dxy) < 10^-16)
            break; 
        end
        x_hat = -H \ dD;
        
        
        
        % 调整极值点坐标 以便进行下一次插值计算
        kpt_c = kpt_c + round(x_hat(1));
        kpt_r = kpt_r + round(x_hat(2)); 
        interval = interval + round(x_hat(3));
        
         
        % 判断偏移程度  注意sigma的偏移也要考虑  满足直接通过筛选
        if (all(abs(x_hat) < 0.5))  % 阈值为 0.5
            isdrop = false;
            break;
        end
       
        % 判断下调整后的坐标是否超过边界   超过跳出  否则再次求调整后的偏差
        if (kpt_r <= 1 || kpt_r >= size(DoG, 1) || ...
                kpt_c <= 1 || kpt_c >= size(DoG, 2) || ...
                interval < 2 || interval > S + 1) % interval 不能取第一张与最后一张
            break; 
        end

    end
    
    % 如果该点已删除 处理下一个极值点
    if (isdrop)
        d1 = d1 +1;
        continue;
    end
   
    % 筛选步骤 2 去除响应过小的极值点 阈值越大放过的点越多
    D_hat = DoG(kpt_r, kpt_c) + dD' * x_hat / 2;
   	if (abs(D_hat) * S < 0.04) % 在opencv的SIFT实现里有 "* s"  但原因不明???
        d2 = d2 + 1;
        continue;
    end
    
    % 筛选步骤 3 去除边缘关键点
    % 计算Hessian矩阵的迹与行列式
    trH = dxx + dyy;
    detH = dxx * dyy - dxy * dxy;
    edgeThreshold = 10; % edgeThreshold 越小筛选越严格 不敏感
    if (detH <= 0 || trH^2 * edgeThreshold >= (edgeThreshold + 1)^2 * detH)
        d3 = d3 + 1;
        continue;
    end
    
    % 到这里 极值点可以保留了
    sigma_temp = K^(interval - 1) * sigma; % 计算该极值点尺度sigma
    kpt_temp = [oct, interval, kpt_r, kpt_c, sigma_temp, 0];
    keypoints = [keypoints; kpt_temp];
end
toc
num = size(keypoints, 1);


%% 求特征点主方向 kpts
disp('求特征点主方向')
kpts = []; 
tic
for kpt_i = 1 : num

    % 取出特征点信息
    kpt = keypoints(kpt_i, :);
    oct_i = kpt(1); 
    interval = kpt(2);
    kpt_r = kpt(3);
    kpt_c = kpt(4);
    scale = kpt(5);
     
    radius = round(3 * 1.5 * scale); % 参与计算像素区域半径
    pyr_G_i = pyr_G{oct_i}(:, :, interval);
    % 遍历参与计算像素区域 (radius*2+1)*(radius*2+1)
    img_r = 0; img_c = 0; % 遍历到像素的绝对坐标 (img_r, img_c)
    histtemp = zeros(1, 36 + 4); % 两边各多出的2个空位便于插值运算      1   2    3 ~ 38    39  40
    
    for i = -radius : radius % 从左往右  列数
        img_c = kpt_c + i;
        if (img_c <= 1 || img_c >= size(pyr_G_i, 2)) % 坐标超出图像 且不能是图像最边缘像素
            continue;
        end
        for j = -radius : radius % 从上到下  行数
            img_r = kpt_r + j;
            if (img_r <= 1 || img_r >= size(pyr_G_i, 1))
                continue;
            end
            
            % 计算梯度
            dx = pyr_G_i(img_r, img_c + 1) - pyr_G_i(img_r, img_c - 1);
            dy = pyr_G_i(img_r - 1, img_c) - pyr_G_i(img_r + 1, img_c);
            % 计算梯度 幅值 与 幅角
            mag = sqrt(dx^2 + dy^2);
            % 由于atan2的特性 计算出的度数为 -pi ~ pi 故需要经过一个转化
            if (dx >= 0 && dy >= 0 || dx <= 0 && dy >= 0)
                ang = atan2(dy, dx);
            else
                ang = atan2(dy, dx) + 2*pi;
            end
            ang = ang * 360/(2 * pi); % 转化为角度制
            
            % 计算落入的直方图柱数
            bin = round(36 / 360 * ang);
            if( bin >= 36 )
                bin = bin - 36;
            elseif( bin < 0 ) 
                bin = bin + 36;
            end
            
            % 计算高斯加权项 
            w_G = exp(- (i^2 + j^2) / (2 * (1.5 * scale)^2));
            
            % 存入histtemp   1   2    3 ~ 38    39  40
            histtemp(bin + 3) = histtemp(bin + 3) + mag * w_G;
            
        end
    end
 
    % 对直方图平滑处理 填充histtemp的空位  1   2    3 ~ 38    39  40
    hist = zeros(1, 36); % 存放一个特征点的直方图
    histtemp(1 : 2) = histtemp(37 : 38);
    histtemp(39 : 40) = histtemp(3 : 4);
    for k = 3 : 38   % 加权移动平均  长度为5
        hist(k - 2) = (histtemp(k - 2) + histtemp(k + 2)) * (1/16) + ...
            (histtemp(k - 1) + histtemp(k + 1)) * (4/16) + ...
            histtemp(k) * (6/16);
    end 
    
    % 遍历直方图 求 主方向 与 辅方向
    hist_threshold = 0.8 * max(hist); % 计算幅度阈值

    for k = 1 : 36
        
        if (k == 1) % kl为第k柱左边一柱的索引 下同
            kl = 36;
        else
            kl = k - 1;
        end
        
        if (k == 36)
            kr = 1;
        else
            kr = k + 1;
        end
        
        % 这里不仅要求直方图的柱高超过阈值 还要求比左右相邻的柱都高
        if (hist(k) > hist(kl) && hist(k) > hist(kr) && hist(k) >= hist_threshold)
            % 通过 抛物线插值 计算精确幅角公式  精确值范围  bin 0 ~ 35...  |36
            bin = k + 0.5 * (hist(kl) - hist(kr)) / (hist(kl) - 2*hist(k) + hist(kr));
            if (bin < 0) % bin越界处理
                bin = bin + 36;
            elseif (bin >= 36)
                bin = bin - 36;
            end
            % 计算精确幅角
            ang =  bin * (360/36); % ang为角度制
            if (abs(ang - 360) < 10^-16)
                ang = 0;
            end
            
            % 更新keypoints
            kpt_temp = [kpt(1 : 5), ang];
            kpts = [kpts; kpt_temp];
        end
        
    end

end
toc


%% 特征向量生成
disp('求特征向量')
d = 4; % 子区间数目 就是 4 * 4 * 8 中的4
n = 8; % 梯度直方图柱数目 8个幅角区间
fvec = [];
tic
% 再次遍历特征点  每个特征点计算128维特征向量
for kpt_i = 1 : size(kpts, 1)
    kpt = kpts(kpt_i, :);
    oct_i = kpt(1);
    interval = kpt(2);
    kpt_r = kpt(3);
    kpt_c = kpt(4);
    scale = kpt(5);
    mainang = kpt(6); % 主方向
    % 取出对应尺度的高斯模糊图像
    pyr_G_i = pyr_G{oct_i}(:, :, interval);
    [pyr_G_i_r, pyr_G_i_c] = size(pyr_G_i);
    % 计算旋转矩阵参数
    cos_t = cos(mainang * pi / 180); % cos接受的是弧度制
    sin_t = sin(mainang * pi / 180);
    
    hist_width = 3 * scale; % 每个子区域边长  方域
    radius = round(hist_width * sqrt(2) * (d + 1) * 0.5); % 计算所有参与计算像素区域半径   圆域 
    % 判断下计算出的半径与图像对角线长 最后半径取小的
    % 当半径和图像对角线一样大时是一个极限 再大不过也是遍历图像所有像素 所以没有意义
    radius = min(radius, floor(sqrt(pyr_G_i_r^2 + pyr_G_i_c^2))); 
    % 对旋转矩阵以子区域坐标尺度进行归一化  以便于之后计算出的旋转相对坐标(r_rot, c_rot)是子区域坐标尺度的
    cos_t = cos_t / hist_width;
    sin_t = sin_t / hist_width;
    
    % 初始化存储直方图容器 他是一个三维矩阵 (d+2)*(d+2)*(n+2)
    % 三个维度分别是 子区域坐标行数 子区域坐标列数 直方图的柱数
    % 三个维度都分别多留了2个空位 在之后的插值时会超过 4*4*8的范围
    hist = zeros(d + 2, d + 2, n + 2);
    
    % 依旧遍历区域内所有像素点 计算直方图
    for i = -radius : radius  % 从上到下  行数
        for j = -radius : radius % 从左往右扫描  列数
            
            % 计算旋转后的相对坐标 (注意是子区域坐标尺度的)
            c_rot = j * cos_t - i * sin_t;
            r_rot = j * sin_t + i * cos_t;
            % 计算该像素落入的子区域坐标 (rbin, cbin)
            % |_*_|_*_|_*_|_*_|
            % |_*_|_*_|_*_|_*_|
            % |_*_|_*_|_*_|_*_|
            % |_*_|_*_|_*_|_*_|
            rbin = r_rot + d/2 - 0.5; %  计算子区域坐标尺度下的绝对坐标 并做了0.5的平移
            cbin = c_rot + d/2 - 0.5; % 取值范围  -1 ~ 3... |4
            % 0.5的平移使得子区域坐标轴交点都落在子区域左上角 便于之后插值 计算一个像素点对周围四个子区域的直方图贡献
            % 计算像素图像坐标绝对(img_r, img_c)
            img_r = kpt_r + i;
            img_c = kpt_c + j;
            
            if (-1 < rbin && rbin < d && -1 < cbin && cbin < d && ...
                   1 < img_r && img_r < pyr_G_i_r && 1 < img_c && img_c < pyr_G_i_c)
               
                % 计算梯度 注意 dy
                dx = pyr_G_i(img_r, img_c + 1) - pyr_G_i(img_r, img_c - 1);
                dy = pyr_G_i(img_r - 1, img_c) - pyr_G_i(img_r + 1, img_c);
                
                % 计算梯度 幅值 与 幅角
                mag = sqrt(dx^2 + dy^2);
                % 由于atan2的特性 计算出的度数为 -pi ~ pi 故需要经过一个转化
                if (dx >= 0 && dy >= 0 || dx <= 0 && dy >= 0)
                    ang = atan2(dy, dx);
                else
                    ang = atan2(dy, dx) + 2*pi;
                end
                ang = ang * 360/(2 * pi); % 转化为角度制
                
                % 判断幅角落在哪个直方图的柱内
                obin = (ang - mainang) * (n / 360); % 取值 0 ~ 7...  |8
                % 计算高斯加权后的幅值          
                w_G = exp(- (r_rot^2 + c_rot^2) / (0.5 * d^2)); % 计算高斯加权项   -1/((d/2)^2 * 2)   ->    -1/(d^2 * 0.5)
                mag = mag * w_G;
                  
                r0 = floor(rbin); % -1 0 1 2 3
                c0 = floor(cbin);
                o0 = floor(obin);
              
                % 相当于立方体内点坐标
                rbin = rbin - r0;
                cbin = cbin - c0;
                obin = obin - o0;
                
                % 柱数o0越界循环
                if (o0 < 0)
                    o0 = o0 + n;
                elseif (o0 >= n)
                    o0 = o0 - n;
                end
                 
                % 三线性插值 填充hist的内容 将像素点幅值贡献到周围四个子区域的直方图中去
                % 计算8个贡献值    0 < rbin cbin obin < 1
                % 先计算贡献权值 
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
                % obin    1 ~ 8    |   9 10   插值会到9     o0   0 ~ 7
                temp = zeros(2, 2, 2);
                temp(:, :, 1) = [v_rco000, v_rco010; v_rco100, v_rco110];
                temp(:, :, 2) = [v_rco001, v_rco011; v_rco101, v_rco111];
                hist(r0+2 : r0+3, c0+2 : c0+3, o0+1 : o0+2) = hist(r0+2 : r0+3, c0+2 : c0+3, o0+1 : o0+2) + mag .* temp;
            end 
            
        end
    end
    
    % 遍历子区域  从hist直方图中导出特征向量
    fvec_i = [];
    for i = 1 : d   % 中间 4*4*8 的部分
        for j = 1 : d
            hist(i + 1, j + 1, 1) = hist(i + 1, j + 1, 1) + hist(i + 1, j + 1, 9);
            hist(i + 1, j + 1, 2) = hist(i + 1, j + 1, 2) + hist(i + 1, j + 1, 10);
            fvec_i = [fvec_i; reshape(hist(i + 1, j + 1, 1 : 8), 8, 1)];
        end
    end
    
    % 对特征向量幅值大的分量进行处理  改善非线性光照改变影响
    fvec_norm = norm(fvec_i, 2); % 特征向量模长
    fvec_threshold = 0.2 * fvec_norm;
    fvec_i(fvec_i > fvec_threshold) = fvec_threshold; % 大于阈值的分量设置为阈值
    
    % 归一化特征向量
    fvec_norm = norm(fvec_i, 2);
    fvec_i = fvec_i ./ fvec_norm;
    fvec = [fvec, fvec_i];
    
end
toc



%% 看DoG图像
% close all
% oct_i = 4;
% for i = 1 : s+2
%     figure
%     imshow(pyr_DoG{oct_i}(:, :, i)*25)
% end

%% 显示所有关键点
% close all
% figure
% imshow(imresize(img_org, 2, 'bicubic'))
% hold on
% for i = 1 : size(keypoints, 1)
%     plot(keypoints(i, 4), keypoints(i, 3), 'r.', 'MarkerSize', 15);
% end
% hold off

%% 显示筛选前后极值点
% for o = 1 : octave
%     figure
%     subplot(1,2,1); imshow(uint8(img))
%     title('筛选前')
%     hold on
%     for i = 1 : size(extrema, 1)
%         if (extrema(i, 1) == o)
%             plot(extrema(i, 4), extrema(i, 3), 'r.', 'MarkerSize', 15);
%         end
%     end
%     hold off
%     
%     subplot(1,2,2); imshow(uint8(img))
%     title('筛选后')
%     hold on
%     for i = 1 : size(keypoints, 1)
%         if (keypoints(i, 1) == o)
%             plot(keypoints(i, 4), keypoints(i, 3), 'r.', 'MarkerSize', 15);
%         end
%     end
%     hold off
% end

%% 显示带主方向关键点

% 转换坐标尺度
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





