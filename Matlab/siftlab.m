%% siftÌáÈ¡ÌØÕ÷ Demo
clear
clc
[img1, fvec1, kpts1] = sift('lena1.jpg', 1.6, 3);
[img2, fvec2, kpts2] = sift('lena2.jpg', 1.6, 3);

%% Í¼ÏñÆ¥Åä(±©Á¦Æ¥Åä)
dif = abs(size(img1, 1) - size(img2, 2));
if (size(img1, 1) > size(img2, 2))
    img2 = [img2; 255*ones(dif, size(img2, 2), size(img2, 3))];
else
    img1 = [img1; 255*ones(dif, size(img1, 2), size(img1, 3))]; 
end
img3 = [img1 img2];

march = []; 
d = [];
threshold = 0.15;
for kpt_i = 1 : size(fvec1, 2)
    for kpt_j = 1 : size(fvec2, 2)
        if (norm(fvec1(:, kpt_i) - fvec2(:, kpt_j), 2) < threshold)
            d = [d, norm(fvec1(:, kpt_i) - fvec2(:, kpt_j))];
            march = [march; kpt_i, kpt_j];
        end
    end
end
figure
imshow(img3)
hold on
for kpt_i = 1 : size(kpts1, 1)
    kpt = kpts1(kpt_i, :);
    plot(kpt(4), kpt(3), 'r.', 'MarkerSize', 10);
end

for kpt_i = 1 : size(kpts2, 1)
    kpt = kpts2(kpt_i, :);
    plot(kpt(4) + size(img1, 2), kpt(3), 'r.', 'MarkerSize', 10);
end

for i = 1 : size(march, 1)
    kpt1 = kpts1(march(i, 1), :);
    kpt2 = kpts2(march(i, 2), :);
    line([kpt1(4), kpt2(4) + size(img1, 2)], [kpt1(3), kpt2(3)], 'LineWidth', 1, 'Color', [rand(),rand(),rand()])
end






hold off

%%








