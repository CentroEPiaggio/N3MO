% Sholl Analysis
%load('/Users/Chiara/Documents/SmRG/Risultati MST/Valentina/vert_gr_512_40x _062 neurone n°2.mat')
%mostra
patch('Faces', neurone1.faces, 'Vertices', neurone1.vertices,'FaceColor','green' , 'EdgeColor','none');axis equal;
camlight; camlight(-80, -10); lighting phong;
 
J_morpho = neurone1.neuron;

% % % %interpolazione-solo 1024
[X,Y,Z]=meshgrid(1:size(J_morpho, 1),1:size(J_morpho, 2),1:size(J_morpho,3));
[X2,Y2,Z2] = meshgrid(1:2:size(J_morpho, 1),1:2:size(J_morpho, 2),1:2:size(J_morpho, 3));
Vnew = interp3(X,Y,Z,double(J_morpho),X2,Y2,Z2);

J_morpho = zeros(size(Vnew));

for k=1:size(Vnew,3)
    J_morpho(:,:,k) = im2bw(Vnew(:,:,k));
end;

[p, r] = isosurface(X2, Y2, Z2, J_morpho, 0.5);
patch('Faces', p, 'Vertices',r,'FaceColor', 'red', 'EdgeColor','none');


%skeleton
 cd('skeletonization')

J_skel = Skeleton3D(J_morpho);

y = 1:size(J_morpho, 1);
x = 1:size(J_morpho, 2);
z = 1:size(J_morpho, 3);   

[X,Y,Z] = meshgrid(x,y,z);
[p, r] = isosurface(X, Y, Z, J_skel, 0.5);
patch('Faces', p, 'Vertices',r,'FaceColor', 'red', 'EdgeColor','none');
%patch('Faces', faces, 'Vertices', vertices,'FaceColor','green' , 'EdgeColor','none');axis equal;

cd ..

clear x y z

% % % % dim = zeros(size(J_morpho, 3), 2);
% % % % 
% % % % for i = 1:size(J_morpho, 3)
% % % %     %dim(i, 1) = sum(sum(J_morpho(:,:,i)));
% % % %     [cbw, rbw] = imfindcircles(J_morpho(:,:,i), [5 55]);
% % % %     if size(rbw, 1) == 1
% % % %         dim(i, 1) = rbw;
% % % %     end;
% % % % end;
% % % % 
% % % % %plot(dim)
% % % % tmp1 = find(dim~=0);
% % % % soma1 = min(tmp1);
% % % % soma2 = max(tmp1);
% % % % 
% % % % J_soma = J_morpho;
% % % % 
% % % % for z = 1:size(J_morpho, 3)
% % % %     if z < soma1 || z >soma2
% % % %         J_soma(:,:,z) = zeros(size(J_morpho, 1), size(J_morpho, 2), 1);
% % % %     end;
% % % % end;
% % % % 
% % % % %mostra per verifica
% % % % ys = 1:size(J_soma, 1);
% % % % xs = 1:size(J_soma, 2);
% % % % zs = 1:size(J_soma, 3);   
% % % % 
% % % % [Xs,Ys,Zs] = meshgrid(xs,ys,zs);
% % % % [p_soma, r_soma] = isosurface(Xs, Ys, Zs, J_soma, 0.5);
% % % %  patch('Faces', p_soma, 'Vertices',r_soma,'FaceColor', 'blue', 'EdgeColor','none');
% % % %  
% % % % mean_soma = uint16((soma1+soma2)/2);
% % % % [ms_c, mc_r] = imfindcircles(J_morpho(:,:,mean_soma), [10 35]);
% % % % somacenter = [ms_c, mean_soma];
% % % % somacenter = double(somacenter);
% % % % % % % % % 
% % % % dsc = [0, 0, 0, 0];
% % % % for i = 1:size(J_skel, 1)
% % % %     for j = 1:size(J_skel, 2)
% % % %         for k = 1:size(J_skel, 3)
% % % %             if J_skel(i,j,k) == 1
% % % %                 dsc = [dsc; j, i, k, sqrt((j-somacenter(1,1))^2+(i-somacenter(1,2))^2+(k-somacenter(1,3))^2)];
% % % %             end;
% % % %         end;
% % % %     end;
% % % % end;
% % % % dsc = dsc(2:end, :);
% % % % 
% % % % mind = min(dsc(:, 4));
% % % % coord = find(dsc(:, 4) == mind);
% % % % somacenter = dsc(coord, 1:3)
% % % % 
% % % % %

somacenter = [173, 283, 80];

estr = zeros(1,4);
for i = 1:size(J_skel, 1)
    for j = 1:size(J_skel, 2)
        for k = 1:size(J_skel, 3)
            if J_skel(i, j, k)==1
                dist = sqrt((j-somacenter(1,1))^2+(i-somacenter(1, 2))^2+(k-somacenter(1, 3))^2);
                dist = [dist, j, i, k];
                estr = [estr; dist];
            end;
        end;
    end;
end;
[max_estr, coord_d] = max(estr(:, 1));
coord_d = estr(coord_d, 2:end);

clear dist 

%r_mean = abs((soma1-soma2)/2);
r_mean = 15; %7.5; %.5;

%calcola Sholl. Distanza inter-cerchio costante (dc - ogni 60px)
num_int_dc = 0;
r_dc = r_mean:3:max_estr;
Sholl_dc = cell(size(r_dc, 2), 1);
i = 1;

[xx, yy, zz] = meshgrid(1:size(J_morpho, 1), 1:size(J_morpho, 2), 1:size(J_morpho, 3));

for r = r_mean:3:max_estr
    sphere1 = sqrt((xx-double(somacenter(1,1))).^2+(yy-double(somacenter(1,2))).^2+(zz-double(somacenter(1,3))).^2)<= (r+5);
    sphere2 = sqrt((xx-double(somacenter(1,1))).^2+(yy-double(somacenter(1,2))).^2+(zz-double(somacenter(1,3))).^2)<= (r-5);
    sphere = sphere1-sphere2;
    Sholl_dc_tmp = J_skel+sphere;
    Sholl_dc{i} = Sholl_dc_tmp;
    i = i+1;
    num_int_dc = [num_int_dc, (max(bwlabeln(Sholl_dc_tmp(:)>1)))/10];
end;

num_int_dc = ceil(num_int_dc(2:end));

%save('/Users/Chiara/Documents/ManualSegmentationTool/Risultati MST/Carolina-completa/vert_gr_512_40x _062 neurone n°2.mat', 'neurone1')
