%%% N3MO
% created by Chiara Magliaro
% updated by Valerio Fantozzi, Eugenio Capotorti, Irene Cascavilla
% last release: April 2018

%% LOAD STRUCTURE and SELECT THE NEURON TO ANALYZE. SET THE PIXEL SIZE. SHOW THE NEURON
clear all
close all
clc

DEBUG = false;

uiwait(msgbox('Wellcome in N3MO. Please choose a structure to analyse'));

%LOAD STRUCTURE
[filename, pathname] = uigetfile('*.mat','Choose a structure');
%cd(pathname)
J_morpho = load(strcat(pathname,filename));
clear pathname filename;

multiPlot = true;

if multiPlot
    multiData = figure('Name','N3MO', 'NumberTitle', 'off');
    %'units','normalized','outerposition',[0 0 1 1])
end

%% CLEAN J_MORPHO'S NOISE

for neuronId=1:length(fieldnames(J_morpho))
    
    neuronMap = getfield(J_morpho, ['neurone', num2str(neuronId)]);     %creat copy of a neuron from J_morph
    
    %SMOOTHING
    %neuronMap = setfield(neuronMap, 'neuron', smooth3(neuronMap.neuron, 'box' , 3));
    
    
    neuJ_label = bwlabeln( neuronMap.neuron );
    neuJ_Obj = bwconncomp(neuJ_label);
    neuJ_props = regionprops(neuJ_Obj, 'basic');                        %search for object numbers
    
    if neuJ_Obj.NumObjects ~= 1
        areaMax = max([neuJ_props.Area]);                                 %area max is the only neuron in the map
        id_areaMax = find([neuJ_props.Area] == areaMax);
        
        for id=1:neuJ_Obj.NumObjects
            if id==id_areaMax
                continue                                                %DONOT delete the neuron
            end
            neuronMap.neuron(neuJ_Obj.PixelIdxList{1, id}) = 0;         %delete rubbish
        end
    end
    x = 1:size(neuronMap.neuron, 1);
    y = 1:size(neuronMap.neuron, 2);
    z = 1:size(neuronMap.neuron, 3);
    [X,Y,Z] = meshgrid(x,y,z);
    [nf, nv] = isosurface(X, Y, Z, neuronMap.neuron, 0.5);          %generate cleaned MAP with vertces and faces
    
    
    J_morpho =  setfield(J_morpho, ['neurone',num2str(neuronId)], 'vertices', nv);     %<-----------------------------------------------------------ERRORE!! neurone1 va cambiato
    J_morpho =  setfield(J_morpho, ['neurone',num2str(neuronId)], 'faces', nf);
    J_morpho =  setfield(J_morpho, ['neurone',num2str(neuronId)], 'neuron', neuronMap.neuron);
    
    clear id id_areaMax areaMax neuJ_label neuJ_Obj neuJ_props X Y Z x y z nf nv neuronMap neuronId
end

if DEBUG
    for k=1:144
        figure('Name', num2str(k));
        title(num2str(k),'Color', 'm')
        imshow(J_morpho.neurone1.neuron(:,:,k));%144-k));
    end
end

%% SELECT THE NEURON TO ANALYZE
sizeJ = length(fieldnames(J_morpho));             %neuron number
if sizeJ > 1                                        % neuron > 1
    val = inputdlg (['Number of neurons in the selected structure:', num2str(sizeJ), '. Which one do you want to analyse?']);
    val = val{1,1};
    neuJ = getfield(J_morpho, ['neurone', val]);
else                                                %nueron = 1
    neuJ = J_morpho.neurone1;
end
clear sizeJ;

%SET THE PIXEL SIZE (PIXEL->METER)
pixel_size = inputdlg('Enter the pixel size:');
pixel_size = str2num(pixel_size{1,1});

%% SHOW THE NEURON
if multiPlot
    subplot(2,2,1);
else
    f1 = figure('Name','NEURON');
end

patch('Faces', neuJ.faces, 'Vertices', neuJ.vertices,'FaceColor','yellow', 'EdgeColor','none');
axis equal; camlight; camlight(-80, -10); lighting phong; hold on; alpha 0.25; box on; grid on;
xlabel('x_L');  ylabel('y_L');  zlabel('z_L'); view(3);


if DEBUG
    neuShape = alphaShape(neuJ.vertices, 1.5,'RegionThreshold',1000,'HoleThreshold',1);          %soglia ottima 1.5 ma holes
    figure, plot(neuShape);
    axis equal; camlight; camlight(-80, -10); lighting phong; hold on; alpha 0.1;
end

%% NEURON AREA AND VOLUME
neu_area = 0;
p = neuJ.faces;
vert = neuJ.vertices;

%calculate the area of a 3D triangle
if isempty(p)==0||isempty(vert)==0                  %if (neuJ.faces) OR (neuJ.vertices) matrix are NOT EMPTY
    a = vert(p(:, 2), :) - vert(p(:, 1), :);        %P1P2
    b = vert(p(:, 3), :) - vert(p(:, 1), :);        %P3P1
    cr = cross(a, b, 2);                            %P1P2 cross product P3P1
    neu_area = 1/2 * sum( sqrt(sum(cr.^2, 2)) )*pixel_size*pixel_size;  %NEURON AREA
end

neu_vol = length(find(neuJ.neuron)) *pixel_size^3; %(#pixels of a Neuron) * (pixel_size)^3 = NEURON VOLUME

clear p vert a b cr

if DEBUG
    % CHECK NEU VOLUME e AREA FROM SHAPE
    neu_areaCheck = surfaceArea(neuShape);
    neu_volCheck = volume(neuShape);
    %imfill
    %findHoles regionprops
end



%% SOMA DETECTION. SOMA AREA AND VOLUME

%search for last area of max volume
radius = 5;
somaJ = imopen(neuJ.neuron, strel('diamond', radius));         %morphological opening -> remove elements having radius less than 10 pixels
soma_label = bwlabeln(somaJ);
stat = regionprops(soma_label, 'Area');            %properties soma
while size([stat.Area],2) > 1 %while #area > 1
    radius = radius +1;
    somaJ = imopen(neuJ.neuron, strel('diamond', radius));         %morphological opening -> remove elements having radius less than 10 pixels
    soma_label = bwlabeln(somaJ);
    stat = regionprops(soma_label, 'Area');            %properties soma
end
stat = regionprops(soma_label, 'all');

[somaf, somav] = isosurface(somaJ, 0.5);                %extract volume data from somaJ -> take faces and verteces in arrays
patch('Faces', somaf, 'Vertices', somav, 'FaceColor','magenta' , 'EdgeColor','none');axis equal;  %SHOW SOMA
camlight; camlight(-80, -10); lighting phong;alpha 0.25;


soma_vol = length(find(somaJ))*pixel_size*pixel_size*pixel_size;      %SOMA VOLUME = #somaPixels * pixelSize^3

soma_area = 0;
if isempty(somaf)==0||isempty(somav)==0
    a = somav(somaf(:, 2), :) - somav(somaf(:, 1), :);
    b = somav(somaf(:, 3), :) - somav(somaf(:, 1), :);
    cr = cross(a, b, 2);
    soma_area = 1/2 * sum(sqrt(sum(cr.^2, 2)))*pixel_size*pixel_size;   %SOMA AREA
end

if DEBUG
    somaShape = alphaShape(somav, 4);
    soma_areaCheck = surfaceArea(somaShape);
    soma_volCeck = volume(somaShape);
    figure;
    plot(somaShape);
    axis equal; camlight; camlight(-80, -10); lighting phong; hold on; alpha 0.3;
end

clear somaf a b cr

%% SOMA CENTER 3D-COORDINATES
% soma_label = bwlabeln(somaJ);
% stat = regionprops(soma_label, 'basic');            %properties soma

soma_center = [stat.Centroid(1,2), stat.Centroid(1,1), stat.Centroid(1,3)];   %soma centroid from properties

hold on;
scatter3(soma_center(1,2), soma_center(1,1), soma_center(1,3), 'd')  %plot soma centroid

if DEBUG
    figure;
    for k=1:143
        title(num2str(k))
        imshow(somaJ(:,:,144-k));
    end
end

%clear soma_label stat;
%% SOMA RADIUS
soma_radius = sqrt(soma_area/(4*pi));               %radius_cicrle = sqrt(A/4*pi)
soma_radius_um = soma_radius * pixel_size; %diameter in micro me

max_estr = 1;
for i = 1:size(somav, 1)
    dist = sqrt((somav(i,1)-soma_center(1,2))^2+(somav(i,2)-soma_center(1, 1))^2+(somav(i,3)-soma_center(1, 3))^2);
    if dist>max_estr
        max_estr = dist;
    end
end
soma_radius_MAX = max_estr;
soma_radius_MAX_um = soma_radius_MAX * pixel_size;

clear max_estr i dist;

%% NEURON SKELETONIZATION STEP

addpath(genpath('skeletonization'));
J_skel = Skeleton3D(neuJ.neuron);

%% SHOW SKELETON

x = 1:size(J_skel, 1);
y = 1:size(J_skel, 2);
z = 1:size(J_skel, 3);

[X,Y,Z] = meshgrid(x,y,z);
[p, r] = isosurface(X, Y, Z, J_skel, 0.5);          %create soma map
patch('Faces', p, 'Vertices', r, 'FaceColor', 'red', 'EdgeColor','none'); axis equal; camlight; camlight(-80, -10); lighting phong;

clear x y z X Y Z p r
%% NEURON RADIAL EXTENSION
%search for maximum distance from soma center
max_estr = 1;

for i = 1:size(J_skel, 1)
    for j = 1:size(J_skel, 2)
        for k = 1:size(J_skel, 3)
            
            dist = sqrt((i-soma_center(1,1))^2+(j-soma_center(1, 2))^2+(k-soma_center(1, 3))^2);
            
            if J_skel(i,j,k) == 1 && dist>max_estr
                max_estr = dist;
                max_estr_coord = [i,j,k];
            end
            
        end
    end
end

max_estr_um = max_estr * pixel_size;

clear i j k dist

%% SHOLL ANALYSIS

sholl = struct;

%DATA REQUEST
% r_start = radius where start analysis
% r_step  = radius step for sholl analysius
% r_end   = radius where stop analysis

r_start_default = ceil(soma_radius_MAX+2);
r_end_default = ceil(max_estr-1);

prompt = {'r start:','#circles (only for constant number of circles):', 'r step (only for constant inter-circle distance):', 'r end:'};
dlg_title = '3D Sholl analysis parameters';
num_lines = 1;
defaultans = {num2str(r_start_default),num2str(0), num2str(0), num2str(r_end_default)};

conditions = false;
%conditions =
%       * r_star < soma_radius
%       * r_end > max_estr
%       * both #circles & r_step are not set (==0)
%       *  input aren't both #circles and r_step

while ~conditions
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    r_start = ceil(str2num(answer{1,1}));
    r_end = ceil(str2num(answer{4,1}));
    if r_start<soma_radius || r_end>max_estr || r_start>r_end
        h = warndlg('INVALID INPUT!');
        uiwait(h);
    else if str2num(answer{2, 1})==0 && str2num(answer{3, 1})==0
            h = warndlg('BOTH 2th AND 3th INPUT ARE 0');
            uiwait(h);
        else
            conditions = true;
        end
    end
end

if str2num(answer{2,1})~= 0 && str2num(answer {3,1})~= 0        %if input both circles and r step
    h = warndlg('Do you want to perform constant number of circles or constant inter-circle distance analysis?');
    uiwait(h);
end

if str2num(answer{2, 1})~= 0                                    %if input is circle number
    num_circles = str2num(answer{2,1});
    r_step = ceil((r_end-r_start)/num_circles);
end

if str2num(answer{3, 1}) ~= 0                                   %if input is r step
    r_step = ceil(str2num(answer{3,1}));
end

%ANALYSIS
r_sholl = r_start:r_step:r_end;                                 % contains radius for sholl analysis
num_int = zeros(size(r_sholl, 1));

[cc, rr, pp] = meshgrid(1:size(J_skel, 1), 1:size(J_skel, 2), 1:size(J_skel, 3));

i=1;

inters_coord = struct;
shell_radii = (pixel_size*(r_start:r_step:r_end))';%converts shell radius from pixel size to um

h = waitbar(0,'Sholl Analysis...');

cont = 1;
for r = r_start:r_step:r_end                    %cycle in shells
    sphere1 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r-2); %logical internal sphere area
    sphere2 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r+2); %logical external sphere area
    sphere3 = sphere2-sphere1;
    
    SphSholl = J_skel + sphere3;               	%merge Skel with Thoroid -> obj value = 2, empty area = 1
    SphSholl = SphSholl>1;                     	%trasform map in bw
    [Objects, num] = bwlabeln(SphSholl);     	%Objects = obj founded is shell analyzed ; num = num obj founded
    tmp = regionprops(Objects, 'centroid');
    centroids = cat(1, tmp.Centroid);           %centroids of segment of skel founded in the shell
    
    sholl = setfield(sholl, ['r',num2str(cont)], [centroids, zeros(size(centroids,1),1), zeros(size(centroids,1),1)] ); %save in sholl struct coordinates of centroids
    cont= cont+1;
    
    %plot the intersection points, shell-by-shell
    %     for j = 1:size(centroids, 1)
    %         hold on;
    %         scatter3(centroids(j,1), centroids(j,2), centroids(j,3));
    %     end
    
    fileout = ['shell',num2str(length(fieldnames(inters_coord))+1)];
    inters_coord.(fileout) = centroids;       	%struct with for each shell write centroid postion
    
    num_int(i,1) = num;     %num of obj in each shell
    i = i+1;
    
    waitbar(r / r_end);
end
close(h);
clear conditions sphere1 sphere2 sphere3 tmp r_start_default r_end_default prompt dlg_title num_lines defaultans answer cc centroids cont h i j fileout num num_circles Objects pp r rr  SphSholl

%% SHOLL SHOW
if multiPlot
    subplot(2,2,2);
else
    f2 = figure('Name','SHOLL');
end

hold on;
%show skeletonn
x = 1:size(J_skel, 1);
y = 1:size(J_skel, 2);
z = 1:size(J_skel, 3);
[X,Y,Z] = meshgrid(x,y,z);
[p, r] = isosurface(X, Y, Z, J_skel, 0.5);          %create soma map
patch('Faces', p, 'Vertices', r, 'FaceColor', 'red', 'EdgeColor','none');
axis equal; camlight; camlight(-80, -10); lighting phong; box on; grid on; view(3);



for r=1:numel(fieldnames(sholl))
    for p =1:size(getfield(sholl, ['r' num2str(r)]),1)
        scatter3( getfield(sholl, ['r' num2str(r)], {p,1}), getfield(sholl, ['r' num2str(r)], {p,2}), getfield(sholl, ['r' num2str(r)], {p,3}), 'o', 'b', 'filled')
    end
    step = round(numel(fieldnames(sholl)) / 10);
    if mod(r,step)==0
        [x,y,z] = sphere;
        x= (x* r_sholl(r))+soma_center(2);
        y= (y* r_sholl(r))+soma_center(1);
        z= (z* r_sholl(r))+soma_center(3);
        surf(x,y,z, 'FaceColor', 'k', 'FaceAlpha', 0.015, 'EdgeColor', 'k', 'LineStyle', ':', 'LineWidth', 0.1, 'EdgeAlpha', 0.7);
    end
end

clear x y z step X Y Z p r
%%
%METRIC BASED ON SAMPLED DATA
inters_radii = r_step; %lenght(r);              %number of sampling radii intersecting the arbor at least once
sum_int = sum(num_int);                         %sum of intersections
mean_int = sum_int/inters_radii;                %mean of intersections
median_int = median(num_int);                   %median value of intersections
[max_num_int, tmp1] = max(num_int);             %max number of intersections (i.e. maximum in a linear [N vs Distance] profile)
critical_radius = r_start+r_step*tmp1;          %radius at which the max number of intersections occur
shoenen_index_sampled = max_num_int/num_int(1,1);   %shoenen index

skewness_data = skewness(num_int);              %skewness of sampled data: positive value: asymmetrical distrib with a longer tail in the right
kurtosis_data = kurtosis(num_int);              %kurtosis of sampled data: >0 means a distribution with a more peaked curve than a gaussian


%METRIC BASED ON FITTED DATA
% linear method (data fitted on a polynomial function)
shell_area = pi*pixel_size*pixel_size.*(shell_radii).^2;

% % % % IntArea = num_int./shell_area;  %???
% % % %
% % % % l = log10(IntArea);
% % % % TF = isinf(l);
% % % % for i=1:length(TF)
% % % %     if TF(i)==1
% % % %         l(i) =[ ];
% % % %         shell_radii(i) =[ ];
% % % %     end;
% % % % end;
% % % %
% % % % C1 = corrcoef(shell_radii,l);
% % % % R_sl = -C1(1,2);
% % % %
% % % % C1 = corrcoef(log10(shell_radii),l);
% % % % R_ll = -C1(1,2);
% % % %
% % % % RD = (R_sl/R_ll)^2;
% % % %
% % % % [p, ErrorEst] = polyfit(shell_radii,l,1);
% % % % k_sl = p(1);
% % % %
% % % % % fitting evaluation
% % % % linear_fit = polyval(p,
% % % % pop_fit=polyval(p,log10(shell_radii),ErrorEst);
% % % % % Valutazione del fit e dell'errore di predizione
% % % % [pop_fit,delta] = polyval(p,log10(r),ErrorEst);
% % % % % Plot dei dati, del fit e degli intervalli di confidenza
% % % % % figure
% % % % % plot(log10(r),log10(y),'+',log10(r),pop_fit,'g-',log10(r),pop_fit+2*delta,'r:',log10(r),pop_fit-2*delta,'r:');
% % % %
% % % % [p,ErrorEst]=polyfit(log10(shell_radii),l,1);
% % % % k_ll = p(1);
% % % %
% % % % % fitting evaluation
% % % % pop_fit=polyval(p,log10(shell_radii),ErrorEst);
% % % % % Valutazione del fit e dell'errore di predizione
% % % % [pop_fit,delta] = polyval(p,log10(shell_radii),ErrorEst);
% % % % % Plot dei dati, del fit e degli intervalli di confidenza
% % % % %figure
% % % % %plot(log10(r),log10(y),'+',log10(r),pop_fit,'g-',log10(r),pop_fit+2*delta,'r:',log10(r),pop_fit-2*delta,'r:');

%DRAW SHOLL SHELL - it takes time...
% for r = r_start:r_step:r_end
%     sphere1 = sqrt((xx-double(soma_center(1,2))).^2+(yy-double(soma_center(1,1))).^2+(zz-double(soma_center(1,3))).^2)<= r;
%     sphere2 = sqrt((xx-double(soma_center(1,2))).^2+(yy-double(soma_center(1,1))).^2+(zz-double(soma_center(1,3))).^2)<= (r+1);
%     sphere = sphere2-sphere1;
%     hold on
%     isosurface(sphere, 0)
%     alpha 0.2
% end;




%% TREE

% SHOW
J_skellNode = J_skel;

if multiPlot
    subplot(2,2,3);
else
    f3 = figure('Name', 'TREE');
end

x = 1:size(J_skellNode, 1);
y = 1:size(J_skellNode, 2);
z = 1:size(J_skellNode, 3);
[X,Y,Z] = meshgrid(x,y,z);
[p, r] = isosurface(X, Y, Z, J_skellNode, 0.9);          %create soma map
patch('Faces', p, 'Vertices', r, 'FaceColor', 'red', 'EdgeColor','none');
axis equal; camlight; camlight(-80, -10); lighting phong; hold on; alpha 0.3; box on; grid on; view(3);
xlabel('x_L');  ylabel('y_L');  zlabel('z_L');
clear x y z X Y Z p r;

%d = msgbox('TREE CONSTRUCTION');
disp('TREE CONSTRUCTION');

hold on;

%ARMS SEARCHING ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[cc, rr, pp] = meshgrid(1:size(J_skel, 1), 1:size(J_skel, 2), 1:size(J_skel, 3));

%first point outside Soma
r=soma_radius_MAX+3;
sphere1 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r); %logical internal sphere area
sphere2 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r+1.5); %logical external sphere area
sphere = sphere2-sphere1;
SphSholl = J_skel + sphere;                              %merge Skel with Thoroid -> obj value = 2, empty area = 1
[y,x,z] = ind2sub(size(SphSholl), find(SphSholl==2) ) ;

if DEBUG
    for a=1:size(x,1)
        scatter3(x(a), y(a), z(a) );
    end
end

% tree structure inizialization
tree = struct;
tree.nodes = struct;
tree.nodes = [soma_center];            %nodes in tree

tree.arm1 = struct;
tree.arm1.points = struct;
tree.arm1.points = [];
tree.arm1.points(1,:) = [x(1),y(1),z(1)];

%second point outside Soma
r=soma_radius_MAX+4;
sphere1 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r); %logical internal sphere area
sphere2 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r+1.5); %logical external sphere area
sphere = sphere2-sphere1;
SphSholl = J_skel + sphere;                              %merge Skel with Thoroid -> obj value = 2, empty area = 1
[y,x,z] = ind2sub(size(SphSholl), find(SphSholl==2) ) ;
tree.arm1.points(2,:) = [x(1),y(1),z(1)];

scatter3(tree.arm1.points(1,1),tree.arm1.points(1,2),tree.arm1.points(1,3), 'c');
scatter3(tree.arm1.points(2,1),tree.arm1.points(2,2),tree.arm1.points(2,3), 'c');


POINTS = [tree.arm1.points(1,:); tree.arm1.points(2,:)];    %tree's points

%NODE : points to start arm construction
NODE = [tree.arm1.points(2,1), tree.arm1.points(2,2), tree.arm1.points(2,3)]; %NODE inizialization
nodeFounded = false;
armNumb = 0;
armNumbNode = numel(fieldnames(tree));

%points =2;
%nodes=0;

while ~isempty(NODE)    %search untill there aren't new points in NODE ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    sphere_center = [NODE(1,2), NODE(1,1), NODE(1,3)];      %new arm from a point after a node
    NODE = NODE(2:end, :);                                  %delete first points
    armNumb = armNumb + 1;
    nodeFounded = false;
    
    printstring = ['Points: ' num2str(size(POINTS,1)) '\n' 'Arms:   ' num2str(armNumbNode) '\n' 'Nodes:  ' num2str(size(tree.nodes,1)) '\n'];
    fprintf(printstring);
    
    %fprintf(    ['Points: ' num2str(size(POINTS,1)) '\n' 'Arms:   ' num2str(size(armNumbNode,1)) '\n' 'Nodes:  ' num2str(size(tree.nodes,1)) '\n']	);
    %disp(['ARMS: ' num2str(armNumb)]);
    
    while ~nodeFounded	%search for a new node ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        %search for new points in arm
        intersectionFounded = false;
        R=2;
        while ~intersectionFounded %increase radius until doesn't find anything
            sphere = sqrt((cc-double(sphere_center(1,2))).^2+(rr-double(sphere_center(1,1))).^2+(pp-double(sphere_center(1,3))).^2) < (R); %logical external sphere area
            SphSholl = J_skellNode + sphere;                                 %merge Skel with Thoroid -> obj value = 2, empty area = 1
            x= []; y= []; z= [];
            [y,x,z] = ind2sub(size(SphSholl), find(SphSholl==2)) ;     %search for intersections
            intersections = [x y z];
            
            if size(intersections, 1) <= 1  %doesn't find any points unless sphere_center -> increase search radius
                R = R + 1;
                %disp(['RETRAY INTERSECTION: ' num2str(R)]);
            else
                intersectionFounded = true;
            end
        end
        
        %CONDITION ERROR        ------------------------------------------------------------------------------------
        if isempty(intersections) && ~isempty(NODE)
            %disp('ERRORE 0')
            nodeFounded = true;
        end
        
        % CONDITION 1) ARM END	------------------------------------------------------------------------------------
        % if find 2 points in searchSphere and not all are new
        if size(intersections,1) == 2 && prod(ismember(intersections, POINTS, 'rows'))
            %search for new points with searchRadius 2->5
            %if anything is founded arm is ended
            for  R=2:5
                sphere = sqrt((cc-double(sphere_center(1,2))).^2+(rr-double(sphere_center(1,1))).^2+(pp-double(sphere_center(1,3))).^2) < (R); %logical external sphere area
                SphSholl = J_skellNode + sphere;                                 %merge Skel with Thoroid -> obj value = 2, empty area = 1
                x= []; y= []; z= [];
                [y,x,z] = ind2sub(size(SphSholl), find(SphSholl==2)) ;     %search for intersections
                intersections = [x y z];
                
                %new points founded with radius extended -> ARM CONSTRUCTION CONTINUE
                if size(intersections,1) == 3 && ~prod(ismember(intersections, POINTS,'rows'))
                    %disp('ARM CONSTRUCTION 2');
                    for j=1:3
                        if ~ismember(intersections(j,:), POINTS, 'rows')
                            tree = setfield(tree, ['arm',num2str(armNumb)], 'points', [getfield(tree, ['arm',num2str(armNumb)], 'points') ; intersections(j,:)] );  %add new point to tree's arm
                            sphere_center = [intersections(j,2),intersections(j,1),intersections(j,3)];       	%reset spherecenter to new point
                            scatter3(intersections(j,1),intersections(j,2),intersections(j,3) , 15, 'g'); %plot
                            
                            POINTS = [POINTS; intersections(j,:)];
                            
                            %points = points+1;
                            %disp(['PUNTO: ' num2str(points)]);
                            %disp(['RAMO: ' num2str(armNumb)]);
                            %disp([num2str(intersections(j,1)) ' | ' num2str(intersections(j,2)) ' | ' num2str(intersections(j,3))]);
                        end
                    end
                end
                
                %maximum radius search -> ARM ENDED
                if prod(ismember(intersections, POINTS, 'rows')) && R==5
                    %disp("ARM ENDED");
                    nodeFounded = true;
                end
            end
            
            % CONDITION 2)      ------------------------------------------------------------------------------------
            %if find 2 points and are all new
        elseif size(intersections, 1) == 2 && ~prod(ismember(intersections, POINTS, 'rows'))
            for j=1:2
                if ~ismember(intersections(j,:), POINTS, 'rows')
                    tree = setfield(tree, ['arm',num2str(armNumb)], 'points', [getfield(tree, ['arm',num2str(armNumb)], 'points') ; intersections(j,:)] );  %add new point to tree's arm
                    sphere_center = [intersections(j,2),intersections(j,1),intersections(j,3)];       	%reset spherecenter to new point
                    scatter3(intersections(j,1),intersections(j,2),intersections(j,3) , 15, 'g'); %plot
                    
                    POINTS = [POINTS; intersections(j,:)];
                    
                    %points = points+1;
                    %disp(['PUNTO: ' num2str(points)]);
                    %disp(['RAMO: ' num2str(armNumb)]);
                    %disp([num2str(intersections(j,1)) ' | ' num2str(intersections(j,2)) ' | ' num2str(intersections(j,3))]);
                end
            end
            
            % CONDITION 3) ARM CONSTRUCTION - NOT A NODE ------------------------------------------------------
        elseif size(intersections,1) == 3 && ~prod(ismember(intersections, POINTS,'rows'))
            %disp('ARM CONSTRUCTION');
            
            for j=1:3
                if ~ismember(intersections(j,:), POINTS, 'rows')
                    tree = setfield(tree, ['arm',num2str(armNumb)], 'points', [getfield(tree, ['arm',num2str(armNumb)], 'points') ; intersections(j,:)] );  %add new point to tree's arm
                    sphere_center = [intersections(j,2),intersections(j,1),intersections(j,3)];       	%reset spherecenter to new point
                    scatter3(intersections(j,1),intersections(j,2),intersections(j,3) , 15, 'g'); %plot
                    
                    POINTS = [POINTS; intersections(j,:)];
                    
                    %points = points+1;
                    %disp(['PUNTO: ' num2str(points)]);
                    %disp(['RAMO: ' num2str(armNumb)]);
                    %disp([num2str(intersections(j,1)) ' | ' num2str(intersections(j,2)) ' | ' num2str(intersections(j,3))]);
                end
            end
            
            % CONDITION 4) ERROR -------------------------------------------------------------------------------
        elseif size(intersections, 1) == 3 && prod(ismember(intersections, POINTS,'rows'))
            %disp('ERROR CONDITION 2');
            nodeFounded = true;
            
            % CONDITION 5) NODE FOUNDED ------------------------------------------------------------------------
        elseif size(intersections,1) > 3
            
            %add noded founded in tree.nodes
            %disp("NODE FOUNDED");
            if ~ismember(sphere_center, getfield(tree, 'nodes'),'rows')
                tree.nodes = [tree.nodes; sphere_center];               %save the node coordinates in tree
                scatter3(sphere_center(2),sphere_center(1),sphere_center(3), 30, 'filled', 'b'); %plot
            end
            
            for j=1:3
                if ~ismember(intersections(j,:), POINTS,'rows')
                    POINTS = [POINTS; intersections(j,:)];
                end
            end
            
            % NODE HOLES
            % search for intersection beetwen skel and a toroid of radius r=5 and height 1
            % eliminate errors in nodes
            r=5;
            sphere1 = sqrt((cc-double(sphere_center(1,2))).^2+(rr-double(sphere_center(1,1))).^2+(pp-double(sphere_center(1,3))).^2)<= (r); %logical internal sphere area
            sphere2 = sqrt((cc-double(sphere_center(1,2))).^2+(rr-double(sphere_center(1,1))).^2+(pp-double(sphere_center(1,3))).^2)<= (r+1); %logical external sphere area
            sphere = sphere2-sphere1;
            SphSholl = J_skellNode + sphere;                              %merge Skel with Thoroid -> obj value = 2, empty area = 1
            x= []; y= []; z= [];
            [y,x,z] = ind2sub(size(SphSholl), find(SphSholl==2)) ;
            intersectionsExpanded = [x y z];
            
            %count the new points founded with radius extended
            numbInt = 0;
            for j=1:size(intersectionsExpanded,1)
                if ~ismember(intersectionsExpanded(j,:), POINTS, 'rows')
                    numbInt = numbInt +1;
                end
            end
            
            % error if find with toroid finds only 1 arm (1 new point)
            %if error -> use sphere method -> save new intersections in NODE            and make new arms
            %else     -> use toroid method -> save new intersectionsExpanded in NODE    and make new arms
            if numbInt == 1
                %disp("ERROR HOLE");
                
                for j=1:size(intersections,1)
                    armNumbNode = numel(fieldnames(tree));
                    if ~ismember(intersections(j,:), getfield(tree, ['arm',num2str(armNumb)], 'points'),'rows') && ~ismember(intersections(j,:), NODE,'rows')
                        NODE = [NODE; intersections(j,:)];                                                  %add points new arm
                        
                        %new arm
                        tree = setfield(tree, ['arm',num2str(armNumbNode)], 'points', [sphere_center(2), sphere_center(1), sphere_center(3); intersections(j,1), intersections(j,2), intersections(j,3)]);
                        text( intersections(j,1), intersections(j,2), intersections(j,3) , ['arm' num2str(armNumbNode)] );
                    end
                end
            else
                for j=1:size(intersectionsExpanded,1)
                    if ~ismember(intersectionsExpanded(j,:), POINTS, 'rows')
                        armNumbNode = numel(fieldnames(tree));
                        NODE = [NODE; intersectionsExpanded(j,:)];
                        
                        tree = setfield(tree, ['arm',num2str(armNumbNode)], 'points', [sphere_center(2), sphere_center(1), sphere_center(3); intersectionsExpanded(j,1), intersectionsExpanded(j,2), intersectionsExpanded(j,3)]);
                        text(intersectionsExpanded(j,1), intersectionsExpanded(j,2), intersectionsExpanded(j,3) , ['arm' num2str(armNumbNode)] );
                    end
                end
            end
            
            % HOLE IN THE SKEL
            sphere3 = sqrt((cc-double(sphere_center(1,2))).^2+(rr-double(sphere_center(1,1))).^2+(pp-double(sphere_center(1,3))).^2)< (r-1); %logical internal sphere area
            J_skellNode = J_skellNode - sphere3; %delete the holes from the skel to work with toroids
            
            nodeFounded = true;
            
            %nodes = nodes +1;
            %disp(['NODES: ' num2str(nodes)]);
            
        end
    end     %exit if find a node
    
    %refresh terminal data
    fprintf( repmat('\b', 1, size(printstring,2)-3 ) ) ;
    
end %exit if tree is completed

% %message box data
% set(d, 'position', [100 440 150 100]);
% set(findobj(d,'Tag','MessageBox'),'String',{'TREE COMPLETED' [num2str(armNumbNode) ' arms'] [num2str(size(POINTS,1)) ' points'] [num2str(size(tree.nodes,1)) ' nodes']})

%terminal data
fprintf( repmat('\b', 1, size(printstring,2)-3+17 ) ) ;
disp('TREE COMPLETED');
fprintf(printstring);

% %text data
% text( 120,300,150 , [num2str(armNumbNode) ' arms']);
% text( 120,300,140 , [num2str(size(POINTS,1)) ' points']);
% text( 120,300,130 , [num2str(size(tree.nodes,1)) ' nodes']);

% HIERARCHY
tree.hierarchy = [];
tree.hierarchy(1,1) = 1;
tree.hierarchy(1,2) = 2;


for i=2:numel(fieldnames(tree))-2
    arm = getfield(tree, ['arm',num2str(i)], 'points');
    
    for j=1:size(tree.nodes,1)
        %begin node
        if [tree.nodes(j,2) tree.nodes(j,1) tree.nodes(j,3)] == arm(1,:)
            tree.hierarchy(i,1) = j;
        end
        %end node
        if [tree.nodes(j,2) tree.nodes(j,1) tree.nodes(j,3)] == arm(end,:)
            tree.hierarchy(i,2) = j;
        end
    end
    
end

clear NODE i j d r numbInt intersectionFounded x y z nodeFounded sphere1 sphere2 sphere3 SphSholl sphere printstring sphere_center armNumb armNumbNode R intersectionsExpanded intersections p
%% MINIMUM DISTANCE

%add arm information to every point of sholl and skell intersection
for rSel=1:numel(fieldnames(sholl))
    points = getfield( sholl, ['r', num2str(rSel)]);
    for l=1:size(getfield( sholl, ['r', num2str(rSel)]),1)
        point = points(l,:);
        
        sphere = sqrt((cc-double(point(1,1))).^2+(rr-double(point(1,2))).^2+(pp-double(point(1,3))).^2) < (3); %logical external sphere area
        SphSholl = J_skel + sphere;
        x= []; y= []; z= [];
        [y,x,z] = ind2sub(size(SphSholl), find(SphSholl==2)) ;     %search for intersections
        intersections = [x y z];
        
        founded = false;
        
        while ~founded
            for j=1:size(intersections,1) %for every intersection (j)
                for i=1:numel(fieldnames(tree))-2 %for every arm (i)
                    if ismember(intersections(j,:), getfield(tree, ['arm',num2str(i)], 'points'),'rows') %ceck if intersection-j is in arm-i
                        rowChange = getfield(sholl, ['r', num2str(rSel)]);
                        rowChange(l,4) = i;
                        sholl = setfield(sholl, ['r', num2str(rSel)], rowChange ); %write in 4th columnn of sholl the armNumb of the point
                        
                        founded = true;
                    end
                end
            end
        end
    end
end

clear rSel points sphere SphSholl x y z intersections founded rowChange l

%search minimum distance
armSel = 1;
minDist = r_start;
pointPrec = sholl.r1(1,1:3);
ptiMinDist = [];

%r : #radius of sholl
%i : iesimo point of r-esimo sholl radius
for r=2:numel(fieldnames(sholl)) %select sholl radius points
    
    for i=1:size(getfield(sholl, ['r' num2str(r)]) ,1) %cicle points in current sholl radius
        
        completed = false;
        arm = getfield(sholl, ['r' num2str(r)], {i,4});
        node = getfield(tree.hierarchy, {arm,1});
        
        while ~completed
            
            
            if armSel == 1          %if point is on 1th arm
                rowChange = getfield(sholl, ['r', num2str(r)]);
                rowChange(i,5) = 1;
                sholl = setfield(sholl, ['r', num2str(r)], rowChange ); %write 1 in 5th of sholl
                
                completed = true;
                
            elseif arm == armSel    %if arm of the point is a parent of the arm of the last point
                rowChange = getfield(sholl, ['r', num2str(r)]);
                rowChange(i,5) = 1;
                sholl = setfield(sholl, ['r', num2str(r)], rowChange ); %write 1 in 5th of sholl
                
                completed = true;
                
            elseif arm == 1         %if point doesn't belong to the right tree
                rowChange = getfield(sholl, ['r', num2str(r)]);
                rowChange(i,5) = 0;
                sholl = setfield(sholl, ['r', num2str(r)], rowChange ); %write 0 in 5th of sholl
                
                completed = true;
                
            end
            
            arm = find(tree.hierarchy(:,2)' == node);
            node = getfield(tree.hierarchy, {arm,1});
        end
    end
    
    %minimum distance
    accepted = getfield(sholl, ['r', num2str(r)]);
    accepted = accepted(:,5);
    
    i=find( accepted ==1);
    
    if isempty(i)
        break;
    end
    
    dist = [];
    for l=1:size( i ,1)
        pointSuc = getfield(sholl, ['r', num2str(r)], { i(l),1:3});
        dist(l) = sqrt( sum(sum([pointPrec; -pointSuc]).^2) );
    end
    [mindist, indmindist] = min(dist);
    
    minDist = minDist + mindist;
    
    armSel = getfield(sholl, ['r', num2str(r)], { i(indmindist) ,4});
    
    ptiMinDist = [ptiMinDist ; getfield(sholl, ['r', num2str(r)], { i(indmindist),1:3})];
    
    if multiPlot
        figure(multiData);
        subplot(2,2,2);
    else
        figure(f2);
    end
    
    scatter3( getfield(sholl, ['r', num2str(r)], { i(indmindist),1}) , getfield(sholl, ['r', num2str(r)], { i(indmindist),2}) , getfield(sholl, ['r', num2str(r)], { i(indmindist),3}), 'm');
end

disp(['Minimum distance: ' num2str(minDist) ' in arm' num2str(armSel)]);

%clear mindist pointSuc pointPrec accepted rowChange completed arm node l arm completed i r


%% CONE

% neuron central axis
bary = sum(POINTS)/size(POINTS,1); %barycenter
vaxis = (bary - [soma_center(2) soma_center(1) soma_center(3)]) / norm(bary - [soma_center(2) soma_center(1) soma_center(3)]);

% projection
vPuntoMaxDist = [max_estr_coord(2), max_estr_coord(1), max_estr_coord(3)] - [soma_center(2), soma_center(1), soma_center(3)];
hCone = (dot(vaxis, vPuntoMaxDist) / (norm(vaxis))^2) * vaxis;

tetaMax = 0;
for i=1:size(POINTS,1)
    v = [POINTS(i,1),POINTS(i,2),POINTS(i,3)] - [soma_center(2),soma_center(1),soma_center(3)];
    teta =  atan2d(norm(cross(vaxis,v)),dot(vaxis,v));
    if teta > tetaMax
        tetaMax = teta;
        vmaxPoint = v;
        maxPoint = POINTS(i,:);
    end
end

rCone = norm(hCone)* tand(tetaMax);
apoCone = norm(hCone) / cosd(tetaMax);

clear v teta

%% SHOW CONE
if multiPlot
    figure(multiData);
    subplot(2,2,1);
else
    figure(f1);
end
hold on;

% figure rotation
angleX = (90-atan2d(norm(cross(vaxis,[1 0 0])),dot(vaxis,[1 0 0])));
angleY = -(90-atan2d(norm(cross(vaxis,[0 1 0])),dot(vaxis,[0 1 0])));
%angleZ = atan2d(norm(cross(vaxis,[0 0 1])),dot(vaxis,[0 0 1]));

TRX = [1 0 0 0; 0 cosd(angleY) -sind(angleY) 0; 0 sind(angleY) cosd(angleY) 0; 0 0 0 1];
TRY = [cosd(angleX) 0 sind(angleX) 0 ; 0 1 0 0; -sind(angleX) 0 cosd(angleX) 0; 0 0 0 1];
%TRZ = [ cosd(angleZ) -sind(angleZ) 0;   sind(angleZ) cosd(angleZ) 0             ; 0 0 1];

TTR = [1 0 0 soma_center(2); 0 1 0 soma_center(1); 0 0 1 soma_center(3); 0 0 0 1];

% figure('Name', 'CONE');
% patch('Faces', neuJ.faces, 'Vertices', neuJ.vertices,'FaceColor','yellow', 'EdgeColor','none');
% axis equal; camlight; camlight(-80, -10); lighting phong; hold on; alpha 0.3; box on; grid on; view(3);
% xlabel('x_L');  ylabel('y_L');  zlabel('z_L');
% hold on;

R = rCone; %radius
H = norm(hCone); %height
N = 50; %number of points to define the circumference
[x, y, z] = cylinder([0 R], N);    %cone radius rCone, h=1

%scalatura
z = z*H;

%rotazione
ptiCono = [x(2,:); y(2,:); z(2,:);ones(1,N+1)]; %pti cone base parametrics
ptiCono = TTR * TRX * TRY  * ptiCono; %pti cone base rotated (column : x y z)
x(2,:) = ptiCono(1,:);
y(2,:) = ptiCono(2,:);
z(2,:) = ptiCono(3,:);

x(1,:) = soma_center(2);
y(1,:) = soma_center(1);
z(1,:) = soma_center(3);

mesh(x, y, z,'LineWidth', 0.1, 'EdgeAlpha', 0.3, 'FaceAlpha', 0.2);
%alpha(0.1);
quiver3( soma_center(2), soma_center(1), soma_center(3), hCone(1), hCone(2), hCone(3), 'r');

clear x y z TRX TRY TRZ TTR R H N i ptiCono

%% Fractal Dimension
cd('FractalDimension');
if multiPlot
    subplot(2,2,4);
else
    f4 = figure('Name', 'Fractal Dimension');
end
hold on; view(3); box on; grid on;
[n, r] = boxcount(J_skel, 'slope');

s=-gradient(log(n))./gradient(log(r));
der = diff(s);
cost = der<0.1;

r_range = r(cost);
df_mean = mean(s(cost));
df_std = std(s(cost));

cd ..

datamatrix.MeanFractalDimension = df_mean;
datamatrix.StDFractalDimension = df_std;
datamatrix.RangeFractalDimension = r_range;

dm(1,24) = df_mean;
dm(1, 25) = df_std;
% dm(1, 24) = mean(r_range);


%%
if multiPlot
    set(multiData, 'units','normalized','outerposition',[0 0 1 1]);
end
disp('CODE COMPILED SUCCESSFYLLY');