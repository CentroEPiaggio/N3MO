%% SEARCH ARM SKEL

%identificazione dei rami
[cc, rr, pp] = meshgrid(1:size(J_skel, 1), 1:size(J_skel, 2), 1:size(J_skel, 3));

% creazione struct memorizzazione punti
tree = struct;
tree.arm1 = struct;
tree.arm1.points = struct;
tree.arm1.points = [];

%first point outside Soma
r=soma_radius_MAX+3;
sphere1 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r); %logical internal sphere area
sphere2 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r+1); %logical external sphere area
sphere = sphere2-sphere1;
SphSholl = J_skel + sphere;                              %merge Skel with Thoroid -> obj value = 2, empty area = 1

[y1,x1,z1] = ind2sub(size(SphSholl), find(SphSholl==2) ) ;

%second point outside Soma
r=soma_radius_MAX+4;
sphere1 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r); %logical internal sphere area
sphere2 = sqrt((cc-double(soma_center(1,2))).^2+(rr-double(soma_center(1,1))).^2+(pp-double(soma_center(1,3))).^2)<= (r+1); %logical external sphere area
sphere = sphere2-sphere1;
SphSholl = J_skel + sphere;                              %merge Skel with Thoroid -> obj value = 2, empty area = 1

[y2,x2,z2] = ind2sub(size(SphSholl), find(SphSholl==2) ) ;

addpath(genpath('sphere'));

% % % questa è una bozza (funziona solo fino alla prima intersezione)
% % first_point = [x1, y1, z1];
% % sphere_center = [x2, y2, z2];
% % num_cycle = 1;
% % 
% % 
% % [arm_points, new_arm_points] = findArmPnts(cc, rr, pp, J_skel, sphere_center, first_point);
% % tree.arm1.points = [arm_points];
% % hold on
% % for i=1:size(arm_points,1)
% %     scatter3(arm_points(i,1),arm_points(i,2),arm_points(i,3));
% % end

% nuovo tentativo (su tutto lo Skel)
first_point = [x1, y1, z1];
first_center = [x2, y2, z2];

sphere_center = first_center;

num_arm = 1;
num_cycle = 1;

signed_points = [];

condition = false;

while ~condition
    
        unsigned_points = [];
        breach_points = [];
		
        for i=1:num_cycle
		
		    A = sphere_center(i,:);
			B = first_point(i,:);
			
            [arm_points, new_arm_points] = findArmPnts(cc, rr, pp, J_skel, A, B);
			
			tree =  setfield(tree, ['arm',num2str(num_arm)], 'points', arm_points);
			% facoltativo signed
			signed_points = [signed_points; arm_points];
			unsigned_points = [unsigned_points; new_arm_points];
			
			num_breach = size(new_arm_points,1);
			
			for i=1:num_breach
			
			    breach_points = [breach_points; arm_points(end,:)];
			
			end
			
			% vettori di appoggio
			clear arm_points new_arm_points
			
			num_arm = num_arm + 1;
			
        end
		
        if isempty(unsigned_points)
            condition = true;

        else num_cycle = size(unsigned_points,1);
          
             clear sphere_center first_point
			 
             for i=1:num_cycle
                 sphere_center(i,:) = unsigned_points(i,:);
             end
			 
			 first_point = breach_points;
			 
			 clear unsigned_points breach_points
        end
        
		
end