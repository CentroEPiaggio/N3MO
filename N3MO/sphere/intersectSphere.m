function [pnts_found, num] = intersectSphere(cc, rr, pp, J_skel, r, sphere_center)

x = []; y = []; z = [];

sphere1 = sqrt((cc-double(sphere_center(1))).^2 + (rr-double(sphere_center(2))).^2 + (pp-double(sphere_center(3))).^2) == (0); %logical internal sphere area
sphere2 = sqrt((cc-double(sphere_center(1))).^2 + (rr-double(sphere_center(2))).^2 + (pp-double(sphere_center(3))).^2) < (r+1); %logical external sphere area
sphere = sphere2 - sphere1;

SphSholl = J_skel + sphere;

[y,x,z] = ind2sub(size(SphSholl), find(SphSholl==2));
pnts_found = [x,y,z];
num = size(pnts_found,1);

end
