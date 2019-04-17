function plottacono(center,h,r)
% keyboard
[X,Y,Z]=cylinder([0,r]);
X = X+center(1);
Y = Y+center(2);
Z(2,:) = Z(2,:)*h;
Z = Z+center(3);
surf(X,Y,Z)
end