function [arm_points, new_arm_points] = findArmPnts(cc, rr, pp, J_skel, sphere_center, first_point)

% vettori memorizzazione punti
arm_points = [];
new_arm_points = [];

% condizione di STOP function
stop_search = true;

% inizializzazione raggio, primo centro, primo punto, contatore punti arm e
% new_arm
next_sphere_center = sphere_center;
arm_points = [arm_points; first_point; sphere_center];

num_arm_points = 2;  % primo punto + primo centro
num_new_arm_points = 0;
r = 1;

while stop_search
    
    [intersections, num_intersect] = intersectSphere(cc, rr, pp, J_skel, r, next_sphere_center);
    
    %%% INIZIO CASE 0
    if num_intersect == 0
        warndlg('An error as occured');
        break
    end
    
    %%% INIZIO CASE 1
    
    if num_intersect == 2
        
        if ~ismember(intersections(1,:), arm_points, 'rows')
            arm_points = [arm_points; intersections(1,:)];  %add new point to tree's arm
            next_sphere_center = intersections(1,:);
        end
        
        if ~ismember(intersections(2,:), arm_points, 'rows')
            arm_points = [arm_points; intersections(2,:)];  %add new point to tree's arm
            next_sphere_center = intersections(2,:);
        end
        
        num_arm_points = num_arm_points + 1;
    end
    %%% FINE CASE 1
    
    %%% INIZIO CASE 2
    if num_intersect > 2
        
        for i=1:num_intersect
            % salvo i nuovi punti trovati in un vettore a parte,
            % poichè la loro provenienza è incerta
            if ~ismember(intersections(i,:), arm_points,'rows')
                new_arm_points = [new_arm_points; intersections(i,:)];  % nuovi centri potenzialmente validi per segmentazione rami
                num_new_arm_points = num_new_arm_points + 1;
            end
        end
        
        % se ho trovato almeno 2 punti nuovi allora mi trovo ad una
        % potenziale intersezione:
        % - interruzione funzione
        % - mantengo separati gli arm_points e new_arm_points
        
        if num_new_arm_points >= 2
            disp('intersection found (stop function)')
            break
        end
    end
    %%% FINE CASE 2
    
    %         case num_intersect == 1
    %
    %             %%% terzo caso: 1 intersez -> cerca meglio (r_search come limite imposto dall'utente)
    %             %%% una sola intersezione può significare: buchi nello skel/fine ramo
    %
    %                 r_search_lim = 3;
    %                 undef_arm_points = [];
    %
    %                 %%% ciclo ricerca
    %                 for r=1:r_search_lim
    %
    %                     [intersections, num_intersect] = intersectSphere(r, next_sphere_center);
    %
    %                     % le intersezioni sono memorizzate per un controllo temporaneo (temp)
    %                     temp_intersect(r,:) = intersections;
    %                     temp_num_intersect(r) = num_intersect;
    %
    %                     % controllo che queste intersezioni siano punti nuovi
    %                     for i=1:temp_num_intersect(r)
    %                         if ~ismember(temp_intersect(i,:), arm.points,'rows')
    %                             undef_arm_points(i,:) = temp_intersect(i,:);
    %                         end
    %                     end
    %
    %                     % se trovo anche solo 1 punto nuovo: interrompo la ricerca
    %                     % altrimenti continua
    %                     if isempty(undef_arm_points)
    %                         new_arm_found = true;
    %                         clear temp_intersect temp_num_intersect
    %                     end
    %                 end
    %                 %%% fine ciclo ricerca
    %
    %                 % parte critica: decidere se il nuovo punto/i trovati appartengono allo stesso ramo
    %                 % o fanno parte di un ramo diverso
    if num_intersect == 1
        disp('riprova')
        break
    end
    
    clear intersections num_intersect
    
end
%%% WHILE END

arm_points

new_arm_points

disp('this arm is terminated');
end
%%% function end
