function [ ] = noiseDelete ( neuronMap )
% ELIMINATE ERROR DATA IF 1 NEURON MAP (OBJECTS)
neuJ_label = bwlabeln(neuronMap.neuron);
neuJ_Obj = bwconncomp(neuJ_label);
neuJ_props = regionprops(neuJ_Obj, 'basic');

if neuJ_Obj.NumObjects ~= 1
    areaMax = max(neuJ_props.Area);
    id_areaMax = find([neuJ_props.Area] == areaMax);
    
    for id=1:neuJ_Obj.NumObjects
        if id==id_areaMax
            continue
        end
        neuronMap.neuron(neuJ_Obj.PixelIdxList{1, id}) = 0; %0 logical to the external obj
    end
    
    %clear id id_areaMax areaMax neuJ_label neuJ_Obj neuJ_props
end


end
