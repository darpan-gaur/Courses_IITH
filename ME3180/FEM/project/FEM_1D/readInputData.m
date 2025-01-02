function [nnode,nelem,nen,coord,connect,props,bodyforce,load,fixnodes] = readInputData(inFilename)
%%
%  Extract material and geometric properties
    cellarray=textscan(inFilename,'%s');
    E=str2num(cellarray{1}{3});
    A=str2num(cellarray{1}{5});
    props(1) = E;
    props(2) = A;
%%
%  Extract no. nodes and nodal coordinates
%
    nnode=str2num(cellarray{1}{7});
    dum=9;
    coord=zeros(nnode,1);
    for i=1:nnode
        coord(i,1) = str2num(cellarray{1}{dum});
        dum=dum+1;
    end
%%
%   Extract no. elements and connectivity
%
    dum=dum + 1;
    nelem=str2num(cellarray{1}{dum});
    dum=dum + 2;
    nen=str2num(cellarray{1}{dum});
    connect = zeros(nelem,nen);
    dum = dum + 2;
    for i = 1 : nelem
        for j = 1 : nen
            connect(j,i) = str2num(cellarray{1}{dum});
            dum=dum+1;
        end
    end

%%
%   Extract no. nodes with prescribed displacements and the prescribed displacements
%
    dum = dum + 1;
    nfix=str2num(cellarray{1}{dum});
    dum = dum + 4;
    fixnodes = zeros(nfix,3);
    for i = 1 : nfix
        for j = 1 : 3
            fixnodes(i,j) = str2num(cellarray{1}{dum});
            dum=dum+1;
        end
    end
%%
%   Extract no. loaded element faces, with the loads
%
    dum = dum + 1;
    ndload=str2num(cellarray{1}{dum});
    dum=dum + 4;
    load = zeros(ndload,3);
    for i = 1 : ndload
    for j=1:3
        load(i,j)=str2num(cellarray{1}{dum});
        dum=dum+1;
    end
    end
    bodyforce = str2num(cellarray{1}{dum+1});
end