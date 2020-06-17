function positions = generate_positions(Na, Nb, Nc, m, p, h, seed)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % INPUTS: Number of cell type A, B, C (Na, Nb, Nc), width, length, and
    % height of the network (m, p, h), and random seed (seed)
    
    % OUTPUT: (Na+Nb+Nc,4) array of gid, x, y, and z coordinates.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rng(seed);

    % minimum inter-soma distance in microns
    min_dist = 20;

    % possible x,y,z coordinates (grid)
    [x, y, z] = meshgrid(0:min_dist:m,0:min_dist:p,0:min_dist:h);
    poss_coords = [x(:) y(:) z(:)];

    positions = zeros(Na + Nb + Nc, 4);
    positions(:,1) = linspace(0,Na+Nb+Nc-1,Na+Nb+Nc); %gids
    positions(:,2:4) = datasample(poss_coords, Na+Nb+Nc, 'Replace', false);
end
