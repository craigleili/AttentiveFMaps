% https://github.com/LIX-shape-analysis/GeomFmaps/tree/master/MATLAB_Tools

clear all
addpath('./Utils/');

sourcepath = '../exp/data/';
dataset = 'SHREC_r'; %['FAUST_r' 'SCAPE_r' 'SHREC_r']  %name of the dataset
filetype = 'off';

datafolder = [sourcepath dataset '/shapes/'];
files = dir(fullfile([datafolder ''], ['*.' filetype]));
n_files = length(files);

max_spec_dim = 100;

for pts = 1:n_files

    Vname = [datafolder files(pts).name];
    sourcename = split(files(pts).name, '.');
    sourcename = sourcename{1};
    fprintf('%s\n', sourcename);

    S1 = read_off_shape(Vname);
    S1 = MESH.compute_LaplacianBasis(S1, max_spec_dim);
    sqrt_area = S1.sqrt_area;
    geodist = MESH.compute_geodesic_dist_matrix(S1);

    savefolder_geodist = [sourcepath dataset '/geodist/'];
    if ~exist(savefolder_geodist, 'dir')
       mkdir(savefolder_geodist)
    end
    save([savefolder_geodist sourcename '.mat'],'geodist', 'sqrt_area');
end
