# setup function for tsunami modelling
using Plots, ColorSchemes
using GaussianRandomFields
plotly()

include("setup.jl")
include("simulate.jl")

######################
## Setup parameters ##
######################

# number of grid points in x and y directions
x_size= 300
y_size = 300

# distance between grid pointsin x and y directions
dx = 2000
dy = 2000

# timestep
dt = 1.0

# Gravitational constant
grav = 9.80665

# Depth of sea, uniform for now
depth = 3000
depth_m = ones(y_size, x_size).*depth

# Parameters for boundary conditions, as in Cerjan (1985)
boundary_damping = 0.015
boundary_size = 20

# radius of non-zero height area in iniital conditions
initial_spread = 30000

# number of timesteps between each snapshot for plotting
snapshot_frequency = 10

# total time
total_time = 2000

################
## Simulation ##
################
                
boundary_m = boundary_conditions(boundary_size, x_size, y_size, boundary_damping)
height_m = initial_height(x_size, y_size, dx, dy, initial_spread)
# height_m = sample(grf)
# x_vel_m = zeros(y_size, x_size)
# y_vel_m = zeros(y_size, x_size)
cov = CovarianceFunction(2, Matern(1/4, 3/4))
pts = range(0, stop=1, length=300)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts)
x_vel_m = sample(grf)/40
y_vel_m = sample(grf)/40

heatmap(height_m)

height_snapshots = Array{Float64}[]
for i in 1:total_time
    if i % snapshot_frequency == 0
        push!(height_snapshots, height_m)
    end
    new_x_vel_m, new_y_vel_m, new_height_m = timestep(height_m, x_vel_m, y_vel_m, dx, dy, dt, depth_m, boundary_m, grav)
    global x_vel_m = new_x_vel_m
    global y_vel_m = new_y_vel_m
    global height_m = new_height_m
end

##############
## Plotting ##
##############

# heatmap over time saved as gif
anim = @animate for i in 1:size(height_snapshots)[1]
    heatmap(height_snapshots[i], clim=(-0.3, 0.3), c=cgrad([:blue, :white, :orange]))
end
gif(anim, "heatmap_grf_initial_cond_random_process.gif", fps = 15)

# surface plot over time saved as gif
anim = @animate for i in 1:size(height_snapshots)[1]
    plot(height_snapshots[i], zlims=(-0.3,0.3), st=:surface, clim=(-0.3, 0.3), legend = :none, c=cgrad([:red, :aqua, :blue]))
end
gif(anim, "surface_grf_initial_cond_random_process.gif", fps = 15)

cov = CovarianceFunction(2, Matern(1/4, 3/4))

pts = range(0, stop=1, length=300)

grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts)

sample(grf)



