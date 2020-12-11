# functions to simulate tsunami
using GaussianRandomFields

cov = CovarianceFunction(2, Matern(1/4, 3/4))
pts = range(0, stop=1, length=300)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts)

function timestep(height_m, x_vel_m, y_vel_m, dx, dy, dt, depth_m, boundary_m, grav)
    
    x_size = size(height_m)[2]
    y_size = size(height_m)[1]
    
    new_x_vel_m = zeros(y_size, x_size)
    new_y_vel_m = zeros(y_size, x_size)
    new_height_m = zeros(y_size, x_size)
    
    height_dx = zeros(y_size, x_size)
    height_dy = zeros(y_size, x_size)
    
    vel_dx = zeros(y_size, x_size)
    vel_dy = zeros(y_size, x_size)
    
    for i in 1:y_size
        for j in 2:x_size
            height_dx[i,j] = (height_m[i,j] - height_m[i,j-1])/dx
        end
        height_dx[i,1] = height_m[i,1] / dx
    end
    for j in 1:x_size
        for i in 2:y_size
            height_dy[i,j] = (height_m[i,j] - height_m[i-1,j])/dy
        end
        height_dy[1,j] = height_m[1,j] / dy
    end
    
    for i in 1:y_size
        for j in 1:x_size
            new_x_vel_m[i,j] = (x_vel_m[i,j] - grav*depth_m[i,j]*height_dx[i,j]*dt)*boundary_m[i,j]
            new_y_vel_m[i,j] = (y_vel_m[i,j] - grav*depth_m[i,j]*height_dy[i,j]*dt)*boundary_m[i,j]
        end
    end
    
    for i in 1:y_size
        for j in 1:x_size-1
            vel_dx[i,j] = (new_x_vel_m[i,j+1] - new_x_vel_m[i,j]) / dx
        end
        vel_dx[i,x_size] = -new_x_vel_m[i,x_size] / dx
    end
    for j in 1:x_size
        for i in 1:y_size-1
            vel_dy[i,j] = (new_y_vel_m[i+1,j] - new_y_vel_m[i,j]) / dy
        end
        vel_dy[y_size,j] = -new_y_vel_m[y_size,j] / dy
    end
    
    for i in 1:y_size
        for j in 1:x_size
            new_height_m[i,j] = (height_m[i,j] - (vel_dx[i,j] + vel_dy[i,j])*dt) * boundary_m[i,j]
        end
    end
    
    
    
    return new_x_vel_m + sample(grf)/500, new_y_vel_m + sample(grf)/500, new_height_m + sample(grf)/500
    # return new_x_vel_m, new_y_vel_m, new_height_m
end