#functions for setup of tsunami model
using GaussianRandomFields
function boundary_conditions(boundary_size, x_size, y_size, damping)
    boundary_m = ones(y_size, x_size)
    for i in 1:y_size
        for j in 1:x_size
            north_dist = i - 1
            south_dist = y_size - i
            west_dist = j - 1
            east_dist = x_size - j
            dist_to_bound = min(north_dist, south_dist, west_dist, east_dist)
            if dist_to_bound < boundary_size
                boundary_m[i,j] = exp(-(damping*(boundary_size-dist_to_bound))^2)
            end
        end
    end
    return boundary_m
end


function initial_height(x_size, y_size, dx, dy, spread)
    cov = CovarianceFunction(2, Matern(1/4, 3/4))
    pts = range(0, stop=1, length=300)
    grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts)
    
    height_m = zeros(y_size, x_size)
    
    i_0 = y_size/4
    j_0 = x_size/4
    
    for i in 1:y_size
        if -spread <= (i - i_0)*dy && (i - i_0)*dy <= spread
            hy = (1 + cos(pi * (i - i_0) * dy / spread)) / 2.0
        else
            hy = 0
        end
                
        for j in 1:x_size
            if -spread <= (j - j_0)*dx && (j - j_0)*dx <= spread
                hx = (1 + cos(pi * (j - j_0) * dx / spread)) / 2.0
            else
                hx = 0
            end
            height_m[i,j] = hy * hx
        end
    end
    return height_m + sample(grf)/20
    # return height_m
end