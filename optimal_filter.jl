using Distributions
using LinearAlgebra
using FFTW

struct grid
    p::Int
    q::Int
    P::Float64
    Q::Float64
end

function coord_to_1d(z, grid)
    return Int(z[2]*(grid.q+1) + z[1]+1)
end

function r(x, y, l, sigma_model)
    return (sigma_model^2)*exp(-sqrt((x/l)^2 + (y/l)^2))
#     return (sigma_model^2)*exp(-(abs(x)+abs(y))/(2.0*l))
end

function r_bar(x, y, l, sigma_model, grid)
    return r(
        min(x, 2*grid.P - x),
        min(y, 2*grid.Q - y),
        l,
        sigma_model
    )
end

function rho_bar(l, sigma_model, grid)
    dx = grid.P/grid.p
    dy = grid.Q/grid.q
    
    out = [
        r_bar(i*dx,j*dy, l, sigma_model, grid)
        for j in 0:(2*grid.q-1)
        for i in 0:(2*grid.p-1)
    ]

    return out
end

function W(v, grid)
    ft = ifft(reshape(v, (2*grid.p, 2*grid.q)))
    return sqrt(4*grid.p*grid.q)*vec(ft)
end

function WH(v, grid)
    ft = fft(reshape(v, (2*grid.p, 2*grid.q)))
    return (1.0/sqrt(4*grid.p*grid.q))*vec(ft)
end

function sample_uncond_grf(l, sigma_model, grid)
    p = grid.p
    q = grid.q

    bar_rho = rho_bar(l, sigma_model, grid)
    lambda = sqrt(4*p*q)*Diagonal(real.(W(bar_rho, grid)))
    
    norm_dist = Normal(0,1)
    e1 = rand(norm_dist,4*p*q) + rand(norm_dist,4*p*q)im
    bar_z1 = W(lambda^(1/2)*e1, grid)
    index = zeros(Int32, (p+1)*(q+1))
    
    for j in 0:q
        index[(j*(p+1)+1):((j+1)*(p+1))] = collect((j*2*p+1):(j*2*p+p+1))
    end
    return reshape(real.(bar_z1[index]), (grid.p+1, grid.q+1))
end

function R_12_bar(grid, locs, l, sigma_model)
    dx = grid.P/grid.p
    dy = grid.Q/grid.q
    
    R_bar = Array{Float64}(undef, 4*grid.p*grid.q, 0)
    for loc in locs
        col = [
            r_bar((i - loc[1])*dx, (j- loc[2])*dy, l, sigma_model, grid)
            for j in 0:(2*grid.q-1)
            for i in 0:(2*grid.p-1) 
        ]
        R_bar = hcat(R_bar, col)
    end
    return R_bar
end

function R_12(grid, locs, l, sigma_model)
    dx = grid.P/grid.p
    dy = grid.Q/grid.q
    
    R = Array{Float64}(undef, (grid.p+1)*(grid.q+1), 0)
    for loc in locs
        col = [
            r((i - loc[1])*dx, (j- loc[2])*dy, l, sigma_model)
            for j in 0:grid.q
            for i in 0:grid.p
        ]
        R = hcat(R, col)
    end
    return R
end
    
function R_22(grid, locs, l, sigma_model)
    dx = grid.P/grid.p
    dy = grid.Q/grid.q
    
    R = Array{Float64}(undef, length(locs), length(locs))
    
    for i in 1:length(locs)
        x_i = locs[i][1]
        y_i = locs[i][2]
        for j in 1:length(locs)
            x_j = locs[j][1]
            y_j = locs[j][2]
            
            R[i,j] = r(
                abs(x_j - x_i)*dx,
                abs(y_j - y_i)*dy,
                l,
                sigma_model
            )
        end
    end
    return R
end

function A1(v, stat_locs, grid)
    output = zeros(length(stat_locs))
    for i in 1:length(stat_locs)
        output[i] = v[coord_to_1d(stat_locs[i], grid)]
    end
    return output
end

function A1_T(v, stat_locs, grid)
    output = zeros((grid.p+1)*(grid.q+1))
    for i in 1:length(v)
        output[coord_to_1d(stat_locs[i], grid)] = v[i]
    end
    return output
end

function Sig_1(v, l, sigma_model, grid)
    p = grid.p
    q = grid.q
    n1 = (p+1)*(q+1)
    bar_n1 = 4*p*q
    
    bar_v = zeros(bar_n1)
    
    for j in 0:q
        bar_v[(j*2*p+1):(j*2*p+p+1)] = v[(j*(p+1)+1):((j+1)*(p+1))]
    end
    
    bar_rho = rho_bar(l, sigma_model, grid)
    lambda = sqrt(bar_n1)*Diagonal(real.(W(bar_rho, grid)))
    bar_Sig_v = W(lambda*WH(bar_v, grid), grid)
    
    index = zeros(Int32, n1)
    
    for j in 0:q
        index[(j*(p+1)+1):((j+1)*(p+1))] = collect((j*2*p+1):(j*2*p+p+1))
    end

    Sig_v = bar_Sig_v[index]
    return real.(Sig_v)
end

function calc_means(heights, l, sigma_model, stat_locs, obs, sigma_obs, grid)
    
    R22 = R_22(grid, stat_locs, l, sigma_model)
    R22_S = R22 + Diagonal(fill(sigma_obs^2, length(stat_locs)))
    inv_R22_S = inv(R22_S)

    mu2_1 = Sig_1((sigma_obs^-2)*A1_T(obs, stat_locs, grid), l, sigma_model, grid)
    mu2_2 = Sig_1((sigma_obs^-2)*A1_T(inv_R22_S*(R22*obs), stat_locs, grid), l, sigma_model, grid)
    mu2 = mu2_1 - mu2_2

    output = []
    
    for h_prof in heights
        height_prof = vec(h_prof)
        inv_R22_S*A1(height_prof, stat_locs, grid)
        mu1_2 = Sig_1(
            A1_T(inv_R22_S*A1(height_prof, stat_locs, grid), stat_locs, grid),
            l,
            sigma_model,
            grid
        )
        mu1 = height_prof - mu1_2
        
        mu = mu1 + mu2
        push!(output, reshape(mu, (grid.p+1, grid.q+1)))
    end
    return output
end

function sample_heights(heights, l, sigma_model, stat_locs, obs, sigma_obs, grid)
    p = grid.p
    q = grid.q
    
    bar_rho = rho_bar(l, sigma_model, grid)
    lambda = sqrt(4*p*q)*Diagonal(real.(W(bar_rho, grid)))

    bar_R12 = R_12_bar(grid, stat_locs, l, sigma_model)

    WHbar_R12 = zeros(Complex, 4*p*q, length(stat_locs))
    
    for i in 1:length(stat_locs)
        WHbar_R12[:,i] = WH(bar_R12[:,i], grid)
    end
    
    KH = lambda^(-1/2)*WHbar_R12
    
    K = KH'
    R22 = R_22(grid, stat_locs, l, sigma_model)
    R22_S = R22 + Diagonal(fill(sigma_obs^2, length(stat_locs)))
    inv_R22_S = inv(R22_S)
    
    L = cholesky(real.(R22_S-K*KH)).L

    R12 = R_12(grid, stat_locs, l, sigma_model)
    R12_invR22 = R12*inv_R22_S

    index = zeros(Int32, (p+1)*(q+1))
    
    for j in 0:q
        index[(j*(p+1)+1):((j+1)*(p+1))] = collect((j*2*p+1):(j*2*p+p+1))
    end
    
    means = calc_means(heights, l, sigma_model, stat_locs, obs, sigma_obs, grid)

    samples = []
    
    for i in 1:Int(length(heights)/2)
        norm_dist = Normal(0,1)
        e1 = rand(norm_dist,4*p*q) + rand(norm_dist,4*p*q)im
        e2 = rand(norm_dist,length(stat_locs)) + rand(norm_dist,length(stat_locs))im
        
        bar_z1 = W(lambda^(1/2)*e1, grid)
        z2 = K*e1 + L*e2
        
        Rbar_z = real.([bar_z1; z2])
        Ibar_z = imag.([bar_z1; z2])
        
        Proj1 = [ Rbar_z[index]; Rbar_z[(4*p*q+1):end]]
        Proj2 = [ Ibar_z[index]; Ibar_z[(4*p*q+1):end]]
        
        ##Z1_C_Yobs_1 = muZ1_C_Yobs + Proj1[1:n1] - R12_invR22*Proj1[(n1+1):end]
        Z11 = Proj1[1:(p+1)*(q+1)] - R12_invR22*Proj1[((p+1)*(q+1)+1):end]
        
        ##Z1_C_Yobs_2 = muZ1_C_Yobs + Proj2[1:n1] - R12_invR22*Proj2[(n1+1):end]
        Z12 = Proj2[1:(p+1)*(q+1)] - R12_invR22*Proj2[((p+1)*(q+1)+1):end]
        
        push!(samples, means[2*i-1] + reshape(Z11, (p+1,q+1)))
        push!(samples, means[2*i] + reshape(Z12, (p+1,q+1)))
    end
    return samples
end

function calc_log_weights(heights, grid, stat_locs, l, sigma_model, sigma_obs, obs)
    R22 = R_22(grid, stat_locs, l, sigma_model)
    R22_S = R22 + Diagonal(fill(sigma_obs^2, length(stat_locs)))
    inv_R22_S = inv(R22_S)
    
    prop_index = [coord_to_1d(x, grid) for x in stat_locs]
    
    log_weights = []
    for i in 1:length(heights)
        props = vec(heights[i])[prop_index]
        z = obs - props
        push!(log_weights, -0.5*((z')*inv_R22_S*z))
    end
    return(log_weights.-maximum(log_weights))
end  

