include("simulate.jl")
include("utils.jl")
include("optimal_filter.jl")

function bootstrap(num_particles, l, height_sigma, vel_sigma, gr, min_ess_ratio, observations)
    
    height_m = zeros(gr.p+1, gr.q+1)
    x_vel_m = zeros(gr.p+1, gr.q+1)
    y_vel_m = zeros(gr.p+1, gr.q+1)

    particles = [
        [[
            height_m + sample_uncond_grf(l, height_sigma, gr),
            x_vel_m + sample_uncond_grf(l, vel_sigma, gr),
            y_vel_m + sample_uncond_grf(l, vel_sigma, gr)
        ]] 
        for i in 1:num_particles
        ]
    log_weights = [log.([1/num_particles for i in 1:num_particles])]
    ess_log = []
    min_ess_ratio = 0.5

    for obs in observations
        new_log_weights = []
        for (p, w) in zip(particles, log_weights[end])
            global prop_height = deepcopy(p[end][1])
            global prop_x_vel = deepcopy(p[end][2])
            global prop_y_vel = deepcopy(p[end][3])
            for t in 1:Int(obs_dt/dt)
                new_x_vel, new_y_vel, new_height = timestep(
                    prop_height,
                    prop_x_vel,
                    prop_y_vel,
                    dx,
                    dy,
                    dt,
                    depth_m,
                    boundary_m,
                    grav
                )
                global prop_height = new_height
                global prop_x_vel = new_x_vel
                global prop_y_vel = new_y_vel
            end
            global prop_height += sample_uncond_grf(l, height_sigma, gr)
            global prop_x_vel += sample_uncond_grf(l, vel_sigma, gr)
            global prop_y_vel += sample_uncond_grf(l, vel_sigma, gr)
            push!(
                p,
                [
                    prop_height,
                    prop_x_vel,
                    prop_y_vel
                ]
            )
            log_prob_prop = logpdf(error_dist, A1(vec(prop_height), stat_locs, gr).-obs)
            push!(new_log_weights, w[end] + log_prob_prop)
        end
        new_log_weights = normalise_log_weights(new_log_weights)
        push!(log_weights, new_log_weights)
        effective_ss = ess(exp.(log_weights[end]))
        push!(ess_log, effective_ss)

        if effective_ss < min_ess_ratio*num_particles
            resample_probs = exp.(log_weights[end])
            resample_dist = Categorical(resample_probs)
            draws = rand(resample_dist, length(particles))
            old_particles = deepcopy(particles)
            for i in 1:length(particles)
                particles[i][end] = old_particles[draws[i]][end]
            end
            log_weights[end] = log.([1/length(particles) for i in 1:length(particles)])
        end
    end
    return particles, log_weights, ess_log
end

function optimal_filter(num_particles, l, height_sigma, vel_sigma, gr, min_ess_ratio, observations)
    height_m = zeros(gr.p+1, gr.q+1)
    x_vel_m = zeros(gr.p+1, gr.q+1)
    y_vel_m = zeros(gr.p+1, gr.q+1)

    particles = [
        [[
            height_m + sample_uncond_grf(l, height_sigma, gr),
            x_vel_m + sample_uncond_grf(l, vel_sigma, gr),
            y_vel_m + sample_uncond_grf(l, vel_sigma, gr)
        ]] 
        for i in 1:num_particles
        ]
    log_weights = [log.([1/num_particles for i in 1:num_particles])]
    ess_log = []

    for obs in observations
        for p in particles
            global prop_height = deepcopy(p[end][1])
            global prop_x_vel = deepcopy(p[end][2])
            global prop_y_vel = deepcopy(p[end][3])
            for t in 1:Int(obs_dt/dt)
                new_x_vel, new_y_vel, new_height = timestep(
                    prop_height,
                    prop_x_vel,
                    prop_y_vel,
                    dx,
                    dy,
                    dt,
                    depth_m,
                    boundary_m,
                    grav
                )
                global prop_height = new_height
                global prop_x_vel = new_x_vel
                global prop_y_vel = new_y_vel
            end
            global prop_height += sample_uncond_grf(l, height_sigma, gr)
            global prop_x_vel += sample_uncond_grf(l, vel_sigma, gr)
            global prop_y_vel += sample_uncond_grf(l, vel_sigma, gr)
            push!(
                p,
                [
                    prop_height,
                    prop_x_vel,
                    prop_y_vel
                ]
            )
        end
        heights = [particles[i][end][1] for i=1:length(particles)]
        global optimal_heights = sample_heights(
            heights,
            l,
            height_sigma,
            stat_locs,
            obs,
            sigma_obs,
            gr
        )
        for i in 1:length(particles)
            particles[i][end][1] = optimal_heights[i]
        end
        log_prob_props = calc_log_weights(
            optimal_heights, gr, stat_locs, l, height_sigma, sigma_obs, obs
        )
        new_log_weights = normalise_log_weights(log_weights[end] + log_prob_props)
        push!(log_weights, new_log_weights)
        effective_ss = ess(exp.(log_weights[end]))
        push!(ess_log, effective_ss)

        if effective_ss < min_ess_ratio*num_particles
            resample_probs = exp.(log_weights[end])
            resample_dist = Categorical(resample_probs)
            draws = rand(resample_dist, length(particles))
            old_particles = deepcopy(particles)
            for i in 1:length(particles)
                particles[i][end] = old_particles[draws[i]][end]
            end
            log_weights[end] = log.([1/length(particles) for i in 1:length(particles)])
        end
    end
    return particles, log_weights, ess_log
end