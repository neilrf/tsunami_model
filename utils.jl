function coord_to_1d(z, n_x, n_y)
    return z[2]*n_y - n_x + z[1]
end

function coord_to_2d(z, n_y)
    y = ceil(Int,z/n_y)
    x = mod(z, n_y)
    return (x,y)
end

# function A(station_locations, n_x, n_y)
#     output= zeros(length(station_locations), n_x*n_y)
#     for i in 1:length(station_locations)
#         output[i, coord_to_1d(station_locations[i], n_x, n_y)] = 1
#     end
#     return output
# end


function normalise_log_weights(log_weights)
    max_weight = maximum(log_weights)
    norm_val = max_weight + log(sum(exp.(log_weights .- max_weight)))
    return log_weights .- norm_val
end


function ess(weights)
    sum_squares = sum(weights .* weights)
    return 1/sum_squares
end

function uniform_stations(gr, num, border)
    x_diff = (gr.p-2*border)/(num+1)
    y_diff = (gr.q-2*border)/(num+1)
    
    locs = []
    for i in 1:num
        for j in 1:num
            push!(locs, (Int(floor(border+(x_diff*i))), Int(floor(border+(y_diff*j)))))
        end
    end        
    return locs
end 

function threshold_height(height, height_min)
    return (abs.(height) .> height_min).*height
end