
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