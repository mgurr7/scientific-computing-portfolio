#import Pkg
#Pkg.activate(@__DIR__)

using SpecialFunctions
using QuadGK
using Plots
using LinearAlgebra
using Printf
using CSV
using DataFrames
using SpecialFunctions
using Base.Threads
using StaticArrays
using ProgressMeter
using LaTeXStrings  



################################################################################
# CONSTANTS AND PARAMETERS
# Safe looping over velocities in case of threads
# Physical constants and dimensionless parameters
if !isdefined(Main, :MaterialConstants)
    include(joinpath(@__DIR__, "material_self.jl"))
end

using .MaterialConstants

if !isdefined(Main, :GreenParams)
    struct GreenParams
        v::Float64
        gamma::Float64
        za_ref::Float64
    end
end

function GreenParams(params::MaterialConstants.MaterialParameters; v_dimless::Real = params.v / params.c)
    return GreenParams(
        Float64(v_dimless),
        Float64(params.gamma / params.omega_p),
        Float64(params.z_a * params.omega_p / params.c))
end

params = MaterialConstants.MaterialParameters()
const SCALING_FACT = inv((2π)^2)
const I3 = Matrix{ComplexF64}(I, 3, 3)
p = GreenParams(params)

omega_start_def = 1e-12
omega_end_def = 1.0
tolerance_omega_def = 1e-6


@inline function bessel_kernel(x, p::GreenParams)
    ax = abs(x)
    u = 2 * p.za_ref * ax

    K0 = besselk(0, u)
    K1 = besselk(1, u)
    K2 = besselk(2, u)

    return K0, K1, K2
end

@inline function bessel_derivative(x, K0, K1, K2, p::GreenParams)
    ax = abs(x)
    K0_d = -2 * ax * K1
    K1_d = -K1 / p.za_ref - 2 * ax * K0
    K2_d = -2 * K2 / p.za_ref - 2 * ax * K1
    return K0_d, K1_d, K2_d
end

@inline function assemble_G(c11, c13, c22, c33)
    return ComplexF64[c11 0 c13;
                      0 c22 0;
                      -c13 0 c33]
end

@inline function antihermitian_part(G::AbstractMatrix)
    return (G - adjoint(G)) / (2im)
end

@inline function hermitian_part(G::AbstractMatrix)
    return (G + adjoint(G)) / 2
end



#################################################################
# FUNCTIONS
# Compute and integrate Green function over the entire real plane
function Green_function_full(ω, p::GreenParams; rtol=1e-5)
    # Main integrand function 
    function integrand(x)
        
        ξp = ω + x * p.v
        ξm = ω - x * p.v
        
        Dp = 1 - 2ξp^2 - 2im * p.gamma * ξp
        Dm = 1 - 2ξm^2 - 2im * p.gamma * ξm
        invD = inv(Dp * Dm)
        Dsum = Dp + Dm
        Ddiff = Dm - Dp

        K0, K1, K2 = bessel_kernel(x, p)

        pp11 = x^2 * Dsum * K0 * SCALING_FACT * invD
        pp13 = -im * x^2 * Ddiff * K1 * SCALING_FACT * invD
        pp22 = x^2 * (K2 - K0) * Dsum * SCALING_FACT * invD / 2
        pp33 = x^2 * (K2 + K0) * Dsum * SCALING_FACT * invD / 2
        return @SVector ComplexF64[pp11, pp13, pp22, pp33]
    end

    # Integrating multiple tensor components together
    vals, _ = quadgk(integrand, 0.0, Inf; rtol=rtol, atol=0.0)
    
    # Compose Green tensor using symmetry
    return assemble_G(vals[1], vals[2], vals[3], vals[4])
end

# Compute and integrate Green function over a defined interval of the momentum
function Green_function_cutoff(ω, p::GreenParams; rtol=1e-5)
    # Main integrand function 
    function integrand(x)

        ξp = ω + x * p.v
        Den = inv(1 - 2ξp^2 - 2im * p.gamma * ξp)
        K0, K1, K2 = bessel_kernel(x, p)

        pp11 = - x^2 * K0 * SCALING_FACT * Den
        pp13 = im * x * abs.(x) * K1 * SCALING_FACT * Den
        pp22 = - x^2 * (K2 - K0) / 2 * SCALING_FACT * Den
        pp33 = - x^2 * (K2 + K0) / 2 * SCALING_FACT * Den   
        return @SVector ComplexF64[pp11, pp13, pp22, pp33]
    end

    # Integrating multiple tensor components together
    vals, _ = quadgk(integrand, -Inf, -ω/p.v; rtol=rtol, atol=0.0)
    
    # Compose Green tensor using symmetry
    return assemble_G(vals[1], vals[2], vals[3], vals[4])
end

# Anti-Hermitian part of the Green tensor over a defined interval of the momentum
function Green_function_I_cutoff(ω, p::GreenParams; rtol=1e-5)
    G = Green_function_cutoff(ω, p; rtol=rtol)
    return antihermitian_part(G)
end

# Gradient of the Hermitian part of the Green tensor 
function Green_function_R_def(x, ω, p::GreenParams; rtol=1e-5)
  
    ξp = ω + x * p.v
    Den = inv(1 - 2ξp^2 - 2im * p.gamma * ξp)
    K0, K1, K2 = bessel_kernel(x, p)
    K0_d, K1_d, K2_d = bessel_derivative(x, K0, K1, K2, p)

    pp11 = x^2 * K0_d * SCALING_FACT * Den
    pp13 = - im * x * abs.(x) * K1_d * SCALING_FACT * Den
    pp22 = x^2 * (K2_d - K0_d) / 2 * SCALING_FACT * Den
    pp33 = x^2 * (K2_d + K0_d) / 2 * SCALING_FACT * Den

    return @SVector ComplexF64[pp11, pp13, pp22, pp33]
end

# Integration of the Gradient of Hermitian part of the Green tensor over momentum
function Green_function_R(ω, p::GreenParams; mode::Symbol = :cutoff, rtol=1e-5)
    integrand = x -> Green_function_R_def(x, ω, p)
    
    if mode == :full
        ε = 1e-30
        val1, _ = quadgk(integrand, -Inf, -ε; atol=0, rtol=1e-5)
        val2, _ = quadgk(integrand, ε, Inf; atol=0, rtol=1e-5)
        vals = val1 + val2

    elseif mode == :cutoff
        vals, _ = quadgk(integrand, -Inf, -ω/p.v; rtol=rtol, atol=0.0)

    else
        error("Unknown mode: $mode. Use :full or :cutoff.")
    end

    G = assemble_G(vals[1], vals[2], vals[3], vals[4])
    return hermitian_part(G)
end

# Polarizability tensor and J tensor computation
function alpha_B(ω::Float64)
    rescaling_factor = params.epsilon_0 * (params.c / params.omega_p)^3
    ω_ratio_sq = (params.omega_a / params.omega_p)^2
    Num = params.alpha_0 / rescaling_factor * ω_ratio_sq
    Den = ω_ratio_sq - ω^2 #- 1im * ω * params.gamma / params.omega_p

    return Num / Den
end

function clean_tensor_parts(T::Matrix{ComplexF64}, thresh::Float64 = 1e-12)
    T_clean = copy(T)
    for i in eachindex(T_clean)
        real_part = real(T_clean[i])
        imag_part = imag(T_clean[i])
        real_part = abs(real_part) < thresh ? 0.0 : real_part
        imag_part = abs(imag_part) < thresh ? 0.0 : imag_part
        T_clean[i] = complex(real_part, imag_part)
    end
    return T_clean
end

function alpha_and_J(ω::Float64, p; show_progress=true)
    
    Gφ = Green_function_full(ω, p)
    GI_theta = Green_function_I_cutoff(ω, p)

    α = alpha_B(ω)
    α_tensor = α * inv(I - α * Gφ)

    α_tensor_I = (α_tensor - adjoint(α_tensor)) / (2im)
    J = α_tensor * GI_theta * adjoint(α_tensor)

    return  clean_tensor_parts(α_tensor_I, 1e-16), clean_tensor_parts(J, 1e-16)
end

# Force components computation
function Force_r(omega, p::GreenParams)
    t0 = time()
    alpha_val_I, J_val = alpha_and_J(omega, p)

    t1 = time()
    GRcut = Green_function_R(omega, p; mode=:cutoff)
    GRfull = Green_function_R(omega, p; mode=:full)
    
    t2 = time()
    hbar_rescaled = 1.0
    force_r = 1 / pi * hbar_rescaled * sum(diag(imag(alpha_val_I) * imag(GRcut))) + 1/ pi * hbar_rescaled * sum(diag(imag(J_val) * imag(GRfull)))
    
    # The minus comes from the product of the two immaginary constants i
    return -(force_r) 
end

function Force_t(omega, p::GreenParams)
    t0 = time()
    alpha_val_I, J_val = alpha_and_J(omega, p)

    t1 = time()
    GRcut = Green_function_R(omega, p; mode=:cutoff)
    GRfull = Green_function_R(omega, p; mode=:full)

    t2 = time()
    hbar_rescaled = 1.0
    force_t = 1 / pi * hbar_rescaled * sum(diag(real(alpha_val_I) * real(GRcut))) + 1/ pi * hbar_rescaled * sum(diag(real(J_val) * real(GRfull)))
      
    return force_t
end



################################################################################
# INTEGRATION AND PLOTTING FUNCTIONS
# Integrate the force over the frequency domain using logarithmic spacing and adaptive quadrature
function integrate_omega_quad(func, omega_start, omega_end, omega_tol; log_splits=200, verbose=false, scale=1e18, show_progress=true, skip_threshold=0.0)#1e-30)

    # --- Integration Strategy ---
    # 1. We log-space partition the ω domain
    # 2. Very small function values (below skip_threshold) are ignored
    # 3. The integrand is scaled by scale to improve numerical stability

    ω_edges = 10 .^ range(log10(omega_start), log10(omega_end), length=log_splits + 1)
    contributions = Vector{Float64}(undef, log_splits)

    function integrate_interval(a, b)

        if skip_threshold > 0
            fa = abs(func(a))
            fb = abs(func(b))

            if fa < skip_threshold && fb < skip_threshold
                return 0.0
            end
        end

        integrand_scaled = ω -> scale * func(ω)
        val_scaled, _ = quadgk(integrand_scaled, a, b; rtol=omega_tol, atol=0.0)
        value = val_scaled / scale

        if verbose
            println("[LOG] [$a, $b] → $value")
        end
            return value
        end

        if show_progress
            @showprogress 1 "Integrating over ω..." for i in eachindex(contributions)
                contributions[i] = integrate_interval(ω_edges[i], ω_edges[i + 1])
            end
        else
            for i in eachindex(contributions)
                contributions[i] = integrate_interval(ω_edges[i], ω_edges[i + 1])
            end
        end

    return sum(contributions)
end

# Compute and plot the force components for a range of velocities
function compute_force_for_velocity(v_vals::Vector{Float64}; verbose=true)
    forces_t = zeros(length(v_vals))
    forces_r = zeros(length(v_vals))
    resc_factor = params.hbar * params.omega_p^2 / params.c / params.m_atom * 1e6 

    for (i, v_dimless) in enumerate(v_vals)
        p_v = GreenParams(params; v_dimless = v_dimless)

        # Compute both integrals
        force_t_integrand = ω -> Force_t(ω, p_v)
        force_r_integrand = ω -> Force_r(ω, p_v)
        force_t = integrate_omega_quad(force_t_integrand, omega_start_def, omega_end_def, tolerance_omega_def)
        force_r = integrate_omega_quad(force_r_integrand, omega_start_def, omega_end_def, tolerance_omega_def)

        # Rescale to physical units
        forces_t[i] = force_t * resc_factor
        forces_r[i] = force_r * resc_factor

        if verbose
            @info "v = $(round(v_dimless, sigdigits=3)) → Force_t = $(round(forces_t[i], sigdigits=5)), Force_r = $(round(forces_r[i], sigdigits=5))"
        end
    end

    return v_vals, forces_t, forces_r
end

function plot_force_vs_velocity(v_range::AbstractVector{Float64}; save_plot=false,save_data=false, base_filename="force_vs_velocity")
    
    v_vals, force_t, force_r = compute_force_for_velocity(v_range)
    F_total = force_t + force_r

   
    # === Superscript helper ===
    function superscript(n::Int)
        sup = Dict(
            '-' => '⁻',
            '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
            '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹'
        )
        return join(sup[c] for c in string(n))
    end

    # === Y-axis ticks for log plots ===
    all_y = abs.([force_t; force_r; F_total])
    ymin = floor(Int, log10(minimum(all_y)))
    ymax = ceil(Int, log10(maximum(all_y)))

    ytick_exponents = collect(ymin:3:ymax)
    ytick_vals = 10.0 .^ ytick_exponents
    ytick_labels = ["10" * superscript(e) for e in ytick_exponents]

    # === Plot Styles Helper ===
    function make_plot(; xscale=:identity, yscale=:identity, title_str::String, use_abs=false)
        f_r = use_abs ? abs.(force_r) : force_r
        f_t = use_abs ? abs.(force_t) : force_t
        f_tot = use_abs ? abs.(F_total) : F_total


        plt = plot(v_vals, f_r, label = "aₐₛ", lw = 2, color = :red,xlabel = "v/c", ylabel = "a [μm/s²]",xscale = xscale, yscale = yscale, legend = :bottomright, 
                   legendfontsize = 8,title = title_str,size = (1000, 400),)
        plot!(plt, v_vals, f_t, label = "-aₛ", lw = 2, color = :gold)
        plot!(plt, v_vals, f_tot, label = "-aₜₕ", lw = 2, linestyle = :dash, color = :green)
        

        if yscale == :log10
            plot!(yticks = (ytick_vals, ytick_labels), yminorgrid = true)
        end
        if xscale == :log10
            plot!(xminorgrid = true)
        end

        return plt
    end

    # === Generate all plots ===
    plt_linear = make_plot(xscale = :identity, yscale = :identity, title_str = "Linear: Correction to C-P force", use_abs = true )
    plt_semilogy = make_plot(yscale = :identity, xscale = :log10, title_str = "Semi-Log (Y): Correction to C-P force", use_abs = true)
    plt_loglog = make_plot(yscale = :log10, xscale = :log10, title_str = "Log: Correction to C-P force", use_abs = true)

    # === Display Plots ===
    display(plt_linear)
    display(plt_semilogy)
    display(plt_loglog)

    # === Optional Save ===
    saved_files = Dict{String,String}()
    if save_data
        # Ensure filenames
        f_as_filename = base_filename * "_f_as.csv"
        f_sym_filename = base_filename * "_f_sym.csv"
        f_tot_filename = base_filename * "_f_tot.csv"

        # Create DataFrames
        df_f_as = DataFrame(v = v_vals, f_as = force_r)
        df_f_sym = DataFrame(v = v_vals, f_sym = force_t)
        df_f_tot = DataFrame(v = v_vals, f_tot = F_total)

        # Write CSVs (overwrite if exist)
        CSV.write(f_as_filename, df_f_as)
        CSV.write(f_sym_filename, df_f_sym)
        CSV.write(f_tot_filename, df_f_tot)

        saved_files["f_as_m"] = f_as_filename
        saved_files["_f_sym"] = f_sym_filename
        saved_files["f_tot"] = f_tot_filename

        println("Saved data to: ", f_as_filename, " and ", f_sym_filename, " and ", f_tot_filename)
    end
    

    return plt_linear, plt_semilogy,plt_loglog, v_vals, force_t, force_r, (save_data ? saved_files : nothing)

end

## Convergence study for the quadrature integration of the force over frequency
function convergence_study_quad(func, omega_start_list, omega_end_list, tolerance_list; label_prefix="Force", verbose=true)

    results = Dict("tol" => Float64[],"a" => Float64[], "b" => Float64[])
    scale = params.hbar * params.omega_p^2 / params.c / params.m_atom * 1e6 

    # Sweep tolerance values 
    for (i, tol) in enumerate(tolerance_list)
        println("Integrating tolerance sweep = $tol ($i / $(length(tolerance_list)))")
        val = integrate_omega_quad(func, omega_start_def, omega_end_def, tol)
        push!(results["tol"], val * scale)
    end

    # Sweep omega_start values (with fixed omega_end)
    for (i, a) in enumerate(omega_start_list)
        println("Integrating omega_start sweep = $a ($i / $(length(omega_start_list)))")
        val = integrate_omega_quad(func, a, omega_end_def, tolerance_omega_def)
        push!(results["a"], val * scale)
    end

    # Sweep omega_end values (with fixed omega_start)
    for (i, b) in enumerate(omega_end_list)
        println("Integrating omega_end sweep = $b ($i / $(length(omega_end_list)))")
        val = integrate_omega_quad(func, omega_start_def, b, tolerance_omega_def)
        push!(results["b"], val * scale)
    end

    # Plotting function
    fig_linear = plot(layout=(1, 3), size=(1800, 400), title="$label_prefix convergence study")
    fig_loglog = plot(layout=(1, 3), size=(1800, 400), title="$label_prefix convergence study")

    sweeps = [
        (tolerance_list, results["tol"], "Relative tolerance"),
        (omega_start_list, results["a"], "Lower frequency bound"),
        (omega_end_list, results["b"], "Upper frequency bound"),]

    for (i, (x_values, y_values, x_label)) in enumerate(sweeps)

        plot!(fig_linear[i], x_values,y_values; marker = :circle,label = "",xlabel = x_label,ylabel = "$label_prefix [μm/s²]",xscale = :log10,grid = true)

        plot!(fig_loglog[i],x_values,abs.(y_values);marker = :circle,label = "",
        xlabel = x_label,ylabel = "|$label_prefix| [μm/s²]",xscale = :log10,yscale = :log10,grid = true,)
    end

    display(fig_linear)
    display(fig_loglog)

    return results, fig_linear, fig_loglog
end




velocity = 10 .^ range(log10(9e-5),log10(1.2e-2), 5)
plot_force_vs_velocity(velocity)

v_dimless = params.v / params.c
p_v = GreenParams(params; v_dimless = v_dimless)
#convergence_study_quad(ω -> Force_t(ω, p_v), [ 1e-5, 1e-3], [1e-14, 1e-6], [0.05, 2,3,4];)  


