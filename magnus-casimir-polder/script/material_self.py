module MaterialConstants

using LinearAlgebra

export MaterialParameters, epsilon_drude, derivative_epsilon_drude,
       epsilon_lorentz, derivative_epsilon_lorentz,
       r_TM, r_TE, T_v, phi_func

       #new data type named MaterialParameters. This acts like a box that holds all the physical constants 
struct MaterialParameters
    hbar::Float64
    hbar_eV::Float64
    k_B::Float64
    e::Float64
    m_e::Float64
    epsilon_0::Float64
    c::Float64
    z_a::Float64
    t::Float64
    omega_a::Float64
    v::Float64
    d_Hydrogen::Float64
    d::Float64
    alpha_0::Float64
    theta::Float64
    phi::Float64
    m_atom::Float64
    n_e::Float64
    omega_p::Float64
    gamma::Float64
    rho::Float64
    epsilon_inf::Float64
    omega_j::Float64
    gamma_j::Float64
    mu::Float64
    v_Fermi::Float64
end

function MaterialParameters()
    return MaterialParameters(
        1.0545718e-34,                   # hbar
        6.582119569e-16,                 # hbar_eV
        1.380649e-23,                    # k_B
        1.602e-19,                       # e
        9.109e-31,                       # m_e
        8.854187817e-12,                # epsilon_0
        3e8,                             # c
        5e-9,                            # z_a
        1e-8,                            # t
        1.6 / 6.582119569e-16,           # omega_a
        4.88525e-04 * 3e8,               #v                 # v
        1e-30,                           # d_Hydrogen
        4e-29,                           # d
        4 * π * 8.854187817e-12 * 4.728e-29,  # alpha_0
        0,                               # theta
        π / 3,                           # phi
        1.44301e-25,                     # m_atom
        8.85e28,                         # n_e
        9 / 6.582119569e-16,             # omega_p
        0.035/ 6.582119569e-16,         # gamma
        (0.035 / 6.582119569e-16) / (8.854187817e-12 * (9 / 6.582119569e-16)^2),  # rho
        2.5,                             # epsilon_inf
        1.0e15,                          # omega_j
        1.0e12,                          # gamma_j
        1,                                # mu
        1.4e6                              #v_Fermi
    )
end

function epsilon_drude(material::MaterialParameters, omega)
    return 1 - 1 / (omega * (omega + im * material.gamma / material.omega_p))
end

function derivative_epsilon_drude(material::MaterialParameters, omega)
    num = (material.omega_p^2) * (2 * omega + im * material.gamma)
    den = (material.omega_p * (omega + im * material.gamma))^2
    return num / den
end

function epsilon_lorentz(material::MaterialParameters, omega)
    return material.epsilon_inf +
           (material.epsilon_0 - material.epsilon_inf) * material.omega_j^2 /
           (material.omega_j^2 - omega^2 - im * material.gamma_j * omega)
end

function derivative_epsilon_lorentz(material::MaterialParameters, omega)
    num = (material.epsilon_0 - material.epsilon_inf) * material.omega_j^2 *
          (2 * omega + im * material.gamma_j)
    den = (material.omega_j^2 - omega^2 - im * material.gamma_j * omega)^2
    return num / den
end

function _force_branch(z)
    z = complex(z)
    w = sqrt(z)
    if real(w) < 0
        w = -w
    end
    if imag(w) > 0
        w = conj(w)
    end
    return w
end

function r_TM(material::MaterialParameters, epsilon)
    #function r_TM(material::MaterialParameters, omega, k, epsilon, mu)
    #argument_m = k^2 - epsilon * mu * omega^2
    #argument = k^2 - omega^2
    #kappa_m = _force_branch(argument_m)
    #kappa = _force_branch(argument)
#return (epsilon * kappa - kappa_m) / (epsilon * kappa + kappa_m)
return (epsilon -1) / (epsilon +1)
end


function r_TE(material::MaterialParameters, omega, k, epsilon, mu)
    argument_m = k^2 - epsilon * mu * omega^2
    argument = k^2 - omega^2
    kappa_m = _force_branch(argument_m)
    kappa = _force_branch(argument)
    return (mu * kappa - kappa_m) / (mu * kappa + kappa_m)
end

function phi_func(phi, theta)
    return 1 + (2 * cos(2 * phi) * sin(theta)^2) / (3 * cos(2 * theta) + 3)
end

function T_v(material::MaterialParameters, v, phi, theta, z_a)
    return (phi_func(phi, theta) * material.hbar * v) / (2 * material.k_B * z_a)
end

end # module
