import numpy as np


def _unit(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n


def _lorentz_accel(v, q, b_field, inv_mass):
    b_vec = np.array([0.0, 0.0, b_field], dtype=np.float64)
    return q * np.cross(v, b_vec) * inv_mass


def _radius_from_pt(pt, charge, b_field):
    denom = max(1e-6, 0.3 * abs(charge) * max(1e-6, b_field))
    return 1000.0 * pt / denom


def simulate_quirk_pair_tracks(
    event_id,
    n_steps=6000,
    t_max=8.0,
    b_field=2.0,
    charge=1.0,
    quirk_mass=1000.0,
    pair_pt=5.0,
    pair_pz=1.0,
    opening_angle=np.pi / 2.0,
    phi0=0.0,
    x0=0.0, y0=0.0, z0=0.0,
    string_tension=0.02,
    oscillation_jitter=0.0,
    velocity_scale=1500.0,
):
    """
    Quirk pair simulation using linear (Hookean) string force + RK4.

    The string force is F = kappa * delta (linear restoring force),
    which correctly produces oscillating tangled-ribbon trajectories
    matching the intermediate oscillation regime of Sha et al. 2410.00269.

    String dominates over Lorentz when:
        kappa * sep >> charge * v * B * inv_mass
    Tune string_tension and velocity_scale to achieve this regime.
    """
    n_steps        = int(max(200, n_steps))
    t_max          = float(max(1e-4, t_max))
    dt             = t_max / (n_steps - 1)
    charge         = float(charge)
    b_field        = float(b_field)
    quirk_mass     = float(max(1e-3, quirk_mass))
    pair_pt        = float(max(1e-4, pair_pt))
    pair_pz        = float(pair_pz)
    opening_angle  = float(np.clip(opening_angle, 1e-3, np.pi - 1e-3))
    string_tension = float(max(0.0, string_tension))
    inv_mass       = float(max(1e-6, velocity_scale)) / quirk_mass

    # Initial momenta: COM drift + opposite transverse kicks
    p_com    = np.array([pair_pt*np.cos(phi0),
                         pair_pt*np.sin(phi0),
                         pair_pz], dtype=np.float64)
    perp_dir = np.array([-np.sin(phi0), np.cos(phi0), 0.0],
                        dtype=np.float64)
    rel_mag  = 0.5 * pair_pt * np.tan(0.5 * opening_angle)
    p_rel    = rel_mag * perp_dir

    p1 = 0.5 * p_com + p_rel
    p2 = 0.5 * p_com - p_rel

    sep0   = 2.0  # mm
    pos1_0 = np.array([x0, y0, z0], dtype=np.float64) + 0.5*sep0*perp_dir
    pos2_0 = np.array([x0, y0, z0], dtype=np.float64) - 0.5*sep0*perp_dir
    v1_0   = p1 * inv_mass
    v2_0   = p2 * inv_mass

    b_vec = np.array([0.0, 0.0, b_field], dtype=np.float64)

    def derivatives(state):
        r1 = state[0:3];  u1 = state[3:6]
        r2 = state[6:9];  u2 = state[9:12]

        # Lorentz force (opposite charges)
        a1_lor = (+charge) * np.cross(u1, b_vec) * inv_mass
        a2_lor = (-charge) * np.cross(u2, b_vec) * inv_mass

        # Linear string force: F = kappa * (r_other - r_self)
        # This is the Hookean restoring force used in Sha et al. Colab sim.
        # Produces oscillation when string_tension >> Lorentz bending.
        delta  = r2 - r1
        a1_str = string_tension * delta * inv_mass
        a2_str = -string_tension * delta * inv_mass

        return np.concatenate([u1, a1_lor+a1_str, u2, a2_lor+a2_str])

    # RK4 integration
    state = np.concatenate([pos1_0, v1_0, pos2_0, v2_0])

    xyz1 = np.zeros((n_steps, 3), dtype=np.float64)
    xyz2 = np.zeros((n_steps, 3), dtype=np.float64)
    vel1 = np.zeros((n_steps, 3), dtype=np.float64)
    vel2 = np.zeros((n_steps, 3), dtype=np.float64)

    for i in range(n_steps):
        xyz1[i] = state[0:3];  vel1[i] = state[3:6]
        xyz2[i] = state[6:9];  vel2[i] = state[9:12]

        if not np.all(np.isfinite(state)):
            xyz1[i:] = xyz1[max(0, i-1)]
            xyz2[i:] = xyz2[max(0, i-1)]
            vel1[i:] = vel1[max(0, i-1)]
            vel2[i:] = vel2[max(0, i-1)]
            break

        k1 = derivatives(state)
        k2 = derivatives(state + 0.5*dt*k1)
        k3 = derivatives(state + 0.5*dt*k2)
        k4 = derivatives(state + dt*k3)
        state = state + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    pxyz1 = (vel1 * quirk_mass).astype(np.float32)
    pxyz2 = (vel2 * quirk_mass).astype(np.float32)

    return {
        "event_id": int(event_id),
        "xyz_q":    xyz1.astype(np.float32),
        "xyz_aq":   xyz2.astype(np.float32),
        "pxyz_q":   pxyz1,
        "pxyz_aq":  pxyz2,
        "q_q":      np.float32(charge),
        "q_aq":     np.float32(-charge),
        "time":     np.linspace(0.0, t_max, n_steps, dtype=np.float32),
    }


def simulate_quirk_track(
    event_id,
    n_steps=6000,
    t_max=8.0,
    b_field=2.0,
    charge=1.0,
    pt=5.0,
    pz=1.0,
    phi0=0.0,
    x0=0.0, y0=0.0, z0=0.0,
    quirky_amplitude=0.0,
    quirky_frequency=3.0,
    radius_mm=900.0,
):
    """Synthetic helical quirk-like trajectory with optional radial oscillation."""
    n_steps = int(max(10, n_steps))
    t_max   = float(max(1e-3, t_max))
    charge  = float(charge)
    pt      = float(max(1e-6, pt))
    pz      = float(pz)
    b_field = float(b_field)

    t     = np.linspace(0.0, t_max, n_steps, dtype=np.float64)
    omega = 0.3 * charge * b_field / pt
    phi   = phi0 + omega * t
    r0    = float(max(150.0, radius_mm))
    radius = r0 * (1.0 + quirky_amplitude * np.sin(quirky_frequency * t))

    x = x0 + radius * np.cos(phi)
    y = y0 + radius * np.sin(phi)
    z = z0 + (pz / pt) * radius * omega * t

    dt_val = t[1] - t[0]
    vx = np.gradient(x, dt_val)
    vy = np.gradient(y, dt_val)
    vz = np.gradient(z, dt_val)
    vnorm = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-12

    xyz  = np.stack([x, y, z], axis=1).astype(np.float32)
    pxyz = np.stack([pt*vx/vnorm, pt*vy/vnorm, pt*vz/vnorm],
                    axis=1).astype(np.float32)

    return {
        "event_id": int(event_id),
        "xyz":  xyz,
        "pxyz": pxyz,
        "q":    np.float32(charge),
    }
