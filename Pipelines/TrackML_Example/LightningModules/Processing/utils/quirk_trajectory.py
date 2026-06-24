import numpy as np


def _unit(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n


def simulate_quirk_pair_tracks(
    event_id,
    n_steps=70000,
    t_max=80.0,
    b_field=2.0,
    charge=1.0,
    quirk_mass=500.0,
    pair_pt=50.0,
    pair_pz=10.0,
    opening_angle=1.3,
    phi0=0.0,
    x0=0.0,
    y0=0.0,
    z0=0.0,
    string_tension=900.0,
    oscillation_jitter=0.0,
    velocity_scale=1.0,
):
    """
    Physical unit quirk pair simulator.

    Units:
        position = mm
        time = ns
        mass = GeV/c^2
        momentum = GeV/c
        magnetic field = Tesla
        string_tension = Lambda in eV

    string_tension is Lambda in eV, not old Hookean kappa.
    velocity_scale is kept only for API compatibility and is not used.
    """

    n_steps = int(max(200, n_steps))
    t_max = float(max(1e-6, t_max))
    dt = t_max / float(n_steps - 1)

    b_field = float(b_field)
    charge = float(charge)
    quirk_mass = float(max(1e-9, quirk_mass))
    pair_pt = float(max(1e-9, pair_pt))
    pair_pz = float(pair_pz)
    opening_angle = float(np.clip(opening_angle, 1e-6, np.pi - 1e-6))

    c_mm_ns = 299.792458
    hbar_c_gev_mm = 1.973269804e-13

    lambda_ev = float(max(0.0, string_tension))
    lambda_gev = lambda_ev * 1.0e-9
    T_gev_mm = (lambda_gev * lambda_gev) / hbar_c_gev_mm if lambda_gev > 0.0 else 0.0

    p_com = np.array(
        [
            pair_pt * np.cos(phi0),
            pair_pt * np.sin(phi0),
            pair_pz,
        ],
        dtype=np.float64,
    )

    perp_dir = np.array(
        [-np.sin(phi0), np.cos(phi0), 0.0],
        dtype=np.float64,
    )

    rel_mag = 0.5 * pair_pt * np.tan(0.5 * opening_angle)
    p_rel = rel_mag * perp_dir

    p1_0 = 0.5 * p_com + p_rel
    p2_0 = 0.5 * p_com - p_rel

    sep0 = 2.0
    r0 = np.array([x0, y0, z0], dtype=np.float64)

    r1_0 = r0 + 0.5 * sep0 * perp_dir
    r2_0 = r0 - 0.5 * sep0 * perp_dir

    b_vec = np.array([0.0, 0.0, b_field], dtype=np.float64)

    def velocity_from_p(p):
        p2 = float(np.dot(p, p))
        energy = np.sqrt(quirk_mass * quirk_mass + p2)
        return c_mm_ns * p / (energy + 1e-30)

    def rhs(state):
        r1 = state[0:3]
        p1 = state[3:6]
        r2 = state[6:9]
        p2 = state[9:12]

        v1 = velocity_from_p(p1)
        v2 = velocity_from_p(p2)

        beta1 = v1 / c_mm_ns
        beta2 = v2 / c_mm_ns

        dpdt1_lor = 0.3 * (+charge) * np.cross(beta1, b_vec)
        dpdt2_lor = 0.3 * (-charge) * np.cross(beta2, b_vec)

        delta = r2 - r1
        string_hat = _unit(delta)

        dpdt1_str = +T_gev_mm * c_mm_ns * string_hat
        dpdt2_str = -T_gev_mm * c_mm_ns * string_hat

        drdt1 = v1
        drdt2 = v2

        dpdt1 = dpdt1_lor + dpdt1_str
        dpdt2 = dpdt2_lor + dpdt2_str

        return np.concatenate([drdt1, dpdt1, drdt2, dpdt2])

    state = np.concatenate([r1_0, p1_0, r2_0, p2_0]).astype(np.float64)

    xyz1 = np.zeros((n_steps, 3), dtype=np.float64)
    xyz2 = np.zeros((n_steps, 3), dtype=np.float64)
    pxyz1 = np.zeros((n_steps, 3), dtype=np.float64)
    pxyz2 = np.zeros((n_steps, 3), dtype=np.float64)

    for i in range(n_steps):
        xyz1[i] = state[0:3]
        pxyz1[i] = state[3:6]
        xyz2[i] = state[6:9]
        pxyz2[i] = state[9:12]

        if not np.all(np.isfinite(state)):
            xyz1[i:] = xyz1[max(0, i - 1)]
            xyz2[i:] = xyz2[max(0, i - 1)]
            pxyz1[i:] = pxyz1[max(0, i - 1)]
            pxyz2[i:] = pxyz2[max(0, i - 1)]
            break

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)

        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return {
        "event_id": int(event_id),
        "xyz_q": xyz1.astype(np.float32),
        "xyz_aq": xyz2.astype(np.float32),
        "pxyz_q": pxyz1.astype(np.float32),
        "pxyz_aq": pxyz2.astype(np.float32),
        "q_q": np.float32(charge),
        "q_aq": np.float32(-charge),
        "time": np.linspace(0.0, t_max, n_steps, dtype=np.float32),
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
    x0=0.0,
    y0=0.0,
    z0=0.0,
    quirky_amplitude=0.0,
    quirky_frequency=3.0,
    radius_mm=900.0,
):
    """
    Legacy single track helper kept only for API compatibility.
    Main quirk pair generation uses simulate_quirk_pair_tracks().
    """

    n_steps = int(max(10, n_steps))
    t_max = float(max(1e-3, t_max))
    charge = float(charge)
    pt = float(max(1e-6, pt))
    pz = float(pz)
    b_field = float(b_field)

    t = np.linspace(0.0, t_max, n_steps, dtype=np.float64)
    omega = 0.3 * charge * b_field / pt
    phi = phi0 + omega * t
    r0 = float(max(150.0, radius_mm))
    radius = r0 * (1.0 + quirky_amplitude * np.sin(quirky_frequency * t))

    x = x0 + radius * np.cos(phi)
    y = y0 + radius * np.sin(phi)
    z = z0 + (pz / pt) * radius * omega * t

    dt_val = t[1] - t[0]
    vx = np.gradient(x, dt_val)
    vy = np.gradient(y, dt_val)
    vz = np.gradient(z, dt_val)
    vnorm = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-12

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    pxyz = np.stack(
        [pt * vx / vnorm, pt * vy / vnorm, pt * vz / vnorm],
        axis=1,
    ).astype(np.float32)

    return {
        "event_id": int(event_id),
        "xyz": xyz,
        "pxyz": pxyz,
        "q": np.float32(charge),
    }
