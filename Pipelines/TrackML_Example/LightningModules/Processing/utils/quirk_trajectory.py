import numpy as np


def _unit(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n


def _lorentz_accel(v, q, b_field, inv_mass):
    b_vec = np.array([0.0, 0.0, b_field], dtype=np.float64)
    return q * np.cross(v, b_vec) * inv_mass


def _string_accel(p_self, p_other, string_tension, inv_mass):
    delta = p_other - p_self
    return string_tension * _unit(delta) * inv_mass


def _radius_from_pt(pt, charge, b_field):
    # rho[m] = pT[GeV] / (0.3 * |q| * B[T]); convert to mm
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
    x0=0.0,
    y0=0.0,
    z0=0.0,
    string_tension=0.02,
    oscillation_jitter=0.0,
    velocity_scale=1500.0,
):
    """
    Integrate coupled quirk/anti-quirk trajectories in a uniform B field.

    This model uses:
    - Lorentz force on each endpoint (opposite electric charges)
    - String-like central force coupling the pair
    - Pair-level non-zero pT drift (from ISR-like boost)
    """
    n_steps = int(max(200, n_steps))
    t_max = float(max(1e-4, t_max))
    dt = t_max / (n_steps - 1)

    charge = float(charge)
    b_field = float(b_field)
    quirk_mass = float(max(1e-3, quirk_mass))
    pair_pt = float(max(1e-4, pair_pt))
    pair_pz = float(pair_pz)
    opening_angle = float(np.clip(opening_angle, 1e-3, np.pi - 1e-3))
    string_tension = float(max(0.0, string_tension))
    # `velocity_scale` maps momentum-like inputs to practical detector-scale motion (mm).
    inv_mass = float(max(1e-6, velocity_scale)) / quirk_mass

    # Split pair momentum into COM drift + opposite components in transverse plane.
    p_com = np.array(
        [pair_pt * np.cos(phi0), pair_pt * np.sin(phi0), pair_pz],
        dtype=np.float64,
    )
    perp_dir = np.array([-np.sin(phi0), np.cos(phi0), 0.0], dtype=np.float64)
    rel_mag = 0.5 * pair_pt * np.tan(0.5 * opening_angle)
    p_rel = rel_mag * perp_dir

    p1 = 0.5 * p_com + p_rel
    p2 = 0.5 * p_com - p_rel

    # Initialize velocity proportional to momentum.
    v1 = p1 * inv_mass
    v2 = p2 * inv_mass

    # Quirks are born at the interaction point with a tiny initial separation.
    # Previously sep0 was 400-900 mm which placed quirks already inside the
    # strip barrel and gave trajectories that never crossed module planes.
    # With sep0=2 mm both quirks start near the origin and their velocity
    # carries them outward through the pixel and strip barrel layers.
    sep0 = 2.0
    pos1 = np.array([x0, y0, z0], dtype=np.float64) + 0.5 * sep0 * perp_dir
    pos2 = np.array([x0, y0, z0], dtype=np.float64) - 0.5 * sep0 * perp_dir

    xyz1 = np.zeros((n_steps, 3), dtype=np.float64)
    xyz2 = np.zeros((n_steps, 3), dtype=np.float64)
    vel1 = np.zeros((n_steps, 3), dtype=np.float64)
    vel2 = np.zeros((n_steps, 3), dtype=np.float64)

    for i in range(n_steps):
        xyz1[i] = pos1
        xyz2[i] = pos2
        vel1[i] = v1
        vel2[i] = v2

        a1 = _lorentz_accel(v1, +charge, b_field, inv_mass) + _string_accel(
            pos1, pos2, string_tension, inv_mass
        )
        a2 = _lorentz_accel(v2, -charge, b_field, inv_mass) + _string_accel(
            pos2, pos1, string_tension, inv_mass
        )

        if oscillation_jitter > 0.0:
            jitter = float(oscillation_jitter)
            a1 += jitter * np.random.normal(size=3)
            a2 += jitter * np.random.normal(size=3)

        # Semi-implicit Euler keeps integration stable for this use-case.
        v1 = v1 + dt * a1
        v2 = v2 + dt * a2
        pos1 = pos1 + dt * v1
        pos2 = pos2 + dt * v2

    pxyz1 = (vel1 * quirk_mass).astype(np.float32)
    pxyz2 = (vel2 * quirk_mass).astype(np.float32)

    return {
        "event_id": int(event_id),
        "xyz_q": xyz1.astype(np.float32),
        "xyz_aq": xyz2.astype(np.float32),
        "pxyz_q": pxyz1,
        "pxyz_aq": pxyz2,
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
    Generate a synthetic quirk-like trajectory in a uniform B field.

    The base motion is a helical path; an optional radial oscillation can be added
    through `quirky_amplitude` and `quirky_frequency`.
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

    # Use a controllable base radius for synthetic detector intersection.
    r0 = float(max(150.0, radius_mm))
    radial_mod = 1.0 + quirky_amplitude * np.sin(quirky_frequency * t)
    radius = r0 * radial_mod

    x = x0 + radius * np.cos(phi)
    y = y0 + radius * np.sin(phi)
    z = z0 + (pz / pt) * radius * omega * t

    # Approximate momentum direction from local tangent.
    dt = t[1] - t[0]
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    vnorm = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-12
    px = pt * vx / vnorm
    py = pt * vy / vnorm
    pz_arr = pt * vz / vnorm

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    pxyz = np.stack([px, py, pz_arr], axis=1).astype(np.float32)

    return {
        "event_id": int(event_id),
        "xyz": xyz,
        "pxyz": pxyz,
        "q": np.float32(charge),
    }
