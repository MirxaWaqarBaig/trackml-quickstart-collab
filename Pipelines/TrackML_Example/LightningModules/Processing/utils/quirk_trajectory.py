import numpy as np


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
