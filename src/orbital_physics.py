import numpy as np

class OrbitalMechanicsEngine:
    """
    High-precision 3D Astrodynamics Engine with J2 Earth Oblateness Perturbations
    and 4th-Order Runge-Kutta (RK4) Orbital Propagation.
    """
    def __init__(self, num_debris=120, seed=42):
        np.random.seed(seed)
        self.mu = 398600.4418  # Earth standard gravitational parameter (km^3 / s^2)
        self.R_earth = 6378.137  # Earth equatorial radius (km)
        self.J2 = 1.08263e-3     # First zonal harmonic coefficient
        self.num_debris = num_debris
        
        # Satellite initial orbital state (~700 km altitude, near-circular LEO)
        self.r_sat = np.array([7078.0, 0.0, 0.0], dtype=np.float64)  # Position (km)
        v_circ = np.sqrt(self.mu / 7078.0)
        self.v_sat = np.array([0.0, v_circ * 0.95, v_circ * 0.31], dtype=np.float64)  # Velocity (km/s)
        
        # Initialize debris cluster on intersecting inclination planes
        self.debris_r = []
        self.debris_v = []
        for _ in range(num_debris):
            theta = np.random.uniform(0, 2 * np.pi)
            r_dist = np.random.uniform(7000.0, 7150.0)
            inc = np.random.uniform(np.radians(20), np.radians(85))
            
            rx = r_dist * np.cos(theta)
            ry = r_dist * np.sin(theta) * np.cos(inc)
            rz = r_dist * np.sin(theta) * np.sin(inc)
            
            # Perpendicular velocity
            pos_vec = np.array([rx, ry, rz])
            v_mag = np.sqrt(self.mu / r_dist)
            rand_axis = np.cross(pos_vec, np.array([0, 0, 1]))
            if np.linalg.norm(rand_axis) < 1e-4:
                rand_axis = np.array([1, 0, 0])
            v_dir = rand_axis / np.linalg.norm(rand_axis)
            
            self.debris_r.append(pos_vec)
            self.debris_v.append(v_dir * v_mag)
            
        self.debris_r = np.array(self.debris_r, dtype=np.float64)
        self.debris_v = np.array(self.debris_v, dtype=np.float64)
        
    def _acceleration(self, r):
        """Computes Keplerian gravity + J2 perturbation."""
        r_norm = np.linalg.norm(r)
        if r_norm < self.R_earth:
            return np.zeros(3)
            
        # Two-body central gravity
        a_kepler = - (self.mu / (r_norm**3)) * r
        
        # J2 Perturbation acceleration
        z2_over_r2 = (r[2] / r_norm)**2
        factor = (1.5 * self.J2 * self.mu * (self.R_earth**2)) / (r_norm**5)
        a_j2_x = factor * r[0] * (5 * z2_over_r2 - 1)
        a_j2_y = factor * r[1] * (5 * z2_over_r2 - 1)
        a_j2_z = factor * r[2] * (5 * z2_over_r2 - 3)
        a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z])
        
        return a_kepler + a_j2
        
    def step_rk4(self, dt=10.0, thrust=None):
        """Advances satellite and debris positions by dt seconds via RK4."""
        if thrust is None:
            thrust = np.zeros(3)
            
        # RK4 for Satellite
        k1_v = self._acceleration(self.r_sat) + thrust
        k1_r = self.v_sat
        
        k2_v = self._acceleration(self.r_sat + 0.5 * dt * k1_r) + thrust
        k2_r = self.v_sat + 0.5 * dt * k1_v
        
        k3_v = self._acceleration(self.r_sat + 0.5 * dt * k2_r) + thrust
        k3_r = self.v_sat + 0.5 * dt * k2_v
        
        k4_v = self._acceleration(self.r_sat + dt * k3_r) + thrust
        k4_r = self.v_sat + dt * k3_v
        
        self.r_sat += (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        self.v_sat += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        
        # Update Debris (Euler approximation for speed over swarm)
        for i in range(self.num_debris):
            a_deb = self._acceleration(self.debris_r[i])
            self.debris_v[i] += a_deb * dt
            self.debris_r[i] += self.debris_v[i] * dt
            
        # Compute minimum distance to nearest debris fragment
        diffs = self.debris_r - self.r_sat
        dists = np.linalg.norm(diffs, axis=1)
        min_dist = np.min(dists)
        closest_idx = np.argmin(dists)
        
        return min_dist, diffs[closest_idx]
