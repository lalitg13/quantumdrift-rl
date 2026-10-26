import os
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from src.orbital_physics import OrbitalMechanicsEngine
from src.rl_pilot import OrbitalRLPilot

def generate_spinning_animation():
    print("[*] Generating 3D Spinning Orbit Animation for QuantumDrift-RL...")
    engine = OrbitalMechanicsEngine(num_debris=70, seed=123)
    pilot = OrbitalRLPilot(state_dim=9, action_dim=3, hidden_dim=32)
    
    # Pre-simulate 40 steps of flight
    sat_traj = []
    debris_traj = engine.debris_r.copy()
    dt = 15.0
    for _ in range(40):
        sat_traj.append(engine.r_sat.copy())
        dists = np.linalg.norm(engine.debris_r - engine.r_sat, axis=1)
        rel_pos = engine.debris_r[np.argmin(dists)] - engine.r_sat
        state = np.concatenate([engine.r_sat / 7000.0, engine.v_sat / 8.0, rel_pos / 1000.0])
        action, _ = pilot.get_action(state)
        engine.step_rk4(dt=dt, thrust=action)
    sat_traj = np.array(sat_traj)
    
    # Create animated rotation frames
    frames = []
    total_frames = 36  # 10 degrees per frame for a 360 loop
    
    # Earth wireframe coordinates
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    re = 6378.137
    x_earth = re * np.outer(np.cos(u), np.sin(v))
    y_earth = re * np.outer(np.sin(u), np.sin(v))
    z_earth = re * np.outer(np.ones(np.size(u)), np.cos(v))
    
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(7, 6), dpi=120)
    
    for i, azim in enumerate(np.linspace(0, 360, total_frames, endpoint=False)):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#030712")
        fig.patch.set_facecolor("#030712")
        
        # Earth Sphere
        ax.plot_wireframe(x_earth, y_earth, z_earth, color="#38BDF8", alpha=0.22, linewidth=0.6)
        
        # Debris Cloud
        ax.scatter(
            debris_traj[:, 0], debris_traj[:, 1], debris_traj[:, 2],
            color="#EF4444", s=14, alpha=0.8, edgecolors="#FCA5A5", linewidths=0.3
        )
        
        # Satellite Trajectory Ribbon
        ax.plot(sat_traj[:, 0], sat_traj[:, 1], sat_traj[:, 2], color="#06B6D4", linewidth=2.5, alpha=0.9)
        
        # Current Satellite Position Marker
        curr_idx = int((i / total_frames) * len(sat_traj)) % len(sat_traj)
        ax.scatter([sat_traj[curr_idx, 0]], [sat_traj[curr_idx, 1]], [sat_traj[curr_idx, 2]], color="#F59E0B", s=55, marker="o", edgecolors="#FFFFFF", linewidths=1.2)
        
        # Camera rotation
        elev = 24.0 + 8.0 * np.sin(np.radians(azim * 2))
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xlim([-9000, 9000])
        ax.set_ylim([-9000, 9000])
        ax.set_zlim([-9000, 9000])
        ax.axis("off")
        ax.set_title("Autonomous LEO Debris Evasion Telemetry", color="#F8FAFC", fontsize=11, fontweight="bold", pad=-5)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#030712", pad_inches=0.1)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        
    plt.close()
    
    output_path = "assets/orbital_rotation.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=65,
        loop=0
    )
    print(f"[✓] Saved 3D spinning orbit animation to {output_path}")

if __name__ == "__main__":
    generate_spinning_animation()
