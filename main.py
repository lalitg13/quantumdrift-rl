import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.orbital_physics import OrbitalMechanicsEngine
from src.rl_pilot import OrbitalRLPilot

def run_simulation():
    print("[*] Initializing QuantumDrift-RL Orbital Astrodynamics Engine...")
    pilot = OrbitalRLPilot(state_dim=9, action_dim=3, hidden_dim=32, lr=0.015)
    
    num_episodes = 100
    steps_per_episode = 40
    dt = 15.0  # seconds per step
    
    pareto_miss_dists = []
    pareto_fuel_burns = []
    avoidance_success = []
    
    best_sat_traj = []
    sample_debris_traj = []
    
    print(f"[*] Training Policy Gradient RL Agent over {num_episodes} orbital rendezvous episodes...")
    for ep in range(num_episodes):
        engine = OrbitalMechanicsEngine(num_debris=80, seed=ep + 100)
        states, actions, rewards = [], [], []
        sat_traj = []
        ep_delta_v = 0.0
        min_encounter_dist = float("inf")
        
        for step in range(steps_per_episode):
            sat_traj.append(engine.r_sat.copy())
            if ep == num_episodes - 1 and step == 0:
                sample_debris_traj = engine.debris_r.copy()
                
            # Relative state: [r_sat (3), v_sat (3), relative_pos_closest (3)]
            dists = np.linalg.norm(engine.debris_r - engine.r_sat, axis=1)
            closest_idx = np.argmin(dists)
            rel_pos = engine.debris_r[closest_idx] - engine.r_sat
            
            state = np.concatenate([
                engine.r_sat / 7000.0,
                engine.v_sat / 8.0,
                rel_pos / 1000.0
            ])
            
            action, _ = pilot.get_action(state)
            min_dist, _ = engine.step_rk4(dt=dt, thrust=action)
            
            min_encounter_dist = min(min_encounter_dist, min_dist)
            fuel_cost = np.linalg.norm(action) * dt
            ep_delta_v += fuel_cost
            
            # Reward: high miss distance, low fuel, crash penalty
            reward = 0.05 * min(min_dist, 300.0) - 12.0 * fuel_cost
            if min_dist < 25.0:  # Collision breach
                reward -= 100.0
            elif min_dist > 80.0:  # Safe clearance
                reward += 15.0
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
        pilot.train_step(states, actions, rewards)
        pareto_miss_dists.append(min_encounter_dist)
        pareto_fuel_burns.append(ep_delta_v)
        avoidance_success.append(1.0 if min_encounter_dist > 35.0 else 0.0)
        
        if ep == num_episodes - 1:
            best_sat_traj = np.array(sat_traj)
            
        if (ep + 1) % 25 == 0:
            avg_rew = np.mean(pilot.episode_rewards[-25:])
            success_rate = np.mean(avoidance_success[-25:]) * 100
            print(f"    Episode {ep+1:03d}/{num_episodes} | Avg Reward: {avg_rew:.1f} | Avoidance Rate: {success_rate:.1f}% | Min Dist: {min_encounter_dist:.1f} km")
            
    print("[*] Generating high-DPI 3D orbital telemetry visualization...")
    os.makedirs("assets", exist_ok=True)
    
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], wspace=0.25, hspace=0.3)
    
    # Panel A: 3D Orbital Trajectory & Debris Cloud
    ax_3d = fig.add_subplot(gs[0, :], projection='3d')
    # Draw Earth Wireframe
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    re = 6378.137
    x_earth = re * np.outer(np.cos(u), np.sin(v))
    y_earth = re * np.outer(np.sin(u), np.sin(v))
    z_earth = re * np.outer(np.ones(np.size(u)), np.cos(v))
    ax_3d.plot_surface(x_earth, y_earth, z_earth, color="#1E293B", alpha=0.45, edgecolor="#334155", linewidth=0.4)
    
    # Plot Debris Swarm Points
    if len(sample_debris_traj) > 0:
        ax_3d.scatter(
            sample_debris_traj[:, 0], sample_debris_traj[:, 1], sample_debris_traj[:, 2],
            color="#EF4444", s=18, alpha=0.75, label="LEO Debris Hazards (Kessler Field)"
        )
        
    # Plot Satellite Evasive Trajectory
    ax_3d.plot(
        best_sat_traj[:, 0], best_sat_traj[:, 1], best_sat_traj[:, 2],
        color="#06B6D4", linewidth=2.8, label="RL-Optimized Evasive Flight Path"
    )
    ax_3d.scatter(
        [best_sat_traj[0, 0]], [best_sat_traj[0, 1]], [best_sat_traj[0, 2]],
        color="#10B981", s=60, marker="o", label="Orbit Ingress Point"
    )
    ax_3d.scatter(
        [best_sat_traj[-1, 0]], [best_sat_traj[-1, 1]], [best_sat_traj[-1, 2]],
        color="#F59E0B", s=60, marker="^", label="Egress Telemetry Vector"
    )
    ax_3d.set_title("A. 3D Astrodynamics Simulation: Autonomous Satellite Debris Evasion Maneuver", fontsize=11, fontweight="bold", color="#F8FAFC", pad=12)
    ax_3d.set_xlabel("X (km)", fontsize=8)
    ax_3d.set_ylabel("Y (km)", fontsize=8)
    ax_3d.set_zlabel("Z (km)", fontsize=8)
    ax_3d.legend(loc="upper right", fontsize=8, framealpha=0.4)
    ax_3d.view_init(elev=28, azim=45)
    
    # Panel B: RL Training Convergence & Success Rate
    ax_rew = fig.add_subplot(gs[1, 0])
    eps_arr = np.arange(1, num_episodes + 1)
    ax_rew.plot(eps_arr, pilot.episode_rewards, color="#38BDF8", alpha=0.35, linewidth=1.0)
    # Moving average
    window = 10
    mov_avg = np.convolve(pilot.episode_rewards, np.ones(window)/window, mode='valid')
    ax_rew.plot(np.arange(window, num_episodes + 1), mov_avg, color="#0284C7", linewidth=2.4, label="Discounted Return (10-ep MA)")
    ax_rew.set_title("B. Policy Gradient Convergence & Learning Dynamics", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    ax_rew.set_xlabel("Training Episode", fontsize=9)
    ax_rew.set_ylabel("Episodic Return", fontsize=9)
    ax_rew.grid(color="#334155", linestyle="--", alpha=0.5)
    ax_rew.legend(loc="lower right", fontsize=8, framealpha=0.4)
    
    # Panel C: Pareto Frontier: Fuel Consumption vs Miss Distance
    ax_pareto = fig.add_subplot(gs[1, 1])
    sc = ax_pareto.scatter(
        pareto_fuel_burns,
        pareto_miss_dists,
        c=eps_arr,
        cmap="cool",
        s=35,
        alpha=0.85,
        edgecolors="#FFFFFF",
        linewidths=0.5
    )
    cbar_p = plt.colorbar(sc, ax=ax_pareto)
    cbar_p.set_label("Training Epoch", color="#E2E8F0", fontsize=8)
    ax_pareto.axhline(y=35.0, color="#EF4444", linestyle="--", linewidth=1.4, label="Critical Collision Sphere (35 km)")
    ax_pareto.set_title("C. Pareto Optimization: Fuel Delta-V vs. Safety Margin", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    ax_pareto.set_xlabel("Total Maneuver Delta-V (km/s)", fontsize=9)
    ax_pareto.set_ylabel("Closest Approach Miss-Distance (km)", fontsize=9)
    ax_pareto.grid(color="#334155", linestyle="--", alpha=0.5)
    ax_pareto.legend(loc="upper left", fontsize=8, framealpha=0.4)
    
    plt.savefig("assets/orbital_telemetry.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] QuantumDrift-RL telemetry saved to assets/orbital_telemetry.png")

if __name__ == "__main__":
    run_simulation()
