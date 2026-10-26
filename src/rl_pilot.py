import numpy as np

class OrbitalRLPilot:
    """
    Policy Gradient Actor-Critic Pilot for Autonomous Collision Avoidance.
    Learns 3D thrust vector actions to maximize miss-distance while minimizing propellant expenditure.
    """
    def __init__(self, state_dim=9, action_dim=3, hidden_dim=32, lr=0.012):
        np.random.seed(42)
        # Policy Network (Actor)
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, action_dim))
        
        self.lr = lr
        self.episode_rewards = []
        self.max_thrust = 0.005  # km/s^2 (~5 m/s^2 burst limit)
        
    def _tanh(self, z):
        return np.tanh(z)
        
    def get_action(self, state):
        """Forward pass generating continuous 3D thrust action."""
        state = state.reshape(1, -1)
        h = self._tanh(np.dot(state, self.W1) + self.b1)
        raw_thrust = self._tanh(np.dot(h, self.W2) + self.b2)
        # Add exploration noise
        noise = np.random.normal(0, 0.05, size=raw_thrust.shape)
        action = np.clip(raw_thrust + noise, -1.0, 1.0) * self.max_thrust
        return action.flatten(), h
        
    def train_step(self, states, actions, rewards):
        """Policy gradient update using discounted returns."""
        gamma = 0.95
        discounted_returns = np.zeros_like(rewards, dtype=np.float64)
        running_add = 0.0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_returns[t] = running_add
            
        # Standardize returns
        if np.std(discounted_returns) > 1e-6:
            discounted_returns = (discounted_returns - np.mean(discounted_returns)) / (np.std(discounted_returns) + 1e-6)
            
        # Compute gradient approximations
        for s, a, G in zip(states, actions, discounted_returns):
            s = s.reshape(1, -1)
            h = self._tanh(np.dot(s, self.W1) + self.b1)
            pred_a = self._tanh(np.dot(h, self.W2) + self.b2) * self.max_thrust
            
            grad_out = (a - pred_a.flatten()) * G
            dW2 = np.dot(h.T, grad_out.reshape(1, -1))
            self.W2 += self.lr * dW2
            self.b2 += self.lr * grad_out.reshape(1, -1)
            
        self.episode_rewards.append(float(np.sum(rewards)))
