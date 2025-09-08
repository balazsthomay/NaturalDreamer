import gymnasium as gym
import numpy as np
import pygame


class AdaptiveCourseEnv(gym.Env):
    """Simple adaptive obstacle course with curriculum learning."""
    
    def __init__(self, max_obstacles=6, render_mode=None):
        super().__init__()
        
        self.max_obstacles = max_obstacles
        self.render_mode = render_mode
        
        # Curriculum learning  
        self.difficulty = 0.5  # Start with some obstacles
        self.success_history = []
        
        # Environment state
        self.agent_pos = np.array([0.1, 0.5])
        self.goal_pos = np.array([0.9, 0.5])  # Default, will be randomized in reset()
        self.obstacles = []  # [(x, y, radius), ...]
        self.episode_steps = 0
        self.episode_reward = 0.0  # Track cumulative episode reward
        
        # Spaces (agent_pos + goal_pos + obstacles_flattened + difficulty)
        obs_size = 2 + 2 + max_obstacles * 3 + 1
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (obs_size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        
        # Rendering
        self.screen = None
        
    def _generate_obstacles(self):
        """Generate obstacles based on difficulty."""
        self.obstacles = []
        num_obstacles = int(self.difficulty * 6)  # 0-6 obstacles based on curriculum
        
        for _ in range(num_obstacles):
            while True:
                x, y = np.random.uniform(0.2, 0.8, 2)
                # Avoid start and goal areas
                start_dist = np.linalg.norm([x - 0.1, y - 0.5])
                goal_dist = np.linalg.norm([x - self.goal_pos[0], y - self.goal_pos[1]])
                if start_dist > 0.15 and goal_dist > 0.15:
                    radius = 0.03 + np.random.random() * 0.05
                    self.obstacles.append((x, y, radius))
                    break
    
    def _update_difficulty(self, success):
        """Difficulty adaptation."""
        self.success_history.append(success)
        if len(self.success_history) > 10:
            self.success_history.pop(0)
            
        if len(self.success_history) >= 5:
            rate = np.mean(self.success_history)
            if rate > 0.7:
                self.difficulty = min(1.0, self.difficulty + 0.02)
            elif rate < 0.3:
                self.difficulty = max(0.0, self.difficulty - 0.02)
    
    def _get_observation(self):
        """Get current observation."""
        obs = []
        
        # Agent position (normalized to [-1,1])
        obs.extend(self.agent_pos * 2 - 1)
        
        # Goal position (normalized)
        obs.extend(self.goal_pos * 2 - 1)
        
        # Obstacles (padded to max_obstacles)
        for i in range(self.max_obstacles):
            if i < len(self.obstacles):
                x, y, r = self.obstacles[i]
                obs.extend([x * 2 - 1, y * 2 - 1, (r - 0.055) / 0.025])  # Properly normalize radius
            else:
                obs.extend([0.0, 0.0, 0.0])  # Padding
                
        # Difficulty
        obs.append(self.difficulty * 2 - 1)
        
        return np.array(obs, dtype=np.float32)
    
    def _check_collision(self, pos):
        """Check collision with obstacles or boundaries."""
        if not (0 <= pos[0] <= 1 and 0 <= pos[1] <= 1):
            return True
        
        for x, y, r in self.obstacles:
            if np.linalg.norm(pos - [x, y]) < r + 0.015:  # Slightly more forgiving
                return True
        return False
    
    def _goal_reached(self):
        return np.linalg.norm(self.agent_pos - self.goal_pos) < 0.05
    
    def step(self, action):
        action = np.clip(action, -1, 1) * 0.05  # Scale to reasonable velocity
        new_pos = self.agent_pos + action
        
        reward = 0
        terminated = False
        
        # Progress reward - reduced magnitude for gentler learning
        old_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        new_dist = np.linalg.norm(new_pos - self.goal_pos)
        reward += (old_dist - new_dist) * 3  # Reduced from 10 to 3
        
        # Check collision
        if self._check_collision(new_pos):
            reward -= 3  # Reduced collision penalty
            terminated = True
        else:
            self.agent_pos = new_pos
            reward += 0.1  # Increased exploration reward to offset negative progress
            
        # Check goal
        if self._goal_reached():
            reward += 10
            terminated = True
        else:
            # Proximity bonus - extra reward for getting close to goal
            current_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
            if current_dist < 0.1:  # Very close
                reward += 1.0
            elif current_dist < 0.2:  # Close
                reward += 0.5
            elif current_dist < 0.3:  # Somewhat close
                reward += 0.2
            
        reward -= 0.01  # Time penalty
        self.episode_steps += 1
        self.episode_reward += reward  # Accumulate episode reward
        truncated = self.episode_steps >= 200
        
        # Align success criteria with main training loop (reward > 2.0)
        episode_success = False
        if terminated or truncated:
            # Use SAME threshold as main training for consistency
            episode_success = self.episode_reward > 2.0
            
            # Debug logging for high-reward episodes
            if self.episode_reward > 10.0:
                print(f"ALIGNED SUCCESS DEBUG: reward={self.episode_reward:.2f}, "
                      f"success={episode_success} (threshold: >2.0)")
            
        info = {
            'success': episode_success,
            'difficulty': self.difficulty,
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Update difficulty from last episode
        if options and 'last_episode_success' in options:
            self._update_difficulty(options['last_episode_success'])
            
        self.agent_pos = np.array([0.1, 0.5])
        # Randomize goal position in right portion of environment
        self.goal_pos = np.array([0.7 + np.random.random() * 0.2, 
                                  0.2 + np.random.random() * 0.6])
        self.episode_steps = 0
        self.episode_reward = 0.0  # Reset episode reward
        self._generate_obstacles()
        
        info = {
            'difficulty': self.difficulty,
            'num_obstacles': len(self.obstacles)
        }
        
        return self._get_observation(), info
    
    def render(self):
        if self.render_mode != "human":
            return
            
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((512, 512))
            pygame.display.set_caption("Adaptive Course")
            
        self.screen.fill((40, 40, 40))
        
        # Draw obstacles
        for x, y, r in self.obstacles:
            pos = (int(x * 512), int(y * 512))
            radius = int(r * 512)
            pygame.draw.circle(self.screen, (150, 150, 150), pos, radius)
        
        # Draw goal
        goal_pos = (int(self.goal_pos[0] * 512), int(self.goal_pos[1] * 512))
        pygame.draw.circle(self.screen, (100, 255, 100), goal_pos, 25)
        
        # Draw agent
        agent_pos = (int(self.agent_pos[0] * 512), int(self.agent_pos[1] * 512))
        pygame.draw.circle(self.screen, (100, 150, 255), agent_pos, 10)
        
        pygame.display.flip()
    
    def close(self):
        if self.screen:
            pygame.quit()


class AdaptiveCourseWrapper(gym.Wrapper):
    """Wrapper for Dreamer compatibility."""
    
    def __init__(self, env):
        super().__init__(env)
        self.last_success = False
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.last_success = info.get('success', False)
        return obs, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        opts = {'last_episode_success': self.last_success}
        if options:
            opts.update(options)
        return self.env.reset(seed=seed, options=opts)