import gymnasium as gym
import numpy as np

def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()
    return observationShape, actionSize, actionLow, actionHigh

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    
class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs
    
# Simple wrappers for Dreamer integration

class VectorObservationWrapper(gym.ObservationWrapper):
    """Ensures vector observations are float32."""
    
    def observation(self, obs):
        return np.array(obs, dtype=np.float32)


class EnvironmentStateWrapper(gym.Wrapper):
    """Simple curriculum learning tracking."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        
    def step(self, action):
        obs, reward, done = self.env.step(action)
        if done:
            self.episode_count += 1
        return obs, reward, done
    
    def get_curriculum_metrics(self):
        return {'episode_count': self.episode_count}