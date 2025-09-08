import torch
import numpy as np
import time
import sys
from dreamer import Dreamer
from utils import loadConfig, seedEverything
from adaptive_course_env import AdaptiveCourseEnv, AdaptiveCourseWrapper
from envs import EnvironmentStateWrapper, CleanGymWrapper, VectorObservationWrapper, getEnvProperties

def watch_agent(checkpoint_path, config_path, num_episodes=5, use_rendering=True):
    """Watch the trained agent play the game"""
    
    # Load config and setup
    config = loadConfig(config_path)
    seedEverything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Creating environment...")
    # Create environment - try with rendering, fallback to without
    try:
        if use_rendering:
            base_env = AdaptiveCourseEnv(render_mode="human")
        else:
            base_env = AdaptiveCourseEnv(render_mode=None)
        env = EnvironmentStateWrapper(CleanGymWrapper(VectorObservationWrapper(AdaptiveCourseWrapper(base_env))))
        print("Environment created successfully!")
    except Exception as e:
        print(f"Failed to create environment with rendering: {e}")
        print("Falling back to no rendering...")
        base_env = AdaptiveCourseEnv(render_mode=None)
        env = EnvironmentStateWrapper(CleanGymWrapper(VectorObservationWrapper(AdaptiveCourseWrapper(base_env))))
        use_rendering = False
    
    # Get environment properties
    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    
    # Load trained agent
    print("Loading trained agent...")
    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, device, config.dreamer)
    dreamer.loadCheckpoint(checkpoint_path)
    print("Agent loaded successfully!")
    
    print(f"Watching agent play for {num_episodes} episodes...")
    if not use_rendering:
        print("(Running without visual rendering)")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        # Initialize states
        recurrentState = torch.zeros(1, dreamer.recurrentSize, device=device)
        latentState = torch.zeros(1, dreamer.latentSize, device=device)
        action = torch.zeros(1, dreamer.actionSize, device=device)
        
        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
        episode_reward = 0
        step = 0
        max_steps = 200
        
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        # Get the base environment for accessing positions
        current_env = env
        while hasattr(current_env, 'env'):
            current_env = current_env.env
        
        try:
            while step < max_steps:
                # Generate action
                with torch.no_grad():
                    encodedObservation = dreamer.encoder(torch.from_numpy(obs).float().unsqueeze(0).to(device))
                    recurrentState = dreamer.recurrentModel(recurrentState, latentState, action)
                    latentState, _ = dreamer.posteriorNet(torch.cat((recurrentState, encodedObservation.view(1, -1)), -1))
                    action = dreamer.actor(torch.cat((recurrentState, latentState), -1))
                    actionNumpy = action.cpu().numpy().reshape(-1)
                
                # Take step
                result = env.step(actionNumpy)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                elif len(result) == 3:
                    obs, reward, done = result
                    terminated = truncated = done
                    info = {}
                else:
                    obs, reward = result[0], result[1]
                    terminated = truncated = False
                    info = {}
                    
                episode_reward += reward
                step += 1
                
                # Render if available
                if use_rendering and hasattr(current_env, 'render'):
                    current_env.render()
                    time.sleep(0.05)  # Slow down for viewing
                
                # Print progress every 20 steps or on interesting events
                if step % 20 == 0 or abs(reward) > 5:
                    if hasattr(current_env, 'agent_pos') and hasattr(current_env, 'goal_pos'):
                        dist = np.linalg.norm(current_env.agent_pos - current_env.goal_pos)
                        print(f"  Step {step:3d}: reward={reward:6.2f}, total={episode_reward:6.2f}, dist={dist:.3f}")
                    else:
                        print(f"  Step {step:3d}: reward={reward:6.2f}, total={episode_reward:6.2f}")
                
                if terminated or truncated:
                    success = info.get('success', False) if isinstance(info, dict) else False
                    difficulty = info.get('difficulty', 0) if isinstance(info, dict) else 0
                    print(f"  Episode ended: success={success}, difficulty={difficulty:.3f}")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
            
        if step >= max_steps:
            print(f"  Episode reached max steps ({max_steps})")
            
        print(f"Episode {episode + 1} completed: {step} steps, total reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
        
        if episode < num_episodes - 1:
            time.sleep(0.5)  # Brief pause between episodes
    
    # Summary
    if total_rewards:
        avg_reward = np.mean(total_rewards)
        print(f"\n=== Summary ===")
        print(f"Episodes: {len(total_rewards)}")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Rewards: {[f'{r:.1f}' for r in total_rewards]}")
    
    try:
        env.close()
    except:
        pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="adaptive-course-balanced.yml", help="Config file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch")
    parser.add_argument("--no-render", action="store_true", help="Disable visual rendering")
    
    args = parser.parse_args()
    
    watch_agent(args.checkpoint, args.config, args.episodes, not args.no_render)