import attridict
import numpy as np
import torch

# Code comes from SimpleDreamer repo, I only changed some formatting and names, but I should really remake it.
class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observations        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.nextObservations   = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions             = np.empty((self.capacity, actions_size), dtype=np.float32)
        self.rewards             = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones               = np.empty((self.capacity, 1), dtype=np.float32)
        self.environment_state   = np.empty((self.capacity, 18), dtype=np.float32)  # max_obstacles * 3
        self.difficulty_levels   = np.empty((self.capacity, 1), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False
        
    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, action, reward, nextObservation, done, environment_state=None, difficulty=0.0):
        self.observations[self.bufferIndex]     = observation
        self.actions[self.bufferIndex]          = action
        self.rewards[self.bufferIndex]          = reward
        self.nextObservations[self.bufferIndex] = nextObservation
        self.dones[self.bufferIndex]            = done
        self.difficulty_levels[self.bufferIndex] = difficulty
        if environment_state is not None:
            self.environment_state[self.bufferIndex] = environment_state

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"
        sampleIndex = np.random.randint(0, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations         = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        nextObservations    = torch.as_tensor(self.nextObservations[sampleIndex], device=self.device).float()

        actions  = torch.as_tensor(self.actions[sampleIndex], device=self.device)
        rewards  = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
        dones    = torch.as_tensor(self.dones[sampleIndex], device=self.device)

        environment_state = torch.as_tensor(self.environment_state[sampleIndex], device=self.device)
        
        sample = attridict.AttriDict({
            "observations"      : observations,
            "actions"           : actions,
            "rewards"           : rewards,
            "nextObservations"  : nextObservations,
            "dones"             : dones,
            "environment_state" : environment_state})
        return sample
    
    def sample_stratified(self, batchSize, sequenceSize, difficulty_ranges=[(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]):
        """Sample data stratified by difficulty ranges for curriculum learning."""
        samples_per_range = batchSize // len(difficulty_ranges)
        remaining_samples = batchSize % len(difficulty_ranges)
        
        all_indices = []
        current_difficulties = self.difficulty_levels[:len(self)] if not self.full else self.difficulty_levels
        
        for i, (min_diff, max_diff) in enumerate(difficulty_ranges):
            valid_indices = np.where((current_difficulties >= min_diff) & (current_difficulties < max_diff))[0]
            
            if len(valid_indices) == 0:
                continue
                
            range_samples = samples_per_range + (1 if i < remaining_samples else 0)
            range_samples = min(range_samples, len(valid_indices) - sequenceSize + 1)
            
            if range_samples > 0:
                selected = np.random.choice(valid_indices[:-sequenceSize+1], range_samples, replace=False)
                all_indices.extend(selected)
        
        if len(all_indices) < batchSize:
            return self.sample(batchSize, sequenceSize)
            
        sampleIndex = np.array(all_indices[:batchSize]).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)
        sampleIndex = (sampleIndex + sequenceLength) % self.capacity
        
        observations = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        nextObservations = torch.as_tensor(self.nextObservations[sampleIndex], device=self.device).float()
        actions = torch.as_tensor(self.actions[sampleIndex], device=self.device)
        rewards = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
        dones = torch.as_tensor(self.dones[sampleIndex], device=self.device)
        environment_state = torch.as_tensor(self.environment_state[sampleIndex], device=self.device)
        
        return attridict.AttriDict({
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "nextObservations": nextObservations,
            "dones": dones,
            "environment_state": environment_state})
    
    def track_environment_changes(self):
        """Track when environment configuration changes occur."""
        if len(self) < 2:
            return []
        
        current_states = self.environment_state[:len(self)] if not self.full else self.environment_state
        changes = []
        
        for i in range(1, len(current_states)):
            if not np.allclose(current_states[i], current_states[i-1], atol=1e-6):
                changes.append(i)
        
        return changes
