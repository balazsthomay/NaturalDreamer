# DreamerV3 Adaptive Environment Experiment

An experiment exploring how DreamerV3 learns through curriculum learning using a custom adaptive obstacle course environment. Built for my own understanding of the DreamerV3 architecture, not as a contribution to the field.

## Overview

Built on [Natural Dreamer](https://github.com/InexperiencedMe/NaturalDreamer) implementation with:
- **Adaptive Course Environment**: 2D obstacle course adjusting difficulty based on performance
- **Curriculum Learning**: Progressive scaling from simple to complex configurations

## The Environment

- **Agent** (blue): Navigates from start to goal
- **Goal** (green): Randomly positioned target
- **Obstacles** (gray): 0-6 obstacles based on success rate
- **Adaptive Difficulty**: Automatically scales based on recent performance (>70% success increases difficulty)

## Results

1. **Curriculum Learning**: Successfully guided from simple navigation to complex obstacle avoidance
2. **Performance Breakthrough**: Late-training jumps (100+ rewards from around 20) may show strategy discovery  

## Usage

**Training:**
```bash
python main.py --config adaptive-course-balanced.yml
```

**Watch Agent (no training needed, these two checkpoints are in the repo):**
```bash

# Visual (may hang with pygame) -- before training
python watch_agent_play.py --checkpoint checkpoints/adaptive-course_BalancedTraining_2k.pth

# Visual (may hang with pygame) -- after training
python watch_agent_play.py --checkpoint checkpoints/adaptive-course_BalancedTraining_46k.pth

# Text output
python watch_agent_play.py --checkpoint checkpoints/adaptive-course_BalancedTraining_46k.pth --no-render
```
