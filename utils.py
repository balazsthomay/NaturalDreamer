import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import os
import attridict
import gymnasium as gym
import csv
import pandas as pd
import plotly.graph_objects as pgo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = findFile(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def getEnvProperties(env):
    observationShape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discreteActionBool = True
        actionSize = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discreteActionBool = False
        actionSize = env.action_space.shape[0]
    else:
        raise Exception
    return observationShape, discreteActionBool, actionSize


def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue
        
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))
        
        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize
    
    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


def computeEnvironmentComplexityMetrics(buffer):
    """Compute environment complexity metrics from buffer data."""
    if len(buffer) == 0:
        return {}
    
    difficulties = buffer.difficulty_levels[:len(buffer)] if not buffer.full else buffer.difficulty_levels
    difficulties = difficulties.flatten()
    
    changes = buffer.track_environment_changes()
    
    return {
        'mean_difficulty': float(np.mean(difficulties)),
        'std_difficulty': float(np.std(difficulties)),
        'min_difficulty': float(np.min(difficulties)),
        'max_difficulty': float(np.max(difficulties)),
        'difficulty_progression': float(np.mean(np.diff(difficulties)) if len(difficulties) > 1 else 0),
        'num_env_changes': len(changes),
        'change_frequency': len(changes) / len(buffer) if len(buffer) > 0 else 0
    }


def plotCurriculumMetrics(filename, buffer, title="Curriculum Progress", savePath="curriculumPlot"):
    """Plot curriculum learning metrics alongside training metrics."""
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    data = pd.read_csv(filename)
    
    # Get environment complexity metrics
    complexity_metrics = computeEnvironmentComplexityMetrics(buffer)
    
    # Create subplot figure
    fig = pgo.Figure()
    
    # Plot training metrics
    colors = ["gold", "coral", "dodgerblue", "forestgreen", "mediumorchid"]
    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue
        smoothed_data = data[column].rolling(window=10, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column}",
            line=dict(color=colors[idx % len(colors)], width=2),
            yaxis="y"))
    
    # Add difficulty progression if buffer has data
    if len(buffer) > 0:
        difficulties = buffer.difficulty_levels[:len(buffer)] if not buffer.full else buffer.difficulty_levels
        difficulties = difficulties.flatten()
        
        # Create x-axis aligned with gradient steps (approximate)
        if len(data) > 0:
            x_difficulty = np.linspace(data["gradientSteps"].iloc[0], data["gradientSteps"].iloc[-1], len(difficulties))
            
            fig.add_trace(pgo.Scatter(
                x=x_difficulty, y=difficulties, mode='lines',
                name="Difficulty Level",
                line=dict(color="red", width=3, dash="dash"),
                yaxis="y2"))
    
    # Update layout with dual y-axis
    fig.update_layout(
        title=dict(text=f"{title}<br>Mean Difficulty: {complexity_metrics.get('mean_difficulty', 0):.3f}, Changes: {complexity_metrics.get('num_env_changes', 0)}",
                   x=0.5, font=dict(size=20)),
        xaxis=dict(title="Gradient Steps"),
        yaxis=dict(title="Training Metrics", side="left"),
        yaxis2=dict(title="Difficulty Level", side="right", overlaying="y", range=[0, 1]),
        template="plotly_dark",
        height=800,
        width=1200,
        legend=dict(x=0.02, y=0.98)
    )
    
    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns


def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)


class Moments(nn.Module):
    def __init__( self, device, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()
