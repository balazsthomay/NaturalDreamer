import gymnasium as gym
import torch
import argparse
import os
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics, computeEnvironmentComplexityMetrics, plotCurriculumMetrics
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper, VectorObservationWrapper, EnvironmentStateWrapper
from adaptive_course_env import AdaptiveCourseEnv, AdaptiveCourseWrapper
from utils      import saveLossesToCSV, ensureParentFolders
# device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    
    if config.environmentName == 'adaptive-course':
        base_env = AdaptiveCourseEnv(render_mode=None)
        env = EnvironmentStateWrapper(CleanGymWrapper(VectorObservationWrapper(AdaptiveCourseWrapper(base_env))))
        
        base_env_eval = AdaptiveCourseEnv(render_mode="rgb_array") 
        envEvaluation = EnvironmentStateWrapper(CleanGymWrapper(VectorObservationWrapper(AdaptiveCourseWrapper(base_env_eval))))
    else:
        env             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName), (64, 64))))
        envEvaluation   = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName, render_mode="rgb_array"), (64, 64))))
    
    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}")

    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, device, config.dreamer)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    # Curriculum learning tracking
    success_history = []
    use_stratified_sampling = False
    
    iterationsNum = config.gradientSteps // config.replayRatio
    print(f"Starting training: {iterationsNum} iterations, {config.gradientSteps} total gradient steps")
    
    for iteration in range(iterationsNum):
        for _ in range(config.replayRatio):
            # Use stratified sampling if curriculum learning is active
            if use_stratified_sampling and len(dreamer.buffer) > 1000:
                sampledData = dreamer.buffer.sample_stratified(dreamer.config.batchSize, dreamer.config.batchLength)
            else:
                sampledData = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics    = dreamer.worldModelTraining(sampledData)
            behaviorMetrics                     = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}_{suffix}")
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        
        # Update curriculum learning based on recent performance
        if config.environmentName == 'adaptive-course':
            episode_success = (mostRecentScore or 0) > 2.0  # Lowered threshold for early success
            success_history.append(episode_success)
            if len(success_history) > 20:  # Keep recent history
                success_history.pop(0)
            
            # Enable stratified sampling if success rate is reasonable
            if len(success_history) >= 10:
                recent_success_rate = sum(success_history) / len(success_history)
                use_stratified_sampling = 0.3 < recent_success_rate < 0.8  # Use when moderately successful
            
            # Progress update every 10 iterations
            if (iteration + 1) % 10 == 0:
                curr_step = dreamer.totalGradientSteps
                recent_rate = sum(success_history[-5:]) / min(5, len(success_history)) if success_history else 0
                print(f"Iteration {iteration+1}/{iterationsNum} | Steps: {curr_step} | Recent success: {recent_rate:.2f} | Stratified: {use_stratified_sampling}")
        
        if config.saveMetrics:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
            
            # Add curriculum complexity metrics if using adaptive environment
            if config.environmentName == 'adaptive-course':
                complexity_metrics = computeEnvironmentComplexityMetrics(dreamer.buffer)
                metricsBase.update(complexity_metrics)
                
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            
            # Use curriculum plotting for adaptive environments
            if config.environmentName == 'adaptive-course':
                plotCurriculumMetrics(f"{metricsFilename}", dreamer.buffer, savePath=f"{plotFilename}", title=f"{config.environmentName}")
            else:
                plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    main(parser.parse_args().config)
