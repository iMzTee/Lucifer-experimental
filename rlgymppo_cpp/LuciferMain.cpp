// LuciferMain.cpp — RLGymPPO_CPP training entry point for Lucifer 2v2 bot.
//
// Replaces examplemain.cpp. Drop this file into:
//   RLGymPPO_CPP/RLGymPPO_CPP/
// alongside examplemain.cpp, then build.
//
// Hardware target: RTX 2060 (6GB) + i7-9750H (6-core)
// Expected SPS: 40-60k
//
// Run: ./RLGymPPO_CPP (with collision_meshes/ in working directory)

#include <RLGymPPO_CPP/Learner.h>

#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>

// Lucifer headers (placed alongside LuciferMain.cpp at repo root)
#include "LuciferObs.h"
#include "LuciferRewards.h"
#include "LuciferStateSetter.h"
#include "LuciferCurriculum.h"

#include <cstdio>
#include <chrono>
#include <mutex>

using namespace RLGPC;
using namespace RLGSC;

// Simple timeout condition (total env steps, not no-touch)
class TimeoutCondition : public TerminalCondition {
    int maxSteps;
    int stepCount = 0;
public:
    TimeoutCondition(int maxSteps) : maxSteps(maxSteps) {}
    void Reset(const GameState& initialState) override { stepCount = 0; }
    bool IsTerminal(const GameState& currentState) override {
        return ++stepCount >= maxSteps;
    }
};

// ===================================================================
// Global curriculum tracker (shared across threads)
// ===================================================================
static CurriculumTracker g_curriculum(0);

// Mutex for stage transitions that modify non-atomic state
static std::mutex g_curriculumMutex;

// ===================================================================
// Per-step callback (called from worker threads — keep lightweight)
// ===================================================================
void OnStep(GameInst* gameInst, const RLGSC::Gym::StepResult& stepResult,
            Report& gameMetrics) {
    auto& state = stepResult.state;

    for (auto& player : state.players) {
        gameMetrics.AccumAvg("ball_touch_ratio",
            (float)player.ballTouchedStep);
        gameMetrics.AccumAvg("player_speed",
            player.phys.vel.Length() / CommonValues::CAR_MAX_SPEED);
        gameMetrics.AccumAvg("player_boost",
            player.boostFraction);
        gameMetrics.AccumAvg("player_on_ground",
            (float)player.carState.isOnGround);
    }

    gameMetrics.AccumAvg("ball_speed",
        state.ball.vel.Length() / CommonValues::BALL_MAX_SPEED);
    gameMetrics.AccumAvg("ball_height",
        state.ball.pos.z / CommonValues::CEILING_Z);
}

// ===================================================================
// Per-iteration callback (single-threaded, after PPO update)
// ===================================================================
void OnIteration(Learner* learner, Report& allMetrics) {
    // Aggregate game metrics
    auto allGameMetrics = learner->GetAllGameMetrics();

    AvgTracker avgBallTouch{}, avgPlayerSpeed{}, avgBallSpeed{};
    AvgTracker avgPlayerBoost{}, avgOnGround{}, avgBallHeight{};

    for (auto& report : allGameMetrics) {
        avgBallTouch += report.GetAvg("ball_touch_ratio");
        avgPlayerSpeed += report.GetAvg("player_speed");
        avgBallSpeed += report.GetAvg("ball_speed");
        avgPlayerBoost += report.GetAvg("player_boost");
        avgOnGround += report.GetAvg("player_on_ground");
        avgBallHeight += report.GetAvg("ball_height");
    }

    allMetrics["ball_touch_ratio"] = avgBallTouch.Get();
    allMetrics["avg_player_speed"] = avgPlayerSpeed.Get();
    allMetrics["avg_ball_speed"] = avgBallSpeed.Get();
    allMetrics["avg_player_boost"] = avgPlayerBoost.Get();
    allMetrics["avg_on_ground"] = avgOnGround.Get();
    allMetrics["avg_ball_height"] = avgBallHeight.Get();

    // Read PPO metrics (exact key names from PPOLearner.cpp)
    float entropy = allMetrics.Has("Policy Entropy") ? allMetrics["Policy Entropy"] : 99.0f;
    float clipFrac = allMetrics.Has("SB3 Clip Fraction") ? allMetrics["SB3 Clip Fraction"] : 0.0f;
    float meanReward = allMetrics.Has("Mean Reward") ? allMetrics["Mean Reward"] : 0.0f;

    // Print status line
    printf("[Iter] Steps: %llu | Reward: %.3f | Entropy: %.3f | Clip: %.3f | "
           "Stage: %d (%s) | Touch: %.3f | Speed: %.2f\n",
           (unsigned long long)learner->totalTimesteps,
           meanReward, entropy, clipFrac,
           g_curriculum.GetStage(),
           CurriculumTracker::STAGE_NAMES[g_curriculum.GetStage()],
           avgBallTouch.Get(),
           avgPlayerSpeed.Get());

    // Curriculum update
    {
        std::lock_guard<std::mutex> lock(g_curriculumMutex);
        bool changed = g_curriculum.Update(entropy, clipFrac);

        if (g_curriculum.lrReduced) {
            learner->UpdateLearningRates(g_curriculum.policyLR, g_curriculum.criticLR);
        }

        // Stage advance: update reward weights
        // Note: The LuciferReward instances inside existing Gym objects will
        // read the new stage via the atomic g_curriculum.stage when computing
        // rewards. New Gym instances created by the learner will use the
        // updated stage from EnvCreateFunc.
    }
}

// ===================================================================
// Environment creation function
// Called once per game instance (numThreads * numGamesPerThread times)
// ===================================================================
EnvCreateResult EnvCreateFunc() {
    int stage = g_curriculum.GetStage();
    const auto& cfg = STAGE_CONFIGS[stage];

    // Rewards: LuciferReward handles all 13 signals + events + team spirit
    // Pass pointer to curriculum stage for dynamic weight changes
    auto* reward = new LuciferReward(&g_curriculum.stage, stage);

    // Terminal conditions: goal scored OR total timeout
    std::vector<TerminalCondition*> terminals = {
        new GoalScoreCondition(),
        new TimeoutCondition(cfg.timeout),
    };

    // Observation builder
    auto* obs = new LuciferObs();

    // Action parser (Discrete: ~90 actions)
    auto* actionParser = new DiscreteAction();

    // State setter with stage-dependent probabilities
    auto* stateSetter = new LuciferStateSetter(
        cfg.pKickoff, cfg.pGround, cfg.pAerial, cfg.pCeiling);

    // Create match: 2v2
    auto* match = new Match(
        reward,
        terminals,
        obs,
        actionParser,
        stateSetter,
        2,      // teamSize
        true    // spawnOpponents
    );

    auto* gym = new Gym(match, cfg.tickSkip);

    return { match, gym };
}

// ===================================================================
// Main entry point
// ===================================================================
int main(int argc, char* argv[]) {
    printf("=== Lucifer 2v2 Bot — RLGymPPO_CPP Training ===\n");
    printf("Hardware: RTX 2060 (6GB) + i7-9750H (6-core)\n");
    printf("Network: 15.2M params, policy+critic (2048, 2048, 1024, 1024)\n");
    printf("Action space: Discrete(~90)\n");
    printf("Obs space: 127\n\n");

    // Initialize RocketSim
    RocketSim::Init("./collision_meshes");
    printf("[*] RocketSim initialized.\n");

    // Learner configuration
    LearnerConfig cfg = {};

    // Threading: 6-core i7-9750H
    cfg.numThreads = 6;
    cfg.numGamesPerThread = 167;        // ~1000 total envs
    cfg.minInferenceSize = 80;

    // Steps
    cfg.timestepsPerIteration = 200000;
    cfg.expBufferSize = 200000;
    cfg.timestepLimit = 0;              // Unlimited

    // PPO
    cfg.ppo.batchSize = 50000;
    cfg.ppo.miniBatchSize = 50000;
    cfg.ppo.epochs = 2;
    cfg.ppo.policyLR = 2e-4f;
    cfg.ppo.criticLR = 2e-4f;
    cfg.ppo.entCoef = 0.01f;
    cfg.ppo.clipRange = 0.2f;

    // Network architecture: 15.2M params
    cfg.ppo.policyLayerSizes = {2048, 2048, 1024, 1024};
    cfg.ppo.criticLayerSizes = {2048, 2048, 1024, 1024};

    // AMP for 6GB VRAM
    cfg.ppo.autocastLearn = true;

    // GAE
    cfg.gaeLambda = 0.95f;
    cfg.gaeGamma = 0.99f;

    // Pipeline parallelism: collect during learn
    cfg.collectionDuringLearn = true;

    // Returns normalization
    cfg.standardizeReturns = true;
    cfg.standardizeOBS = false;

    // Reward clipping
    cfg.rewardClipRange = 10.0f;

    // Checkpoints
    cfg.checkpointSaveFolder = "Checkpoints_2v2_cpp";
    cfg.checkpointsToKeep = 3;
    cfg.timestepsPerSave = 1000000;     // Save every 1M steps
    cfg.saveFolderAddUnixTimestamp = false;

    // GPU
    cfg.deviceType = LearnerDeviceType::GPU_CUDA;

    // Metrics (wandb)
    cfg.sendMetrics = true;
    cfg.metricsProjectName = "lucifer-2v2";
    cfg.metricsGroupName = "rlgymppo-cpp";
    cfg.metricsRunName = "lucifer-cpp-v1";

    // No rendering during training
    cfg.renderMode = false;

    cfg.randomSeed = 42;

    printf("[*] Config:\n");
    printf("    Threads: %d x %d games = %d total envs\n",
           cfg.numThreads, cfg.numGamesPerThread,
           cfg.numThreads * cfg.numGamesPerThread);
    printf("    Steps/iter: %lld, Batch: %lld, MiniBatch: %lld, Epochs: %d\n",
           (long long)cfg.timestepsPerIteration,
           (long long)cfg.ppo.batchSize,
           (long long)cfg.ppo.miniBatchSize,
           cfg.ppo.epochs);
    printf("    Policy LR: %.2e, Critic LR: %.2e, Ent: %.4f, Clip: %.2f\n",
           cfg.ppo.policyLR, cfg.ppo.criticLR, cfg.ppo.entCoef, cfg.ppo.clipRange);
    printf("    Network: [%d, %d, %d, %d] policy + critic\n",
           cfg.ppo.policyLayerSizes[0], cfg.ppo.policyLayerSizes[1],
           cfg.ppo.policyLayerSizes[2], cfg.ppo.policyLayerSizes[3]);
    printf("    AMP: %s, Pipeline: %s\n",
           cfg.ppo.autocastLearn ? "ON" : "OFF",
           cfg.collectionDuringLearn ? "ON" : "OFF");
    printf("    Stage: %d (%s)\n\n",
           g_curriculum.GetStage(),
           CurriculumTracker::STAGE_NAMES[g_curriculum.GetStage()]);

    // Create learner
    Learner learner(EnvCreateFunc, cfg);
    learner.stepCallback = OnStep;
    learner.iterationCallback = OnIteration;

    printf("[*] Starting training...\n");
    printf("    Target SPS: 40,000 - 60,000\n\n");

    // Start training (blocking)
    learner.Learn();

    return 0;
}
