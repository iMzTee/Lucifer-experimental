#pragma once

#include <RLGymSim_CPP/Utils/RewardFunctions/RewardFunction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/ZeroSumReward.h>
#include <RLGymSim_CPP/Utils/CommonValues.h>
#include <RLGymSim_CPP/Utils/Gamestates/GameState.h>
#include <cmath>
#include <algorithm>
#include <map>

// LuciferRewards — 13 continuous reward signals + event rewards for Lucifer bot.
//
// Stage weights control the contribution of each signal.
// Team spirit blends individual vs. team-mean reward.
// ZeroSumReward wraps everything for proper team-based reward shaping.

using namespace RLGSC;

// ===================================================================
// R1: VelocityBallToGoal
// dot(norm(opp_goal - ball), ball_vel / 6000)
// ===================================================================
class R1_VelBallToGoal : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        Vec oppGoal = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
        Vec diff = oppGoal - state.ball.pos;
        float dist = diff.Length();
        if (dist < 1e-6f) return 0.0f;
        Vec dir = diff / dist;
        return dir.Dot(state.ball.vel) / CommonValues::BALL_MAX_SPEED;
    }
};

// ===================================================================
// R2: BallGoalDistancePotential
// exp(-d_opp/6000) - exp(-d_own/6000)
// ===================================================================
class R2_BallGoalDistPotential : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        Vec oppGoal = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
        Vec ownGoal = (player.team == Team::BLUE)
            ? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
        float dOpp = (state.ball.pos - oppGoal).Length();
        float dOwn = (state.ball.pos - ownGoal).Length();
        return std::exp(-dOpp / 6000.0f) - std::exp(-dOwn / 6000.0f);
    }
};

// ===================================================================
// R3: TouchQuality
// touched * height_term * speed_term * wall_factor(1.5)
// ===================================================================
class R3_TouchQuality : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        if (!player.ballTouchedStep) return 0.0f;

        float ballZ = state.ball.pos.z;
        float ballSpeed = state.ball.vel.Length();

        // Height term: 1 + cbrt(max(0, z - 150) / 2044) * 2
        float heightTerm = 1.0f + std::cbrt(std::max(0.0f, ballZ - 150.0f) / 2044.0f) * 2.0f;

        // Speed term: 0.5 + 0.5 * clamp(speed / 2300, 0, 2)
        float speedTerm = 0.5f + 0.5f * std::min(ballSpeed / 2300.0f, 2.0f);

        // Wall factor: 1.5 if ball near side/back wall
        float wallFactor = 1.0f;
        if (std::abs(state.ball.pos.x) > 3800.0f || std::abs(state.ball.pos.y) > 4800.0f)
            wallFactor = 1.5f;

        return heightTerm * speedTerm * wallFactor;
    }
};

// ===================================================================
// R4: PlayerBallProximityVelocity (closest on team only)
// dot(vel, to_ball_dir) / 2300, only for closest teammate
// ===================================================================
class R4_PlayerBallProxVel : public RewardFunction {
public:
    std::vector<float> GetAllRewards(const GameState& state,
                                     const ActionSet& prevActions, bool final) override {
        int n = state.players.size();
        std::vector<float> rewards(n, 0.0f);

        // Find closest player to ball per team
        std::map<Team, int> closestIdx;
        std::map<Team, float> closestDist;
        for (int i = 0; i < n; i++) {
            float d = (state.ball.pos - state.players[i].phys.pos).Length();
            Team t = state.players[i].team;
            if (closestDist.find(t) == closestDist.end() || d < closestDist[t]) {
                closestDist[t] = d;
                closestIdx[t] = i;
            }
        }

        // Reward only the closest on each team
        for (auto& [team, idx] : closestIdx) {
            const auto& p = state.players[idx];
            Vec toBall = state.ball.pos - p.phys.pos;
            float dist = toBall.Length();
            if (dist < 1e-6f) continue;
            Vec dir = toBall / dist;
            float speedToward = std::max(0.0f, p.phys.vel.Dot(dir));
            rewards[idx] = speedToward / CommonValues::CAR_MAX_SPEED;
        }

        return rewards;
    }
};

// ===================================================================
// R5: KickoffReward
// Speed toward ball + distance bonus during kickoff
// ===================================================================
class R5_KickoffReward : public RewardFunction {
    bool isKickoff = false;

public:
    void Reset(const GameState& initialState) override {
        isKickoff = CheckKickoff(initialState);
    }

    void PreStep(const GameState& state) override {
        if (isKickoff && state.ball.vel.Length() >= 100.0f)
            isKickoff = false;
    }

    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        if (!isKickoff) return 0.0f;

        Vec toBall = state.ball.pos - player.phys.pos;
        float dist = toBall.Length();
        if (dist < 1e-6f) return 0.0f;
        Vec dir = toBall / dist;

        float speedToward = std::max(0.0f, player.phys.vel.Dot(dir) / 2300.0f);
        float distBonus = std::exp(-dist / 800.0f);

        return speedToward + distBonus;
    }

private:
    bool CheckKickoff(const GameState& state) {
        return std::abs(state.ball.pos.x) < 50.0f
            && std::abs(state.ball.pos.y) < 50.0f
            && state.ball.pos.z < 120.0f;
    }
};

// ===================================================================
// R6: DefensivePositioning (support role only)
// Alignment * gaussian(dist_ratio, 0.7, 0.15) for non-closest
// ===================================================================
class R6_DefensivePos : public RewardFunction {
public:
    std::vector<float> GetAllRewards(const GameState& state,
                                     const ActionSet& prevActions, bool final) override {
        int n = state.players.size();
        std::vector<float> rewards(n, 0.0f);

        // Find closest to ball per team
        std::map<Team, float> closestDist;
        std::map<Team, int> closestIdx;
        for (int i = 0; i < n; i++) {
            float d = (state.ball.pos - state.players[i].phys.pos).Length();
            Team t = state.players[i].team;
            if (closestDist.find(t) == closestDist.end() || d < closestDist[t]) {
                closestDist[t] = d;
                closestIdx[t] = i;
            }
        }

        for (int i = 0; i < n; i++) {
            const auto& p = state.players[i];
            // Only support role (not closest on team)
            if (closestIdx.count(p.team) && closestIdx[p.team] == i)
                continue;

            Vec ownGoal = (p.team == Team::BLUE)
                ? Vec(0, -CommonValues::BACK_WALL_Y, 0)
                : Vec(0, CommonValues::BACK_WALL_Y, 0);

            Vec g2b = state.ball.pos - ownGoal;
            Vec g2p = p.phys.pos - ownGoal;
            float g2bLen = g2b.Length();
            float g2pLen = g2p.Length();
            if (g2bLen < 1e-6f || g2pLen < 1e-6f) continue;

            float align = std::max(0.0f, g2p.Dot(g2b) / (g2bLen * g2pLen));
            float distRatio = g2pLen / g2bLen;
            float gaussian = std::exp(-((distRatio - 0.7f) * (distRatio - 0.7f))
                                      / (2.0f * 0.15f * 0.15f));

            rewards[i] = align * gaussian;
        }

        return rewards;
    }
};

// ===================================================================
// R7: BoostEfficiency
// sqrt(boost_gained) * pad_mult(2x small), clamped 0.5
// ===================================================================
class R7_BoostEfficiency : public RewardFunction {
    std::map<uint32_t, float> prevBoost;

public:
    void Reset(const GameState& initialState) override {
        prevBoost.clear();
        for (const auto& p : initialState.players)
            prevBoost[p.carId] = p.boostFraction;
    }

    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        float prev = prevBoost.count(player.carId) ? prevBoost[player.carId] : 0.0f;
        float gained = std::max(0.0f, player.boostFraction - prev);
        prevBoost[player.carId] = player.boostFraction;

        if (gained <= 0.001f) return 0.0f;

        // Small pad = gain between 0.01 and 0.15 (~12% = 0.12)
        float padMult = (gained > 0.01f && gained <= 0.15f) ? 2.0f : 1.0f;
        return std::min(std::sqrt(gained) * padMult, 0.5f);
    }
};

// ===================================================================
// R8: DemoAttempt
// proximity_bonus * speed_to_opponent * (speed > 1500)
// ===================================================================
class R8_DemoAttempt : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        float playerSpeed = player.phys.vel.Length();
        if (playerSpeed <= 1500.0f) return 0.0f;

        float nearestDist = 1e9f;
        Vec nearestVec;
        for (const auto& p : state.players) {
            if (p.team == player.team) continue;
            Vec toOpp = p.phys.pos - player.phys.pos;
            float d = toOpp.Length();
            if (d < nearestDist) {
                nearestDist = d;
                nearestVec = toOpp;
            }
        }

        if (nearestDist > 1e8f) return 0.0f;

        Vec dir = nearestVec / (nearestDist + 1e-6f);
        float speedToOpp = std::max(0.0f, player.phys.vel.Dot(dir) / 2300.0f);
        float proxBonus = std::exp(-nearestDist / 500.0f);

        return proxBonus * speedToOpp;
    }
};

// ===================================================================
// R9: AirControl — max(dribble, aerial_facing)
// ===================================================================
class R9_AirControl : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        Vec toBall = state.ball.pos - player.phys.pos;
        float toBallDist = toBall.Length();
        Vec toBallDir = (toBallDist > 1e-6f) ? (toBall / toBallDist) : Vec(0, 0, 0);

        float ballZ = state.ball.pos.z;
        float carZ = player.phys.pos.z;

        // Dribble: ball above car, close overhead, car on ground
        float dribble = 0.0f;
        if (player.carState.isOnGround && ballZ > carZ + 60.0f) {
            float xyDx = state.ball.pos.x - player.phys.pos.x;
            float xyDy = state.ball.pos.y - player.phys.pos.y;
            float xyDist = std::sqrt(xyDx * xyDx + xyDy * xyDy);
            if (xyDist < 180.0f) {
                float prox = std::max(0.0f, 1.0f - xyDist / 180.0f);
                float ht = std::min(1.0f, std::max(0.0f,
                    (ballZ - carZ - 60.0f) / 250.0f));
                dribble = prox * (0.3f + 0.7f * ht);
            }
        }

        // Aerial: off ground, facing ball
        float aerial = 0.0f;
        if (!player.carState.isOnGround) {
            float facingBall = std::max(0.0f,
                player.phys.rotMat.forward.Dot(toBallDir));
            aerial = facingBall;
        }

        return std::max(dribble, aerial);
    }
};

// ===================================================================
// R10: FlipResetDetector
// touched * airborne * (up_z < -0.5) * 10.0
// ===================================================================
class R10_FlipReset : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        if (!player.ballTouchedStep) return 0.0f;
        if (player.carState.isOnGround) return 0.0f;
        if (player.phys.rotMat.up.z >= -0.5f) return 0.0f;
        return 10.0f;
    }
};

// ===================================================================
// R11: AngularVelocity
// norm(ang_vel) / (6*pi)
// ===================================================================
class R11_AngularVel : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        return player.phys.angVel.Length() / (6.0f * M_PI);
    }
};

// ===================================================================
// R12: Speed + Anti-Passive Penalty (smooth ramp)
// speed/2300 - smooth_penalty(0.03)
// Penalty: max(0, 1 - speed/600) * 0.03 when on ground and alive
// ===================================================================
class R12_SpeedAntiPassive : public RewardFunction {
public:
    float GetReward(const PlayerData& player, const GameState& state,
                    const Action& prevAction) override {
        float speed = player.phys.vel.Length();
        float speedRatio = speed / CommonValues::CAR_MAX_SPEED;

        // Smooth penalty: linearly ramps from -0.03 (stationary) to 0 (at 600 uu/s)
        float penalty = 0.0f;
        if (!player.carState.isDemoed && player.carState.isOnGround) {
            penalty = std::max(0.0f, 1.0f - speed / 600.0f) * 0.03f;
        }

        return speedRatio - penalty;
    }
};

// ===================================================================
// R13: BallAcceleration
// (ball_speed - prev_ball_speed) / 6000 on touch
// ===================================================================
class R13_BallAccel : public RewardFunction {
    float prevBallSpeed = 0.0f;

public:
    void Reset(const GameState& initialState) override {
        prevBallSpeed = initialState.ball.vel.Length();
    }

    void PreStep(const GameState& state) override {
        // Update after all rewards computed (PreStep is called before GetReward)
        // We store current for comparison, then update in GetAllRewards
    }

    std::vector<float> GetAllRewards(const GameState& state,
                                     const ActionSet& prevActions, bool final) override {
        int n = state.players.size();
        std::vector<float> rewards(n, 0.0f);

        float curBallSpeed = state.ball.vel.Length();
        float accel = std::max(0.0f, curBallSpeed - prevBallSpeed);
        float accelNorm = accel / CommonValues::BALL_MAX_SPEED;
        prevBallSpeed = curBallSpeed;

        for (int i = 0; i < n; i++) {
            if (state.players[i].ballTouchedStep)
                rewards[i] = accelNorm;
        }

        return rewards;
    }
};

// ===================================================================
// Stage weight configuration
// ===================================================================

struct LuciferStageConfig {
    // 13 continuous reward weights [R1..R13]
    float weights[13];
    // Event weights
    float goalW, concedeW, touchW, shotW, saveW, demoW;
    // Team spirit
    float teamSpirit;
    // Tick skip and timeout
    int tickSkip;
    int timeout;
    // State setter probabilities
    float pKickoff, pGround, pAerial, pCeiling;
};

static const LuciferStageConfig STAGE_CONFIGS[4] = {
    // Stage 0: Foundations
    {
        {2.0f, 1.0f, 2.0f, 2.0f, 3.0f, 0.5f, 0.5f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 1.5f},
        10.0f, -7.0f, 0.5f, 3.0f, 5.0f, 8.0f,
        0.0f,
        8, 300,
        1.0f, 0.0f, 0.0f, 0.0f
    },
    // Stage 1: Game Play
    {
        {5.0f, 3.0f, 3.0f, 1.5f, 2.0f, 1.5f, 1.0f, 0.5f, 1.5f, 0.0f, 0.005f, 0.5f, 2.0f},
        10.0f, -7.0f, 0.5f, 3.0f, 5.0f, 8.0f,
        0.3f,
        8, 400,
        0.5f, 0.3f, 0.2f, 0.0f
    },
    // Stage 2: Mechanics
    {
        {5.0f, 3.0f, 3.0f, 1.0f, 2.0f, 1.5f, 1.0f, 1.0f, 2.0f, 5.0f, 0.005f, 0.3f, 2.0f},
        10.0f, -7.0f, 0.5f, 3.0f, 5.0f, 8.0f,
        0.5f,
        4, 600,
        0.3f, 0.2f, 0.4f, 0.1f
    },
    // Stage 3: Mastery
    {
        {5.0f, 3.0f, 3.0f, 0.5f, 2.0f, 1.5f, 1.0f, 1.0f, 2.0f, 10.0f, 0.005f, 0.2f, 2.0f},
        10.0f, -7.0f, 0.5f, 3.0f, 5.0f, 8.0f,
        0.6f,
        2, 1200,
        0.3f, 0.25f, 0.35f, 0.1f
    }
};

// ===================================================================
// LuciferReward — combines all 13 signals + events + team spirit
// ===================================================================

class LuciferReward : public RewardFunction {
    // Owned reward functions
    R1_VelBallToGoal r1;
    R2_BallGoalDistPotential r2;
    R3_TouchQuality r3;
    R4_PlayerBallProxVel r4;
    R5_KickoffReward r5;
    R6_DefensivePos r6;
    R7_BoostEfficiency r7;
    R8_DemoAttempt r8;
    R9_AirControl r9;
    R10_FlipReset r10;
    R11_AngularVel r11;
    R12_SpeedAntiPassive r12;
    R13_BallAccel r13;
    EventReward events;

    // Pointer to external atomic stage (from CurriculumTracker)
    // Allows dynamic stage changes without recreating reward objects
    const std::atomic<int>* stagePtr;
    int fallbackStage;

public:
    // Constructor with external stage pointer (preferred — reads curriculum stage dynamically)
    LuciferReward(const std::atomic<int>* externalStage, int initialStage = 0)
        : events(EventReward::WeightScales{
            .goal = STAGE_CONFIGS[initialStage].goalW,
            .teamGoal = 0.0f,
            .concede = STAGE_CONFIGS[initialStage].concedeW,
            .touch = STAGE_CONFIGS[initialStage].touchW,
            .shot = STAGE_CONFIGS[initialStage].shotW,
            .save = STAGE_CONFIGS[initialStage].saveW,
            .demo = STAGE_CONFIGS[initialStage].demoW
          }),
          stagePtr(externalStage),
          fallbackStage(initialStage) {}

    // Constructor without external pointer (uses fixed stage)
    LuciferReward(int initialStage = 0)
        : LuciferReward(nullptr, initialStage) {}

    int GetStage() const {
        return stagePtr ? stagePtr->load() : fallbackStage;
    }

    void Reset(const GameState& initialState) override {
        r1.Reset(initialState);
        r2.Reset(initialState);
        r3.Reset(initialState);
        r4.Reset(initialState);
        r5.Reset(initialState);
        r6.Reset(initialState);
        r7.Reset(initialState);
        r8.Reset(initialState);
        r9.Reset(initialState);
        r10.Reset(initialState);
        r11.Reset(initialState);
        r12.Reset(initialState);
        r13.Reset(initialState);
        events.Reset(initialState);
        prevEvents.clear();
    }

    void PreStep(const GameState& state) override {
        r1.PreStep(state);
        r2.PreStep(state);
        r3.PreStep(state);
        r4.PreStep(state);
        r5.PreStep(state);
        r6.PreStep(state);
        r7.PreStep(state);
        r8.PreStep(state);
        r9.PreStep(state);
        r10.PreStep(state);
        r11.PreStep(state);
        r12.PreStep(state);
        r13.PreStep(state);
        events.PreStep(state);
    }

    std::vector<float> GetAllRewards(const GameState& state,
                                     const ActionSet& prevActions, bool final) override {
        const auto& cfg = STAGE_CONFIGS[GetStage()];
        int n = state.players.size();

        // Compute batch rewards for multi-agent signals
        auto r4_all = r4.GetAllRewards(state, prevActions, final);
        auto r6_all = r6.GetAllRewards(state, prevActions, final);
        auto r13_all = r13.GetAllRewards(state, prevActions, final);
        auto ev_all = events.GetAllRewards(state, prevActions, final);

        std::vector<float> rewards(n, 0.0f);

        for (int i = 0; i < n; i++) {
            const auto& p = state.players[i];
            float r = 0.0f;

            // R1-R3: per-player
            if (cfg.weights[0] > 0) r += cfg.weights[0] * r1.GetReward(p, state, prevActions[i]);
            if (cfg.weights[1] > 0) r += cfg.weights[1] * r2.GetReward(p, state, prevActions[i]);
            if (cfg.weights[2] > 0) r += cfg.weights[2] * r3.GetReward(p, state, prevActions[i]);

            // R4: batch (closest on team)
            if (cfg.weights[3] > 0) r += cfg.weights[3] * r4_all[i];

            // R5: per-player
            if (cfg.weights[4] > 0) r += cfg.weights[4] * r5.GetReward(p, state, prevActions[i]);

            // R6: batch (support role)
            if (cfg.weights[5] > 0) r += cfg.weights[5] * r6_all[i];

            // R7-R12: per-player
            if (cfg.weights[6] > 0) r += cfg.weights[6] * r7.GetReward(p, state, prevActions[i]);
            if (cfg.weights[7] > 0) r += cfg.weights[7] * r8.GetReward(p, state, prevActions[i]);
            if (cfg.weights[8] > 0) r += cfg.weights[8] * r9.GetReward(p, state, prevActions[i]);
            if (cfg.weights[9] > 0) r += cfg.weights[9] * r10.GetReward(p, state, prevActions[i]);
            if (cfg.weights[10] > 0) r += cfg.weights[10] * r11.GetReward(p, state, prevActions[i]);
            if (cfg.weights[11] > 0) r += cfg.weights[11] * r12.GetReward(p, state, prevActions[i]);

            // R13: batch (ball accel)
            if (cfg.weights[12] > 0) r += cfg.weights[12] * r13_all[i];

            // Events
            r += ev_all[i];

            rewards[i] = r;
        }

        // Team spirit blending
        float ts = cfg.teamSpirit;
        if (ts > 0.0f) {
            // Compute team means
            float blueSum = 0, orangeSum = 0;
            int blueN = 0, orangeN = 0;
            for (int i = 0; i < n; i++) {
                if (state.players[i].team == Team::BLUE) {
                    blueSum += rewards[i];
                    blueN++;
                } else {
                    orangeSum += rewards[i];
                    orangeN++;
                }
            }
            float blueMean = blueN > 0 ? blueSum / blueN : 0;
            float orangeMean = orangeN > 0 ? orangeSum / orangeN : 0;

            for (int i = 0; i < n; i++) {
                float teamMean = (state.players[i].team == Team::BLUE) ? blueMean : orangeMean;
                rewards[i] = (1.0f - ts) * rewards[i] + ts * teamMean;
            }
        }

        return rewards;
    }
};
