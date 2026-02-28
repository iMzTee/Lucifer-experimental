#pragma once

#include <RLGymSim_CPP/Utils/StateSetters/StateSetter.h>
#include <RLGymSim_CPP/Utils/Gamestates/GameState.h>
#include <RLGymSim_CPP/Utils/CommonValues.h>
#include <RocketSim.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <atomic>

// LuciferStateSetter — MixedStateSetter with 4 scenarios.
//
// Each reset randomly picks a scenario based on configured probabilities:
//   - Kickoff:        Ball at center, cars at standard kickoff positions
//   - Random Ground:  Ball in play, cars on ground with random boost
//   - Aerial:         Ball in air, cars on ground with high boost
//   - Ceiling:        Ball high, one car per team near ceiling
//
// Probabilities change per curriculum stage.
// Thread-safe: each instance gets its own RNG with unique seed.

using namespace RLGSC;

class LuciferStateSetter : public StateSetter {
public:
    float pKickoff, pGround, pAerial;
    // pCeiling = 1 - pKickoff - pGround - pAerial

    static std::atomic<int> seedCounter;

    LuciferStateSetter(float kickoff = 1.0f, float ground = 0.0f,
                       float aerial = 0.0f, float ceiling = 0.0f)
    {
        float total = kickoff + ground + aerial + ceiling;
        pKickoff = kickoff / total;
        pGround = ground / total;
        pAerial = aerial / total;

        // Thread-safe unique seed per instance
        int seed = seedCounter.fetch_add(1) * 7919 + 42;
        rng.seed(seed);
    }

    void SetProbabilities(float kickoff, float ground, float aerial, float ceiling) {
        float total = kickoff + ground + aerial + ceiling;
        pKickoff = kickoff / total;
        pGround = ground / total;
        pAerial = aerial / total;
    }

    GameState ResetState(Arena* arena) override {
        float r = RandFloat(0.0f, 1.0f);

        if (r < pKickoff) {
            SetKickoff(arena);
        } else if (r < pKickoff + pGround) {
            SetRandomGround(arena);
        } else if (r < pKickoff + pGround + pAerial) {
            SetAerial(arena);
        } else {
            SetCeiling(arena);
        }

        return GameState(arena);
    }

private:
    std::mt19937 rng;

    // Standard RL kickoff positions (blue side)
    static constexpr float KICKOFF_POS[][3] = {
        {-2048.0f, -2560.0f, 17.0f},
        { 2048.0f, -2560.0f, 17.0f},
        { -256.0f, -3840.0f, 17.0f},
        {  256.0f, -3840.0f, 17.0f},
        {    0.0f, -4608.0f, 17.0f},
    };
    static constexpr int NUM_KICKOFF_POS = 5;

    float RandFloat(float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    }

    int RandInt(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    }

    // Random sample of k indices from [0, n) without replacement
    std::vector<int> SampleIndices(int n, int k) {
        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(k);
        return indices;
    }

    void SetKickoff(Arena* arena) {
        // Ball at center
        BallState bs = {};
        bs.pos = Vec(0, 0, 93);
        bs.vel = Vec(0, 0, 0);
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);

        auto cars = arena->GetCars();
        std::vector<Car*> blue, orange;
        for (auto* c : cars) {
            if (c->team == Team::BLUE) blue.push_back(c);
            else orange.push_back(c);
        }

        auto blueIdx = SampleIndices(NUM_KICKOFF_POS, (int)blue.size());
        auto orangeIdx = SampleIndices(NUM_KICKOFF_POS, (int)orange.size());

        for (int i = 0; i < (int)blue.size(); i++) {
            CarState cs = {};
            cs.pos = Vec(KICKOFF_POS[blueIdx[i]][0],
                         KICKOFF_POS[blueIdx[i]][1],
                         KICKOFF_POS[blueIdx[i]][2]);
            // Face toward ball (positive Y) — Angle(yaw, pitch, roll)
            cs.rotMat = Angle(M_PI / 2.0f, 0, 0).ToRotMat();
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = 0.33f;
            cs.isOnGround = true;
            blue[i]->SetState(cs);
        }

        for (int i = 0; i < (int)orange.size(); i++) {
            CarState cs = {};
            // Mirror: negate X and Y
            cs.pos = Vec(-KICKOFF_POS[orangeIdx[i]][0],
                         -KICKOFF_POS[orangeIdx[i]][1],
                          KICKOFF_POS[orangeIdx[i]][2]);
            cs.rotMat = Angle(-M_PI / 2.0f, 0, 0).ToRotMat();
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = 0.33f;
            cs.isOnGround = true;
            orange[i]->SetState(cs);
        }
    }

    void SetRandomGround(Arena* arena) {
        BallState bs = {};
        bs.pos = Vec(RandFloat(-3000, 3000), RandFloat(-4000, 4000), 93);
        bs.vel = Vec(RandFloat(-1500, 1500), RandFloat(-1500, 1500), 0);
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);

        auto cars = arena->GetCars();
        for (auto* c : cars) {
            CarState cs = {};
            if (c->team == Team::BLUE) {
                cs.pos = Vec(RandFloat(-3500, 3500), RandFloat(-4500, 0), 17);
            } else {
                cs.pos = Vec(RandFloat(-3500, 3500), RandFloat(0, 4500), 17);
            }
            float yaw = RandFloat(-M_PI, M_PI);
            cs.rotMat = Angle(yaw, 0, 0).ToRotMat();
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = RandFloat(0.0f, 1.0f);
            cs.isOnGround = true;
            c->SetState(cs);
        }
    }

    void SetAerial(Arena* arena) {
        float z = RandFloat(400, 1600);
        BallState bs = {};
        bs.pos = Vec(RandFloat(-2500, 2500), RandFloat(-2500, 2500), z);
        bs.vel = Vec(RandFloat(-400, 400), RandFloat(-400, 400), RandFloat(-200, 200));
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);

        auto cars = arena->GetCars();
        for (auto* c : cars) {
            CarState cs = {};
            if (c->team == Team::BLUE) {
                cs.pos = Vec(RandFloat(-3000, 3000), RandFloat(-4000, 0), 17);
                cs.rotMat = Angle(M_PI / 2.0f, 0, 0).ToRotMat();
            } else {
                cs.pos = Vec(RandFloat(-3000, 3000), RandFloat(0, 4000), 17);
                cs.rotMat = Angle(-M_PI / 2.0f, 0, 0).ToRotMat();
            }
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = RandFloat(0.7f, 1.0f);
            cs.isOnGround = true;
            c->SetState(cs);
        }
    }

    void SetCeiling(Arena* arena) {
        float z = RandFloat(1000, 1800);
        BallState bs = {};
        bs.pos = Vec(RandFloat(-2000, 2000), RandFloat(-2000, 2000), z);
        bs.vel = Vec(RandFloat(-300, 300), RandFloat(-300, 300), RandFloat(-100, 200));
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);

        auto cars = arena->GetCars();
        std::vector<Car*> blue, orange;
        for (auto* c : cars) {
            if (c->team == Team::BLUE) blue.push_back(c);
            else orange.push_back(c);
        }

        // First blue car near ceiling
        if (!blue.empty()) {
            CarState cs = {};
            cs.pos = Vec(RandFloat(-2000, 2000), RandFloat(-3000, 0), 1900);
            cs.rotMat = Angle(M_PI / 2.0f, M_PI, 0).ToRotMat(); // Upside down, facing +Y
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = RandFloat(0.5f, 1.0f);
            cs.isOnGround = false;
            blue[0]->SetState(cs);
        }
        // Second blue car on ground
        if (blue.size() > 1) {
            CarState cs = {};
            cs.pos = Vec(RandFloat(-3000, 3000), RandFloat(-4000, -2000), 17);
            cs.rotMat = Angle(M_PI / 2.0f, 0, 0).ToRotMat();
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = RandFloat(0.3f, 0.8f);
            cs.isOnGround = true;
            blue[1]->SetState(cs);
        }

        // First orange car near ceiling
        if (!orange.empty()) {
            CarState cs = {};
            cs.pos = Vec(RandFloat(-2000, 2000), RandFloat(0, 3000), 1900);
            cs.rotMat = Angle(-M_PI / 2.0f, M_PI, 0).ToRotMat(); // Upside down, facing -Y
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = RandFloat(0.5f, 1.0f);
            cs.isOnGround = false;
            orange[0]->SetState(cs);
        }
        // Second orange car on ground
        if (orange.size() > 1) {
            CarState cs = {};
            cs.pos = Vec(RandFloat(-3000, 3000), RandFloat(2000, 4000), 17);
            cs.rotMat = Angle(-M_PI / 2.0f, 0, 0).ToRotMat();
            cs.vel = Vec(0, 0, 0);
            cs.angVel = Vec(0, 0, 0);
            cs.boost = RandFloat(0.3f, 0.8f);
            cs.isOnGround = true;
            orange[1]->SetState(cs);
        }
    }
};

// Static member initialization
std::atomic<int> LuciferStateSetter::seedCounter{0};
