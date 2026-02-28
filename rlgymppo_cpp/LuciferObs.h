#pragma once

#include <RLGymSim_CPP/Utils/OBSBuilders/OBSBuilder.h>
#include <RLGymSim_CPP/Utils/BasicTypes/Lists.h>
#include <RLGymSim_CPP/Utils/CommonValues.h>
#include <RLGymSim_CPP/Utils/Gamestates/GameState.h>
#include <RLGymSim_CPP/Utils/Gamestates/PlayerData.h>

// LuciferObs — 127-element observation builder for 2v2 Lucifer bot.
//
// Layout (per agent, perspective-normalized):
//   [0-8]    Ball: pos(3), lin_vel(3), ang_vel(3)
//   [9-16]   Previous action (8 floats)
//   [17-50]  Boost pads (34 floats, 1.0 = available)
//   [51-69]  Self player block (19 floats)
//   [70-88]  Ally player block (19 floats)
//   [89-107] Enemy0 player block (19 floats)
//   [108-126] Enemy1 player block (19 floats)
//
// Player block (19 each):
//   pos(3), forward(3), up(3), vel(3), ang_vel(3), boost(1),
//   on_ground(1), has_flip(1), is_demoed(1)
//
// Orange team sees inverted (180-deg Z rotation) coordinates.
// Boost pad order is reversed for orange.

using namespace RLGSC;

class LuciferObs : public OBSBuilder {
public:
    static constexpr float POS_COEF = 1.0f / 2300.0f;
    static constexpr float VEL_COEF = 1.0f / 2300.0f;
    static constexpr float ANG_VEL_COEF = 1.0f / M_PI;

    FList BuildOBS(const PlayerData& player, const GameState& state,
                   const Action& prevAction) override {
        FList obs;
        obs.reserve(127);

        bool inv = (player.team == Team::ORANGE);

        // --- Ball (9 floats) ---
        const PhysObj& ball = state.GetBallPhys(inv);
        obs += ball.pos * POS_COEF;
        obs += ball.vel * VEL_COEF;
        obs += ball.angVel * ANG_VEL_COEF;

        // --- Previous action (8 floats) ---
        for (int i = 0; i < 8; i++)
            obs.push_back(prevAction[i]);

        // --- Boost pads (34 floats) ---
        const auto& pads = state.GetBoostPads(inv);
        for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
            obs.push_back(pads[i] ? 1.0f : 0.0f);

        // --- Identify ally and enemies ---
        // Players sorted by team: [B0, B1, O0, O1] in 2v2
        // For the current player, find self, ally, enemy0, enemy1
        const PlayerData* self_p = nullptr;
        const PlayerData* ally_p = nullptr;
        std::vector<const PlayerData*> enemies;

        for (const auto& p : state.players) {
            if (p.carId == player.carId) {
                self_p = &p;
            } else if (p.team == player.team) {
                ally_p = &p;
            } else {
                enemies.push_back(&p);
            }
        }

        // --- Self (19 floats) [51-69] ---
        AddPlayerBlock(obs, *self_p, inv);

        // --- Ally (19 floats) [70-88] ---
        if (ally_p) {
            AddPlayerBlock(obs, *ally_p, inv);
        } else {
            // No ally — fill zeros
            for (int i = 0; i < 19; i++) obs.push_back(0.0f);
        }

        // --- Enemy 0 (19 floats) [89-107] ---
        if (enemies.size() > 0) {
            AddPlayerBlock(obs, *enemies[0], inv);
        } else {
            for (int i = 0; i < 19; i++) obs.push_back(0.0f);
        }

        // --- Enemy 1 (19 floats) [108-126] ---
        if (enemies.size() > 1) {
            AddPlayerBlock(obs, *enemies[1], inv);
        } else {
            for (int i = 0; i < 19; i++) obs.push_back(0.0f);
        }

        return obs;
    }

private:
    void AddPlayerBlock(FList& obs, const PlayerData& p, bool inv) {
        const PhysObj& phys = p.GetPhys(inv);
        obs += phys.pos * POS_COEF;
        obs += phys.rotMat.forward;
        obs += phys.rotMat.up;
        obs += phys.vel * VEL_COEF;
        obs += phys.angVel * ANG_VEL_COEF;
        obs.push_back(p.boostFraction);
        obs.push_back(p.carState.isOnGround ? 1.0f : 0.0f);
        obs.push_back(p.hasFlip ? 1.0f : 0.0f);
        obs.push_back(p.carState.isDemoed ? 1.0f : 0.0f);
    }
};
