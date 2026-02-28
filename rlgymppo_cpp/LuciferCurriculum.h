#pragma once

#include <cstdio>
#include <algorithm>
#include <string>
#include <atomic>

// LuciferCurriculum — Entropy-driven 4-stage curriculum tracker.
//
// Stages: Foundations → Game Play → Mechanics → Mastery
//
// Advancement requires:
//   1. Minimum iterations in current stage
//   2. Entropy below floor for CONSECUTIVE iterations
//
// Clip guard: if clip fraction > threshold for N iters, reduce policy LR 20%.
//
// Thread-safe reads via atomic stage value.

struct CurriculumTracker {
    // Stage names
    static constexpr const char* STAGE_NAMES[] = {
        "Foundations", "Game Play", "Mechanics", "Mastery"
    };

    // Entropy floors for advancement: stage 0→1, 1→2, 2→3
    static constexpr float ENTROPY_ADVANCE[] = {1.5f, 1.3f, 1.0f};
    static constexpr int ENTROPY_CONSECUTIVE = 10;

    // Minimum iterations per stage before advancement is possible
    static constexpr int MIN_STAGE_ITERS[] = {500, 2000, 2500};

    // Clip guard
    static constexpr float CLIP_HIGH_THRESH = 0.25f;
    static constexpr int CLIP_HIGH_ITERS = 5;

    // State
    std::atomic<int> stage{0};
    float policyLR = 2e-4f;
    float criticLR = 2e-4f;
    float entCoef = 0.01f;
    int clipHighCount = 0;
    int stageIterCount = 0;
    int entropyLowCount = 0;
    bool stageAdvanced = false;
    bool lrReduced = false;

    CurriculumTracker(int initialStage = 0) {
        stage.store(initialStage);
    }

    // Returns true if a stage advance or LR change occurred
    bool Update(float entropy, float clipFraction) {
        stageAdvanced = false;
        lrReduced = false;
        stageIterCount++;

        int currentStage = stage.load();

        // --- Clip fraction guard ---
        if (clipFraction > CLIP_HIGH_THRESH) {
            clipHighCount++;
        } else {
            clipHighCount = 0;
        }

        if (clipHighCount >= CLIP_HIGH_ITERS) {
            float newLR = policyLR * 0.80f;
            if (newLR >= 1e-4f) {
                policyLR = newLR;
                clipHighCount = 0;
                lrReduced = true;
                printf("[CURRICULUM] Clip guard: clip > %.2f for %d iters -> policy_lr = %.2e\n",
                       CLIP_HIGH_THRESH, CLIP_HIGH_ITERS, policyLR);
                return true;
            }
        }

        // --- Stage advancement ---
        if (currentStage < 3) {
            float floor = ENTROPY_ADVANCE[currentStage];
            int minIters = MIN_STAGE_ITERS[currentStage];

            if (entropy < floor) {
                entropyLowCount++;
            } else {
                entropyLowCount = 0;
            }

            if (entropyLowCount >= ENTROPY_CONSECUTIVE && stageIterCount >= minIters) {
                int oldStage = currentStage;
                int newStage = currentStage + 1;
                stage.store(newStage);
                stageIterCount = 0;
                entropyLowCount = 0;
                stageAdvanced = true;

                printf("\n========================================\n");
                printf("  STAGE ADVANCE: %d (%s) -> %d (%s)\n",
                       oldStage, STAGE_NAMES[oldStage],
                       newStage, STAGE_NAMES[newStage]);
                printf("  Entropy: %.2f < %.2f for %d consecutive iters\n",
                       entropy, floor, ENTROPY_CONSECUTIVE);
                printf("========================================\n\n");
                return true;
            }

            // Progress logging every 100 iters
            if (stageIterCount % 100 == 0) {
                printf("[CURRICULUM] Stage %d (%s): iter %d/%d, "
                       "entropy %.2f (floor %.2f), low_count %d/%d\n",
                       currentStage, STAGE_NAMES[currentStage],
                       stageIterCount, minIters,
                       entropy, floor,
                       entropyLowCount, ENTROPY_CONSECUTIVE);
            }
        }

        return false;
    }

    int GetStage() const { return stage.load(); }
};
