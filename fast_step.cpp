/*
 * fast_step.cpp — pybind11 C++ extension for accelerated step execution.
 *
 * Two functions:
 *   fast_set_controls:    Replace Python parse/format/set_controls loop
 *   fast_get_and_extract: Replace get_gym_state + pre-stack + extract + goal + has_flip
 *
 * Build: python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <cstdint>
#include <vector>

namespace py = pybind11;

// ── fast_set_controls ──────────────────────────────────────────────────────
// Replaces: parse_actions + format_actions + _set_controls (5000 Python iters)
// Sets car controls directly on cached Car objects, bypassing rlgym_sim wrapping.
//
// Actions: (E*A, 8) int32 — raw policy output {0,1,2} for [:5], {0,1} for [5:]
// Parse:   first 5 values subtract 1 → {-1,0,1}; last 3 stay {0,1}

void fast_set_controls(
    py::list car_objects_flat,                          // length E*A
    py::array_t<int32_t, py::array::c_style> actions,  // (E*A, 8)
    int n_envs,
    int n_agents
) {
    const int total = n_envs * n_agents;
    const int32_t* act = actions.data();
    const int stride = (int)actions.shape(1);  // 8

    // Create one reusable CarControls object (set_controls copies values)
    py::object rsim = py::module_::import("RocketSim");
    py::object cc = rsim.attr("CarControls")();

    for (int i = 0; i < total; i++) {
        const int32_t* a = act + i * stride;
        // Parse: first 5 subtract 1 ({0,1,2} → {-1,0,1}), last 3 stay {0,1}
        cc.attr("throttle")  = (float)(a[0] - 1);
        cc.attr("steer")     = (float)(a[1] - 1);
        cc.attr("pitch")     = (float)(a[2] - 1);
        cc.attr("yaw")       = (float)(a[3] - 1);
        cc.attr("roll")      = (float)(a[4] - 1);
        cc.attr("jump")      = (bool)a[5];
        cc.attr("boost")     = (bool)a[6];
        cc.attr("handbrake") = (bool)a[7];

        car_objects_flat[i].attr("set_controls")(cc);
    }
}


// ── fast_get_and_extract ──────────────────────────────────────────────────
// Replaces: multi_step + get_gym_state loop + pre-stack + fast_extract +
//           goal detection + has_flip — all in one C++ call.
//
// Matches original Python behavior exactly:
//   1. First physics tick
//   2. Single get_gym_state per arena → goal check + extract (one call, reused)
//   3. Remaining physics ticks
//   4. has_flip detection (car.get_state() for airborne cars, after all ticks)
//
// Returns: (E,) bool array — which envs had goals (for score sync)

py::array_t<bool> fast_get_and_extract(
    py::list arenas,                                          // length E
    py::array_t<int32_t, py::array::c_style> car_orders,     // (E, P)
    py::list car_objects_flat,                                // length E*A
    py::object bs,                                            // BatchState
    py::array_t<int32_t, py::array::c_style> game_scores,    // (E, 2) mutable
    int tick_skip,
    float jump_timer
) {
    const int E = (int)py::len(arenas);
    const int P = (int)car_orders.shape(1);

    int32_t* gs = game_scores.mutable_data();
    const int32_t* co = car_orders.data();

    // Result: which envs had goals
    auto goal_envs = py::array_t<bool>(E);
    bool* ge = goal_envs.mutable_data();
    std::memset(ge, 0, E * sizeof(bool));

    // RocketSim references
    py::object rsim = py::module_::import("RocketSim");
    py::object Arena_cls = rsim.attr("Arena");
    py::object BallState_cls = rsim.attr("BallState");

    // ── Phase 1: First physics tick ──
    Arena_cls.attr("multi_step")(arenas, 1);

    // ── Phase 2: Single get_gym_state → goal check + extract into BatchState ──

    // Get output array references from BatchState (keep alive while using ptrs)
    // Ball
    auto a_bp  = bs.attr("ball_pos").cast<py::array_t<float>>();
    auto a_bv  = bs.attr("ball_vel").cast<py::array_t<float>>();
    auto a_ba  = bs.attr("ball_ang").cast<py::array_t<float>>();
    auto a_ibp = bs.attr("inv_ball_pos").cast<py::array_t<float>>();
    auto a_ibv = bs.attr("inv_ball_vel").cast<py::array_t<float>>();
    auto a_iba = bs.attr("inv_ball_ang").cast<py::array_t<float>>();
    // Pads
    auto a_pd  = bs.attr("boost_pads").cast<py::array_t<float>>();
    auto a_ipd = bs.attr("inv_boost_pads").cast<py::array_t<float>>();
    // Scores
    auto a_bsc = bs.attr("blue_score").cast<py::array_t<int32_t>>();
    auto a_osc = bs.attr("orange_score").cast<py::array_t<int32_t>>();
    // Player position/velocity
    auto a_pp  = bs.attr("player_pos").cast<py::array_t<float>>();
    auto a_pv  = bs.attr("player_vel").cast<py::array_t<float>>();
    auto a_pa  = bs.attr("player_ang_vel").cast<py::array_t<float>>();
    auto a_ip  = bs.attr("inv_player_pos").cast<py::array_t<float>>();
    auto a_iv  = bs.attr("inv_player_vel").cast<py::array_t<float>>();
    auto a_ia  = bs.attr("inv_player_ang_vel").cast<py::array_t<float>>();
    // Rotation vectors
    auto a_fw  = bs.attr("player_fwd").cast<py::array_t<float>>();
    auto a_up  = bs.attr("player_up").cast<py::array_t<float>>();
    auto a_ifw = bs.attr("inv_player_fwd").cast<py::array_t<float>>();
    auto a_iup = bs.attr("inv_player_up").cast<py::array_t<float>>();
    // Attributes
    auto a_bo  = bs.attr("player_boost").cast<py::array_t<float>>();
    auto a_gr  = bs.attr("player_on_ground").cast<py::array_t<float>>();
    auto a_dm  = bs.attr("player_is_demoed").cast<py::array_t<float>>();
    auto a_bt  = bs.attr("player_ball_touched").cast<py::array_t<float>>();
    auto a_tm  = bs.attr("player_team").cast<py::array_t<int32_t>>();
    auto a_ci  = bs.attr("player_car_id").cast<py::array_t<int32_t>>();
    // Events
    auto a_gl  = bs.attr("player_goals").cast<py::array_t<float>>();
    auto a_sv  = bs.attr("player_saves").cast<py::array_t<float>>();
    auto a_sh  = bs.attr("player_shots").cast<py::array_t<float>>();
    auto a_de  = bs.attr("player_demos").cast<py::array_t<float>>();
    // Has flip (Phase 4)
    auto a_hf  = bs.attr("player_has_flip").cast<py::array_t<float>>();

    // Raw output pointers
    float*   o_bp  = a_bp.mutable_data();
    float*   o_bv  = a_bv.mutable_data();
    float*   o_ba  = a_ba.mutable_data();
    float*   o_ibp = a_ibp.mutable_data();
    float*   o_ibv = a_ibv.mutable_data();
    float*   o_iba = a_iba.mutable_data();
    float*   o_pd  = a_pd.mutable_data();
    float*   o_ipd = a_ipd.mutable_data();
    int32_t* o_bs  = a_bsc.mutable_data();
    int32_t* o_os  = a_osc.mutable_data();
    float*   o_pp  = a_pp.mutable_data();
    float*   o_pv  = a_pv.mutable_data();
    float*   o_pa  = a_pa.mutable_data();
    float*   o_ip  = a_ip.mutable_data();
    float*   o_iv  = a_iv.mutable_data();
    float*   o_ia  = a_ia.mutable_data();
    float*   o_fw  = a_fw.mutable_data();
    float*   o_up  = a_up.mutable_data();
    float*   o_ifw = a_ifw.mutable_data();
    float*   o_iup = a_iup.mutable_data();
    float*   o_bo  = a_bo.mutable_data();
    float*   o_gr  = a_gr.mutable_data();
    float*   o_dm  = a_dm.mutable_data();
    float*   o_bt  = a_bt.mutable_data();
    int32_t* o_tm  = a_tm.mutable_data();
    int32_t* o_ci  = a_ci.mutable_data();
    float*   o_gl  = a_gl.mutable_data();
    float*   o_sv  = a_sv.mutable_data();
    float*   o_sh  = a_sh.mutable_data();
    float*   o_de  = a_de.mutable_data();
    float*   o_hf  = a_hf.mutable_data();

    // Pre-cache Python int keys for sequence indexing
    py::int_ _0(0), _1(1), _2(2);

    // Single pass: get_gym_state → goal check → extract (one call per arena)
    for (int i = 0; i < E; i++) {
        py::object raw = arenas[i].attr("get_gym_state")();

        // ── Goal check (from game data) ──
        auto gd = raw[_0].cast<py::array_t<float>>();
        const float* gd_ptr = gd.data();
        int32_t bs_score = (int32_t)gd_ptr[2];
        int32_t os_score = (int32_t)gd_ptr[3];

        if (bs_score != gs[i * 2] || os_score != gs[i * 2 + 1]) {
            gs[i * 2] = bs_score;
            gs[i * 2 + 1] = os_score;
            arenas[i].attr("ball").attr("set_state")(BallState_cls());
            ge[i] = true;
        }

        // ── Extract into BatchState (same raw data, no second call) ──
        o_bs[i] = bs_score;
        o_os[i] = os_score;

        // Boost pads
        py::object pad_data = raw[_1];
        auto pad_n = pad_data[_0].cast<py::array_t<float>>();
        auto pad_i = pad_data[_1].cast<py::array_t<float>>();
        std::memcpy(o_pd  + i * 34, pad_n.data(), 34 * sizeof(float));
        std::memcpy(o_ipd + i * 34, pad_i.data(), 34 * sizeof(float));

        // Ball
        py::object ball_data = raw[_2];
        auto ball_n = ball_data[_0].cast<py::array_t<float>>();
        auto ball_i = ball_data[_1].cast<py::array_t<float>>();
        const float* bn = ball_n.data();
        const float* bi = ball_i.data();
        std::memcpy(o_bp  + i * 3, bn,      3 * sizeof(float));
        std::memcpy(o_bv  + i * 3, bn + 7,  3 * sizeof(float));
        std::memcpy(o_ba  + i * 3, bn + 10, 3 * sizeof(float));
        std::memcpy(o_ibp + i * 3, bi,      3 * sizeof(float));
        std::memcpy(o_ibv + i * 3, bi + 7,  3 * sizeof(float));
        std::memcpy(o_iba + i * 3, bi + 10, 3 * sizeof(float));

        // Players (reordered via car_orders)
        for (int k = 0; k < P; k++) {
            int j  = co[i * P + k];           // reordered player index
            int ij = i * P + j;               // flat output index [i, j]

            py::object car_data = raw[py::int_(3 + k)];
            auto car_n = car_data[_0].cast<py::array_t<float>>();
            auto car_i = car_data[_1].cast<py::array_t<float>>();
            const float* d  = car_n.data();
            const float* di = car_i.data();

            // Scalar fields (normal)
            o_tm[ij] = static_cast<int32_t>(d[1]);
            o_gl[ij] = d[2];
            o_sv[ij] = d[3];
            o_sh[ij] = d[4];
            o_de[ij] = d[5];
            o_dm[ij] = d[7];
            o_gr[ij] = d[8];
            o_bt[ij] = d[9];
            o_bo[ij] = d[10] * 0.01f;         // boost /100
            o_ci[ij] = static_cast<int32_t>(d[0]);

            // 3-vectors (normal): pos=[11:14], vel=[18:21], ang=[21:24],
            //                     fwd=[24:27], up=[30:33]
            std::memcpy(o_pp  + ij * 3, d + 11, 3 * sizeof(float));
            std::memcpy(o_pv  + ij * 3, d + 18, 3 * sizeof(float));
            std::memcpy(o_pa  + ij * 3, d + 21, 3 * sizeof(float));
            std::memcpy(o_fw  + ij * 3, d + 24, 3 * sizeof(float));
            std::memcpy(o_up  + ij * 3, d + 30, 3 * sizeof(float));

            // 3-vectors (inverted)
            std::memcpy(o_ip  + ij * 3, di + 11, 3 * sizeof(float));
            std::memcpy(o_iv  + ij * 3, di + 18, 3 * sizeof(float));
            std::memcpy(o_ia  + ij * 3, di + 21, 3 * sizeof(float));
            std::memcpy(o_ifw + ij * 3, di + 24, 3 * sizeof(float));
            std::memcpy(o_iup + ij * 3, di + 30, 3 * sizeof(float));
        }
    }

    // ── Phase 3: Remaining physics ticks ──
    if (tick_skip > 1) {
        Arena_cls.attr("multi_step")(arenas, tick_skip - 1);
    }

    // ── Phase 4: has_flip detection (after all ticks) ──
    // Grounded → always has_flip; demoed → never; airborne → check car state
    for (int i = 0; i < E; i++) {
        for (int j = 0; j < P; j++) {
            int ij = i * P + j;
            if (o_gr[ij] == 1.0f) {
                o_hf[ij] = 1.0f;
            } else if (o_dm[ij] == 1.0f) {
                o_hf[ij] = 0.0f;
            } else {
                // Airborne + not demoed → need car.get_state()
                py::object car = car_objects_flat[ij];
                py::object cs = car.attr("get_state")();
                float air_time = cs.attr("air_time_since_jump").cast<float>();
                bool flipped = cs.attr("has_flipped").cast<bool>();
                bool double_jumped = cs.attr("has_double_jumped").cast<bool>();
                o_hf[ij] = (air_time < jump_timer && !flipped && !double_jumped)
                            ? 1.0f : 0.0f;
            }
        }
    }

    return goal_envs;
}


PYBIND11_MODULE(fast_step, m) {
    m.doc() = "C++ accelerated step execution for Lucifer RL bot";
    m.def("fast_set_controls", &fast_set_controls,
          "Set car controls directly, bypassing rlgym_sim wrapping");
    m.def("fast_get_and_extract", &fast_get_and_extract,
          "Combined physics step + state extraction + goal detection + has_flip");
}
