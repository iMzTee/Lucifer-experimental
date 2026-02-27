/*
 * fast_extract.cpp — pybind11 C++ extension for accelerated state extraction.
 *
 * Replaces the Python loop in BatchState.extract_raw() with C++ memcpy.
 * Python pre-stacks raw data into contiguous numpy arrays, then this function
 * does indexed memcpy with raw pointers — zero Python loop overhead.
 *
 * Build: python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <cstdint>

namespace py = pybind11;

void extract_raw_fast(
    py::array_t<float, py::array::c_style>   pad_norm,    // (E, 34)
    py::array_t<float, py::array::c_style>   pad_inv,     // (E, 34)
    py::array_t<float, py::array::c_style>   ball_norm,   // (E, B)
    py::array_t<float, py::array::c_style>   ball_inv,    // (E, B)
    py::array_t<float, py::array::c_style>   car_norm,    // (E, P, C)
    py::array_t<float, py::array::c_style>   car_inv,     // (E, P, C)
    py::array_t<int32_t, py::array::c_style> car_orders,  // (E, P)
    py::array_t<int32_t, py::array::c_style> scores,      // (E, 2) [blue, orange]
    py::object bs                                          // BatchState instance
) {
    const int E = car_orders.shape(0);        // num environments
    const int P = car_orders.shape(1);        // players per env
    const int B = (int)ball_norm.shape(1);    // ball data stride
    const int C = (int)car_norm.shape(2);     // car data stride

    // ── Input pointers ──
    const float*   pn = pad_norm.data();
    const float*   pi = pad_inv.data();
    const float*   bn = ball_norm.data();
    const float*   bi = ball_inv.data();
    const float*   cn = car_norm.data();
    const float*   ci = car_inv.data();
    const int32_t* co = car_orders.data();
    const int32_t* sc = scores.data();

    // ── Output arrays from BatchState ──
    // Keep py::array_t references alive while using raw pointers.
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

    // ── Raw output pointers ──
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

    // ── Hot loop — release GIL for pure C++ memcpy ──
    {
        py::gil_scoped_release release;

        for (int i = 0; i < E; i++) {
            // Scores
            o_bs[i] = sc[i * 2];
            o_os[i] = sc[i * 2 + 1];

            // Ball (normal): pos=[:3], vel=[7:10], ang=[10:13]
            const float* b = bn + i * B;
            std::memcpy(o_bp  + i * 3, b,      3 * sizeof(float));
            std::memcpy(o_bv  + i * 3, b + 7,  3 * sizeof(float));
            std::memcpy(o_ba  + i * 3, b + 10, 3 * sizeof(float));

            // Ball (inverted)
            const float* ib = bi + i * B;
            std::memcpy(o_ibp + i * 3, ib,      3 * sizeof(float));
            std::memcpy(o_ibv + i * 3, ib + 7,  3 * sizeof(float));
            std::memcpy(o_iba + i * 3, ib + 10, 3 * sizeof(float));

            // Boost pads (34 floats each)
            std::memcpy(o_pd  + i * 34, pn + i * 34, 34 * sizeof(float));
            std::memcpy(o_ipd + i * 34, pi + i * 34, 34 * sizeof(float));

            // Players (reordered via car_orders)
            for (int k = 0; k < P; k++) {
                int j  = co[i * P + k];           // reordered player index
                int ij = i * P + j;                // flat output index [i, j]
                const float* d  = cn + (i * P + k) * C;  // normal car data
                const float* di = ci + (i * P + k) * C;  // inverted car data

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
                std::memcpy(o_pp + ij * 3, d + 11, 3 * sizeof(float));
                std::memcpy(o_pv + ij * 3, d + 18, 3 * sizeof(float));
                std::memcpy(o_pa + ij * 3, d + 21, 3 * sizeof(float));
                std::memcpy(o_fw + ij * 3, d + 24, 3 * sizeof(float));
                std::memcpy(o_up + ij * 3, d + 30, 3 * sizeof(float));

                // 3-vectors (inverted)
                std::memcpy(o_ip  + ij * 3, di + 11, 3 * sizeof(float));
                std::memcpy(o_iv  + ij * 3, di + 18, 3 * sizeof(float));
                std::memcpy(o_ia  + ij * 3, di + 21, 3 * sizeof(float));
                std::memcpy(o_ifw + ij * 3, di + 24, 3 * sizeof(float));
                std::memcpy(o_iup + ij * 3, di + 30, 3 * sizeof(float));
            }
        }
    }
    // GIL re-acquired automatically
}

PYBIND11_MODULE(fast_extract, m) {
    m.doc() = "C++ accelerated state extraction for Lucifer RL bot";
    m.def("extract_raw_fast", &extract_raw_fast,
          "Fast extraction of raw arena state into BatchState arrays");
}
