// C ABI consumed by rust/ (src/ffi.rs) under `--features cuda`.
// Matches the extern "C" signature declared there exactly.
#include "mps.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>

using namespace tessera;

extern "C" double tessera_qa_anneal(
    const double* h, std::size_t n,
    const std::size_t* ei, const std::size_t* ej, const double* ew, std::size_t m,
    const double* schedule_a, const double* schedule_b, std::size_t steps,
    std::size_t chi,
    std::int8_t* out_s,
    double* out_entropy) {

    std::vector<double> hv(h, h + n);
    std::vector<int> eiv(m), ejv(m);
    std::vector<double> ewv(ew, ew + m);
    for (std::size_t k = 0; k < m; ++k) { eiv[k] = (int)ei[k]; ejv[k] = (int)ej[k]; }
    std::vector<double> sa(schedule_a, schedule_a + steps);
    std::vector<double> sb(schedule_b, schedule_b + steps);

    // Trotter step (total adiabatic time T = steps * dt). Each mps_anneal step
    // internally sub-Trotterises to keep the per-gate error small even for dense
    // long-range graphs.
    const double dt = 0.1;
    AnnealResult r = mps_anneal(hv, eiv, ejv, ewv, sa, sb, dt, (int)chi);

    for (std::size_t i = 0; i < n; ++i) out_s[i] = r.spins[i];
    if (out_entropy) {
        for (std::size_t k = 0; k < steps; ++k)
            out_entropy[k] = (k < r.entropy.size()) ? r.entropy[k] : 0.0;
    }
    return r.energy;
}
