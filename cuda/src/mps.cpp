#include "mps.hpp"
#include <cmath>
#include <algorithm>

namespace tessera {

// ── construction ─────────────────────────────────────────────────────────────

MPS::MPS(int n) : n_(n), s_(n) {
    // |+⟩^n : every tensor (1,2,1) with amplitude 1/sqrt(2).
    const double a = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < n; ++i) {
        s_[i] = Site(1, 1);
        s_[i](0, 0, 0) = a;
        s_[i](0, 1, 0) = a;
    }
}

int MPS::max_bond() const {
    int m = 1;
    for (const auto& st : s_) m = std::max(m, st.Dr);
    return m;
}

// ── single-site gate ─────────────────────────────────────────────────────────

void MPS::apply_single(int i, const cd G[2][2]) {
    Site& A = s_[i];
    Site B(A.Dl, A.Dr);
    for (int a = 0; a < A.Dl; ++a)
        for (int b = 0; b < A.Dr; ++b) {
            cd v0 = A(a, 0, b), v1 = A(a, 1, b);
            B(a, 0, b) = G[0][0] * v0 + G[0][1] * v1;
            B(a, 1, b) = G[1][0] * v0 + G[1][1] * v1;
        }
    A = std::move(B);
}

// ── two-site adjacent gate with SVD truncation ───────────────────────────────

void MPS::apply_adjacent(int i, const cd G[4][4], int chi, double& trunc) {
    Site& A = s_[i];
    Site& C = s_[i + 1];
    const int Dl = A.Dl, Dr = C.Dr, mid = A.Dr;  // A.Dr == C.Dl

    // Theta(a, s1, s2, b) = Σ_c A(a,s1,c) C(c,s2,b)
    // then apply gate over (s1,s2).
    // Matricise as M[(a*2+s1), (s2*Dr + b)] : rows R = Dl*2, cols Cc = 2*Dr.
    const int R = Dl * 2, Cc = 2 * Dr;
    CMat M((std::size_t)R, (std::size_t)Cc);
    for (int a = 0; a < Dl; ++a)
        for (int b = 0; b < Dr; ++b) {
            // contract bond c for the four (s1,s2) combos
            cd th[2][2];
            for (int s1 = 0; s1 < 2; ++s1)
                for (int s2 = 0; s2 < 2; ++s2) {
                    cd acc(0, 0);
                    for (int c = 0; c < mid; ++c) acc += A(a, s1, c) * C(c, s2, b);
                    th[s1][s2] = acc;
                }
            // apply gate: th'[r1][r2] = Σ G[r1*2+r2][s1*2+s2] th[s1][s2]
            for (int r1 = 0; r1 < 2; ++r1)
                for (int r2 = 0; r2 < 2; ++r2) {
                    cd acc(0, 0);
                    for (int s1 = 0; s1 < 2; ++s1)
                        for (int s2 = 0; s2 < 2; ++s2)
                            acc += G[r1 * 2 + r2][s1 * 2 + s2] * th[s1][s2];
                    M((std::size_t)(a * 2 + r1), (std::size_t)(r2 * Dr + b)) = acc;
                }
        }

    CMat U, Vh;
    std::vector<double> S;
    svd(M, U, S, Vh);

    // Truncate: keep up to chi singular values; record discarded weight.
    int kfull = (int)S.size();
    double total = 0.0;
    for (double sv : S) total += sv * sv;
    int keep = std::min(chi, kfull);
    // drop trailing (near-)zeros too
    while (keep > 1 && S[keep - 1] < 1e-14) --keep;
    double disc = 0.0;
    for (int t = keep; t < kfull; ++t) disc += S[t] * S[t];
    if (total > 0) trunc = std::max(trunc, disc / total);

    // A <- U(:,0:keep) reshaped (Dl,2,keep);  C <- diag(S) Vh(0:keep,:) (keep,2,Dr).
    Site An(Dl, keep);
    for (int a = 0; a < Dl; ++a)
        for (int s1 = 0; s1 < 2; ++s1)
            for (int k = 0; k < keep; ++k)
                An(a, s1, k) = U((std::size_t)(a * 2 + s1), (std::size_t)k);
    Site Cn(keep, Dr);
    for (int k = 0; k < keep; ++k)
        for (int s2 = 0; s2 < 2; ++s2)
            for (int b = 0; b < Dr; ++b)
                Cn(k, s2, b) = S[k] * Vh((std::size_t)k, (std::size_t)(s2 * Dr + b));
    A = std::move(An);
    C = std::move(Cn);
}

// ── long-range ZZ via swap network ───────────────────────────────────────────

// SWAP gate (4×4): |s1 s2⟩ -> |s2 s1⟩.
static void swap_gate(cd G[4][4]) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) G[r][c] = cd(0, 0);
    G[0][0] = 1;  // 00->00
    G[1][2] = 1;  // 01<-10 : output 01 from input 10
    G[2][1] = 1;  // 10<-01
    G[3][3] = 1;  // 11->11
}

// exp(-i θ Z⊗Z): diagonal in computational basis. ZZ eigenvalues:
// |00>:+1, |01>:-1, |10>:-1, |11>:+1.
static void zz_gate(double theta, cd G[4][4]) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) G[r][c] = cd(0, 0);
    const cd I(0, 1);
    G[0][0] = std::exp(-I * theta * 1.0);
    G[1][1] = std::exp(-I * theta * (-1.0));
    G[2][2] = std::exp(-I * theta * (-1.0));
    G[3][3] = std::exp(-I * theta * 1.0);
}

void MPS::apply_zz(int i, int j, double theta, int chi, double& trunc) {
    if (i > j) std::swap(i, j);
    if (i == j) return;
    if (j == i + 1) {
        cd G[4][4];
        zz_gate(theta, G);
        apply_adjacent(i, G, chi, trunc);
        return;
    }
    // Bring qubit j adjacent to i via swaps: swap (j-1,j),(j-2,j-1),... down to i+1.
    cd SW[4][4];
    swap_gate(SW);
    for (int k = j; k > i + 1; --k) apply_adjacent(k - 1, SW, chi, trunc);
    // Now the original qubit j sits at position i+1.
    cd G[4][4];
    zz_gate(theta, G);
    apply_adjacent(i, G, chi, trunc);
    // Swap back to restore site order.
    for (int k = i + 1; k < j; ++k) apply_adjacent(k, SW, chi, trunc);
}

// ── contractions / observables ───────────────────────────────────────────────

// Σ_{k∈zsites} insert Z on the ket physical index. Returns ⟨ψ| (Π Z) |ψ⟩ raw.
cd MPS::expect_zstring_raw(const std::vector<int>& zsites) const {
    // Left environment E[a,a'] (ket bond a, bra bond a'). Start 1×1 = [1].
    std::vector<cd> E = {cd(1, 0)};
    int El = 1, Er = 1;  // current env dims (ket, bra)
    for (int site = 0; site < n_; ++site) {
        const Site& A = s_[site];
        bool isz = std::find(zsites.begin(), zsites.end(), site) != zsites.end();
        int nb = A.Dr;  // new ket bond
        std::vector<cd> Enew((std::size_t)nb * nb, cd(0, 0));
        // Enew[b,b'] = Σ_{a,a',s} E[a,a'] conj(A(a',s,b')) * zf(s) * A(a,s,b)
        for (int a = 0; a < El; ++a)
            for (int ap = 0; ap < Er; ++ap) {
                cd e = E[(std::size_t)a * Er + ap];
                if (e == cd(0, 0)) continue;
                for (int s = 0; s < 2; ++s) {
                    double zf = (isz && s == 1) ? -1.0 : 1.0;
                    for (int b = 0; b < nb; ++b) {
                        cd ket = zf * A(a, s, b) * e;
                        if (ket == cd(0, 0)) continue;
                        for (int bp = 0; bp < nb; ++bp)
                            Enew[(std::size_t)b * nb + bp] += ket * std::conj(A(ap, s, bp));
                    }
                }
            }
        E.swap(Enew);
        El = nb; Er = nb;
    }
    return E[0];
}

double MPS::norm2() const { return expect_zstring_raw({}).real(); }

double MPS::expect_z(int i) const {
    double nrm = norm2();
    if (nrm <= 0) return 0.0;
    return expect_zstring_raw({i}).real() / nrm;
}

double MPS::expect_zz(int i, int j) const {
    double nrm = norm2();
    if (nrm <= 0) return 0.0;
    std::vector<int> z = {i, j};
    std::sort(z.begin(), z.end());
    return expect_zstring_raw(z).real() / nrm;
}

// |amplitude(x)|² / norm — probability of a specific ±1 configuration.
double MPS::amplitude_prob(const std::vector<int8_t>& spins) const {
    // Contract MPS with fixed physical indices: product of (Dl×Dr) matrices.
    std::vector<cd> v = {cd(1, 0)};  // 1×1
    int cols = 1;
    for (int site = 0; site < n_; ++site) {
        const Site& A = s_[site];
        int s = (spins[site] > 0) ? 0 : 1;   // +1 -> |0>, -1 -> |1>
        std::vector<cd> nv((std::size_t)A.Dr, cd(0, 0));
        for (int b = 0; b < A.Dr; ++b) {
            cd acc(0, 0);
            for (int a = 0; a < A.Dl && a < cols; ++a) acc += v[a] * A(a, s, b);
            nv[b] = acc;
        }
        v.swap(nv);
        cols = A.Dr;
    }
    cd amp = v[0];
    double nrm = norm2();
    return nrm > 0 ? std::norm(amp) / nrm : 0.0;
}

double MPS::single_site_entropy(int i) const {
    // 1-site reduced density matrix ρ (2×2): ρ[s,s'] via left/right environments.
    // Left env up to i, right env after i, contract physical at i open.
    // Build left environment L[a,a'] over sites < i.
    std::vector<cd> L = {cd(1, 0)};
    int Ll = 1, Lr = 1;
    for (int site = 0; site < i; ++site) {
        const Site& A = s_[site];
        int nb = A.Dr;
        std::vector<cd> Ln((std::size_t)nb * nb, cd(0, 0));
        for (int a = 0; a < Ll; ++a)
            for (int ap = 0; ap < Lr; ++ap) {
                cd e = L[(std::size_t)a * Lr + ap];
                if (e == cd(0, 0)) continue;
                for (int s = 0; s < 2; ++s)
                    for (int b = 0; b < nb; ++b) {
                        cd ket = A(a, s, b) * e;
                        if (ket == cd(0, 0)) continue;
                        for (int bp = 0; bp < nb; ++bp)
                            Ln[(std::size_t)b * nb + bp] += ket * std::conj(A(ap, s, bp));
                    }
            }
        L.swap(Ln); Ll = nb; Lr = nb;
    }
    // Right environment R[b,b'] over sites > i.
    std::vector<cd> R = {cd(1, 0)};
    int Rl = 1, Rr = 1;
    for (int site = n_ - 1; site > i; --site) {
        const Site& A = s_[site];
        int nb = A.Dl;
        std::vector<cd> Rn((std::size_t)nb * nb, cd(0, 0));
        for (int b = 0; b < Rl; ++b)
            for (int bp = 0; bp < Rr; ++bp) {
                cd e = R[(std::size_t)b * Rr + bp];
                if (e == cd(0, 0)) continue;
                for (int s = 0; s < 2; ++s)
                    for (int a = 0; a < nb; ++a) {
                        cd ket = A(a, s, b) * e;
                        if (ket == cd(0, 0)) continue;
                        for (int ap = 0; ap < nb; ++ap)
                            Rn[(std::size_t)a * nb + ap] += ket * std::conj(A(ap, s, bp));
                    }
            }
        R.swap(Rn); Rl = nb; Rr = nb;
    }
    const Site& A = s_[i];
    // ρ[s,s'] = Σ L[a,a'] A(a,s,b) conj(A(a',s',b')) R[b,b']
    cd rho[2][2] = {{cd(0,0),cd(0,0)},{cd(0,0),cd(0,0)}};
    for (int s = 0; s < 2; ++s)
        for (int sp = 0; sp < 2; ++sp) {
            cd acc(0, 0);
            for (int a = 0; a < A.Dl; ++a)
                for (int ap = 0; ap < A.Dl; ++ap) {
                    cd l = L[(std::size_t)a * A.Dl + ap];
                    if (l == cd(0, 0)) continue;
                    for (int b = 0; b < A.Dr; ++b)
                        for (int bp = 0; bp < A.Dr; ++bp)
                            acc += l * A(a, s, b) * std::conj(A(ap, sp, bp)) *
                                   R[(std::size_t)b * A.Dr + bp];
                }
            rho[s][sp] = acc;
        }
    double tr = (rho[0][0] + rho[1][1]).real();
    if (tr <= 0) return 0.0;
    // Eigenvalues of the 2×2 Hermitian ρ/tr.
    double a = rho[0][0].real() / tr, d = rho[1][1].real() / tr;
    double off2 = std::norm(rho[0][1] / tr);
    double mid = (a + d) / 2.0, rad = std::sqrt(std::max(0.0, ((a - d) / 2) * ((a - d) / 2) + off2));
    double l1 = std::min(1.0, std::max(0.0, mid + rad));
    double l2 = std::min(1.0, std::max(0.0, mid - rad));
    double s = 0.0;
    for (double lam : {l1, l2}) if (lam > 1e-12) s -= lam * std::log2(lam);
    return s;
}

// ── anneal driver ────────────────────────────────────────────────────────────

AnnealResult mps_anneal(const std::vector<double>& h,
                        const std::vector<int>& ei,
                        const std::vector<int>& ej,
                        const std::vector<double>& ew,
                        const std::vector<double>& sa,
                        const std::vector<double>& sb,
                        double dt, int chi, int zz_substeps) {
    const int n = (int)h.size();
    const int steps = (int)sa.size();
    MPS psi(n);
    AnnealResult res;
    res.entropy.assign(steps, 0.0);
    res.max_trunc = 0.0;

    for (int k = 0; k < steps; ++k) {
        const double A = sa[k], B = sb[k];

        // Half single-site step: g_i = -A·X - B·h_i·Z  ;  gate = exp(-i dt/2 g_i)
        for (int i = 0; i < n; ++i) {
            CMat g(2, 2);
            g(0, 0) = cd(-B * h[i], 0);
            g(1, 1) = cd(B * h[i], 0);
            g(0, 1) = cd(-A, 0);
            g(1, 0) = cd(-A, 0);
            CMat U = hermitian_expm(g, cd(0, -1) * (dt / 2.0));
            cd G[2][2] = {{U(0,0), U(0,1)}, {U(1,0), U(1,1)}};
            psi.apply_single(i, G);
        }

        // Full two-site step. The ZZ terms all commute (diagonal in Z), but each
        // long-range apply_zz routes through a swap network, and those swaps do
        // NOT commute with the not-yet-applied ZZ gates -> first-order error that
        // grows with the number of long-range couplings. Subdividing the whole
        // ZZ block into `zz_substeps` finer slices drives that error to zero.
        // Symmetrise (forward then reversed order) so the leading error cancels.
        {
            const int ns = zz_substeps < 1 ? 1 : zz_substeps;
            const double sub = dt / ns;
            for (int s = 0; s < ns; ++s) {
                if (s % 2 == 0)
                    for (std::size_t e = 0; e < ew.size(); ++e)
                        psi.apply_zz(ei[e], ej[e], -B * ew[e] * sub, chi, res.max_trunc);
                else
                    for (std::size_t e = ew.size(); e-- > 0;)
                        psi.apply_zz(ei[e], ej[e], -B * ew[e] * sub, chi, res.max_trunc);
            }
        }

        // Half single-site step again (second-order Trotter).
        for (int i = 0; i < n; ++i) {
            CMat g(2, 2);
            g(0, 0) = cd(-B * h[i], 0);
            g(1, 1) = cd(B * h[i], 0);
            g(0, 1) = cd(-A, 0);
            g(1, 0) = cd(-A, 0);
            CMat U = hermitian_expm(g, cd(0, -1) * (dt / 2.0));
            cd G[2][2] = {{U(0,0), U(0,1)}, {U(1,0), U(1,1)}};
            psi.apply_single(i, G);
        }

        res.entropy[k] = psi.single_site_entropy(0);
    }

    // DIAGNOSTIC: expectation energy <psi|H|psi> measured directly from the MPS
    // (no rounding). If the dynamics are correct this tracks the true ground
    // energy; comparing it to the rounded-readout energy isolates readout bugs.
    {
        double Eexp = 0.0;
        for (std::size_t e = 0; e < ew.size(); ++e)
            Eexp -= ew[e] * psi.expect_zz(ei[e], ej[e]);
        for (int i = 0; i < n; ++i) Eexp -= h[i] * psi.expect_z(i);
        res.expected_energy = Eexp;
    }

    // Readout: spins[i] = sign(⟨Z_i⟩).
    res.spins.assign(n, 1);
    for (int i = 0; i < n; ++i) res.spins[i] = (psi.expect_z(i) >= 0.0) ? 1 : -1;

    // Ising energy of the readout: E = -Σ J s s - Σ h s.
    double E = 0.0;
    for (std::size_t e = 0; e < ew.size(); ++e)
        E -= ew[e] * res.spins[ei[e]] * res.spins[ej[e]];
    for (int i = 0; i < n; ++i) E -= h[i] * res.spins[i];
    res.energy = E;
    res.prob = psi.amplitude_prob(res.spins);
    return res;
}

}  // namespace tessera
