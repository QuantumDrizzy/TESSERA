#include "linalg.hpp"
#include <cmath>
#include <algorithm>

namespace tessera {

void jacobi_symmetric(const std::vector<double>& A_in, std::size_t n,
                      std::vector<double>& w, std::vector<double>& V) {
    std::vector<double> A = A_in;  // working copy, mutated in place
    V.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) V[i * n + i] = 1.0;
    if (n == 1) { w = {A[0]}; return; }

    for (int sweep = 0; sweep < 100; ++sweep) {
        // Sum of off-diagonal magnitudes.
        double off = 0.0;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q)
                off += A[p * n + q] * A[p * n + q];
        if (off < 1e-30) break;

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                double apq = A[p * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                double app = A[p * n + p], aqq = A[q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) /
                           (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;
                // Rotate rows/cols p,q of A.
                for (std::size_t k = 0; k < n; ++k) {
                    double akp = A[k * n + p], akq = A[k * n + q];
                    A[k * n + p] = c * akp - s * akq;
                    A[k * n + q] = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    double apk = A[p * n + k], aqk = A[q * n + k];
                    A[p * n + k] = c * apk - s * aqk;
                    A[q * n + k] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors.
                for (std::size_t k = 0; k < n; ++k) {
                    double vkp = V[k * n + p], vkq = V[k * n + q];
                    V[k * n + p] = c * vkp - s * vkq;
                    V[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    w.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) w[i] = A[i * n + i];
}

void hermitian_eig(const CMat& H, std::vector<double>& evals, CMat& evecs) {
    const std::size_t n = H.rows;
    // Real embedding M = [[R, -I], [I, R]]  (2n × 2n real symmetric).
    const std::size_t m = 2 * n;
    std::vector<double> M(m * m, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double R = H(i, j).real();
            double I = H(i, j).imag();
            M[i * m + j] = R;                  // top-left R
            M[(i + n) * m + (j + n)] = R;      // bottom-right R
            M[i * m + (j + n)] = -I;           // top-right -I
            M[(i + n) * m + j] = I;            // bottom-left I
        }
    }
    std::vector<double> w2, V2;
    jacobi_symmetric(M, m, w2, V2);

    // Each complex eigenpair appears twice. Sort embedding eigenpairs by value
    // descending, then greedily accept complex eigenvectors that are orthogonal
    // to those already taken (handles the 2× degeneracy and any true degeneracy).
    std::vector<std::size_t> idx(m);
    for (std::size_t i = 0; i < m; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return w2[a] > w2[b]; });

    evals.assign(n, 0.0);
    evecs = CMat(n, n);
    std::size_t taken = 0;
    for (std::size_t t = 0; t < m && taken < n; ++t) {
        std::size_t col = idx[t];
        // Complex eigenvector v = top + i·bottom.
        std::vector<cd> v(n);
        for (std::size_t i = 0; i < n; ++i)
            v[i] = cd(V2[i * m + col], V2[(i + n) * m + col]);
        // Orthogonalise against already-accepted eigenvectors (complex GS).
        for (std::size_t p = 0; p < taken; ++p) {
            cd dot(0, 0);
            for (std::size_t i = 0; i < n; ++i) dot += std::conj(evecs(i, p)) * v[i];
            for (std::size_t i = 0; i < n; ++i) v[i] -= dot * evecs(i, p);
        }
        double nrm = 0.0;
        for (std::size_t i = 0; i < n; ++i) nrm += std::norm(v[i]);
        nrm = std::sqrt(nrm);
        if (nrm < 1e-7) continue;  // dependent (the duplicate copy) — skip
        for (std::size_t i = 0; i < n; ++i) evecs(i, taken) = v[i] / nrm;
        evals[taken] = w2[col];
        ++taken;
    }
}

void svd(const CMat& A, CMat& U, std::vector<double>& S, CMat& Vh) {
    const std::size_t r = A.rows, c = A.cols;
    const std::size_t k = std::min(r, c);

    // Work with the smaller Gram matrix for efficiency & numerical stability.
    if (c <= r) {
        // G = A^† A  (c×c Hermitian).  Eigenpairs give V and S^2.
        CMat G(c, c);
        for (std::size_t i = 0; i < c; ++i)
            for (std::size_t j = 0; j < c; ++j) {
                cd acc(0, 0);
                for (std::size_t m = 0; m < r; ++m)
                    acc += std::conj(A(m, i)) * A(m, j);
                G(i, j) = acc;
            }
        std::vector<double> w;  // ascending-ish from Jacobi; we sort below
        CMat V;
        hermitian_eig(G, w, V);  // columns of V already sorted desc by eval
        S.assign(k, 0.0);
        U = CMat(r, k);
        Vh = CMat(k, c);
        for (std::size_t t = 0; t < k; ++t) {
            double sv = std::sqrt(std::max(w[t], 0.0));
            S[t] = sv;
            // V column t -> Vh row t (conjugate transpose).
            for (std::size_t j = 0; j < c; ++j) Vh(t, j) = std::conj(V(j, t));
            // u_t = A v_t / sv.
            if (sv > 1e-300) {
                for (std::size_t m = 0; m < r; ++m) {
                    cd acc(0, 0);
                    for (std::size_t j = 0; j < c; ++j) acc += A(m, j) * V(j, t);
                    U(m, t) = acc / sv;
                }
            }
        }
    } else {
        // r < c: work with G = A A^† (r×r). Eigenpairs give U and S^2.
        CMat G(r, r);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < r; ++j) {
                cd acc(0, 0);
                for (std::size_t m = 0; m < c; ++m)
                    acc += A(i, m) * std::conj(A(j, m));
                G(i, j) = acc;
            }
        std::vector<double> w;
        CMat Umat;
        hermitian_eig(G, w, Umat);
        S.assign(k, 0.0);
        U = CMat(r, k);
        Vh = CMat(k, c);
        for (std::size_t t = 0; t < k; ++t) {
            double sv = std::sqrt(std::max(w[t], 0.0));
            S[t] = sv;
            for (std::size_t m = 0; m < r; ++m) U(m, t) = Umat(m, t);
            // v_t^† = u_t^† A / sv  -> Vh row t.
            if (sv > 1e-300) {
                for (std::size_t j = 0; j < c; ++j) {
                    cd acc(0, 0);
                    for (std::size_t m = 0; m < r; ++m) acc += std::conj(Umat(m, t)) * A(m, j);
                    Vh(t, j) = acc / sv;
                }
            }
        }
    }
}

CMat hermitian_expm(const CMat& H, cd coeff) {
    std::vector<double> w;
    CMat U;
    hermitian_eig(H, w, U);
    const std::size_t n = H.rows;
    // exp(coeff·H) = U diag(exp(coeff·w)) U^†.
    CMat D(n, n);
    for (std::size_t k = 0; k < n; ++k) D(k, k) = std::exp(coeff * w[k]);
    // tmp = U D
    CMat tmp(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k)
            tmp(i, k) = U(i, k) * D(k, k);
    // out = tmp U^†
    CMat out(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            cd acc(0, 0);
            for (std::size_t k = 0; k < n; ++k)
                acc += tmp(i, k) * std::conj(U(j, k));
            out(i, j) = acc;
        }
    return out;
}

}  // namespace tessera
