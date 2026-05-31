// Self-test for the linalg foundation (Hermitian eig + matrix exp).
// Verifies against hand-checkable cases before the MPS engine is built on top.
#define _USE_MATH_DEFINES
#include "linalg.hpp"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdio>
#include <complex>

using namespace tessera;
static int failures = 0;

static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++failures;
}

// Frobenius distance between two complex matrices.
static double dist(const CMat& A, const CMat& B) {
    double d = 0.0;
    for (std::size_t i = 0; i < A.a.size(); ++i) d += std::norm(A.a[i] - B.a[i]);
    return std::sqrt(d);
}

int main() {
    const cd I(0, 1);

    // 1. Pauli Z eigenvalues are ±1.
    {
        CMat Z(2, 2);
        Z(0, 0) = 1; Z(1, 1) = -1;
        std::vector<double> w; CMat V;
        hermitian_eig(Z, w, V);
        bool ok = ((std::fabs(w[0] - 1) < 1e-9 && std::fabs(w[1] + 1) < 1e-9) ||
                   (std::fabs(w[0] + 1) < 1e-9 && std::fabs(w[1] - 1) < 1e-9));
        check(ok, "Pauli Z eigenvalues = {+1,-1}");
    }

    // 2. Pauli Y (genuinely complex Hermitian) eigenvalues are ±1.
    {
        CMat Y(2, 2);
        Y(0, 1) = -I; Y(1, 0) = I;
        std::vector<double> w; CMat V;
        hermitian_eig(Y, w, V);
        bool ok = std::fabs(std::fabs(w[0]) - 1) < 1e-9 && std::fabs(std::fabs(w[1]) - 1) < 1e-9
                  && std::fabs(w[0] + w[1]) < 1e-9;
        check(ok, "Pauli Y eigenvalues = {+1,-1}");
    }

    // 3. exp(-i·(pi/2)·X) = -i·X  (rotation identity).
    {
        CMat X(2, 2);
        X(0, 1) = 1; X(1, 0) = 1;
        CMat U = hermitian_expm(X, cd(0, -1) * (M_PI / 2));
        CMat target(2, 2);
        target(0, 1) = -I; target(1, 0) = -I;  // -i*X
        check(dist(U, target) < 1e-9, "exp(-i (pi/2) X) = -i X");
    }

    // 4. Reconstruction H = V diag(w) V^† for a random 3x3 Hermitian.
    {
        CMat H(3, 3);
        H(0, 0) = 2;     H(1, 1) = -1;    H(2, 2) = 0.5;
        H(0, 1) = cd(1, 2);  H(1, 0) = std::conj(H(0, 1));
        H(0, 2) = cd(-0.5, 1); H(2, 0) = std::conj(H(0, 2));
        H(1, 2) = cd(0, -1);   H(2, 1) = std::conj(H(1, 2));
        std::vector<double> w; CMat V;
        hermitian_eig(H, w, V);
        CMat R(3, 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                cd acc(0, 0);
                for (std::size_t k = 0; k < 3; ++k)
                    acc += V(i, k) * w[k] * std::conj(V(j, k));
                R(i, j) = acc;
            }
        check(dist(R, H) < 1e-7, "Hermitian reconstruction V diag(w) V^dag = H");
    }

    // 5. exp is unitary: U U^† = I for a random Hermitian, real-time.
    {
        CMat H(3, 3);
        H(0, 0) = 1; H(1, 1) = 2; H(2, 2) = -1;
        H(0, 1) = cd(0.3, 0.7); H(1, 0) = std::conj(H(0, 1));
        H(1, 2) = cd(-0.2, 0.5); H(2, 1) = std::conj(H(1, 2));
        CMat U = hermitian_expm(H, cd(0, -1) * 0.83);
        CMat P(3, 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                cd acc(0, 0);
                for (std::size_t k = 0; k < 3; ++k) acc += U(i, k) * std::conj(U(j, k));
                P(i, j) = acc;
            }
        CMat Id(3, 3); Id(0, 0) = Id(1, 1) = Id(2, 2) = 1;
        check(dist(P, Id) < 1e-9, "real-time exp is unitary (U U^dag = I)");
    }

    std::printf("%s\n", failures == 0 ? "ALL LINALG TESTS PASS" : "LINALG TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
