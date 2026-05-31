// TESSERA — minimal self-contained linear algebra for the MPS engine.
//
// No external deps (no Eigen/LAPACK) — bare metal. Everything we need for
// matrix-product-state TEBD reduces to ONE primitive: the eigendecomposition of
// a small complex Hermitian matrix. We get it from a real symmetric Jacobi
// eigensolver via the standard 2n real embedding of an n×n Hermitian matrix:
//
//     H = R + iI  (R symmetric, I antisymmetric)   ->   M = [[R, -I], [I, R]]
//
// M is real symmetric (2n×2n); each eigenvalue of H appears twice in M, and a
// complex eigenvector v = a + i b corresponds to the real embedding eigenvector
// [a; b]. This gives us, with no dedup tricks:
//   - matrix exponential exp(z·H) of a Hermitian H (for the TEBD gates), and
//   - the leading eigenvectors of a Hermitian density matrix (for truncation).
#pragma once
#include <complex>
#include <vector>
#include <cstddef>

namespace tessera {

using cd = std::complex<double>;

// Row-major complex matrix.
struct CMat {
    std::size_t rows = 0, cols = 0;
    std::vector<cd> a;
    CMat() = default;
    CMat(std::size_t r, std::size_t c) : rows(r), cols(c), a(r * c, cd(0, 0)) {}
    cd& operator()(std::size_t i, std::size_t j) { return a[i * cols + j]; }
    const cd& operator()(std::size_t i, std::size_t j) const { return a[i * cols + j]; }
};

// Eigenpairs of a real symmetric matrix (row-major, n×n) via cyclic Jacobi.
// Returns eigenvalues `w` (ascending not guaranteed) and eigenvectors as columns
// of `V` (n×n, row-major): A = V diag(w) V^T.
void jacobi_symmetric(const std::vector<double>& A, std::size_t n,
                      std::vector<double>& w, std::vector<double>& V);

// Eigendecomposition of an n×n complex Hermitian matrix H.
// Returns real eigenvalues `evals` (length n) and complex eigenvectors as the
// columns of `evecs` (n×n): H = evecs diag(evals) evecs^†.
void hermitian_eig(const CMat& H, std::vector<double>& evals, CMat& evecs);

// Matrix exponential exp(coeff · H) for Hermitian H, coeff a complex scalar
// (coeff = -i·dt → real-time evolution; coeff = -dt → imaginary-time).
CMat hermitian_expm(const CMat& H, cd coeff);

// Thin SVD of a complex matrix A (rows×cols): A = U diag(S) V^†.
//   U     : rows × k       (k = min(rows,cols)), orthonormal columns
//   S     : length k       singular values, descending, >= 0
//   Vh    : k × cols       (this is V^†, i.e. conjugate-transpose of V)
// Implemented via the eigendecomposition of the smaller of A^†A / A A^† — exact
// and dependency-free, which is all the MPS bond truncation needs.
void svd(const CMat& A, CMat& U, std::vector<double>& S, CMat& Vh);

}  // namespace tessera
