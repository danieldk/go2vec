// Package go2vec loads word2vec embeddings.
//
// This package can load binary word2vec files. It also supports
// distance and analogy queries on the embeddings.
//
// go2vec uses gonum's C BLAS binding by default. Binding to the
// right BLAS library can give nice performance improvements. The
// binding can be configured using CGO flags. For instance, to link
// against OpenBLAS on Linux:
//
//     CGO_LDFLAGS="-L/path/to/OpenBLAS -lopenblas" go install github.com/gonum/blas/cgo
//
// or Accelerate on OS X:
//
//     CGO_LDFLAGS="-framework Accelerate" go install github.com/gonum/blas/cgo
package go2vec
