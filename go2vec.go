// Copyright 2015 Daniël de Kok
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package go2vec

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/gonum/blas"
	cblas "github.com/gonum/blas/cgo"
)

// IterFunc is a function for iterating over word embeddings. The function
// should return 'false' if the iteration should be stopped.
type IterFunc func(word string, vector []float32) bool

// WordSimilarity stores the similarity of a word compared to a query word.
type WordSimilarity struct {
	Word       string
	Similarity float32
}

// Vector is an word embedding.
type Vector []float32

// Vectors stores embeddings for words.
type Vectors struct {
	blas    blas.Float32Level2
	matrix  []float32
	vecSize int
	indices map[string]int
	words   []string
}

// NewVectors creates a set of word embeddings from scratch. This constructor
// should be used in conjunction with 'Put' to populate the embeddings.
func NewVectors(vecSize int) *Vectors {
	return &Vectors{
		blas:    cblas.Implementation{},
		matrix:  make([]float32, 0),
		vecSize: vecSize,
		indices: make(map[string]int),
		words:   make([]string, 0),
	}
}

// ReadWord2VecBinary reads word embeddings from a binary file that is produced
// by word2vec. The vectors can be normalized using their L2 norms.
func ReadWord2VecBinary(r *bufio.Reader, normalize bool) (*Vectors, error) {
	var nWords uint64
	if _, err := fmt.Fscanf(r, "%d", &nWords); err != nil {
		return nil, err
	}

	var vSize uint64
	if _, err := fmt.Fscanf(r, "%d", &vSize); err != nil {
		return nil, err
	}

	matrix := make([]float32, nWords*vSize)
	indices := make(map[string]int)
	words := make([]string, nWords)

	for idx := 0; idx < int(nWords); idx++ {
		word, err := r.ReadString(' ')
		word = strings.TrimSpace(word)
		indices[word] = idx
		words[idx] = word

		start := idx * int(vSize)
		if err = binary.Read(r, binary.LittleEndian, matrix[start:start+int(vSize)]); err != nil {
			return nil, err
		}

		if normalize {
			normalizeVectors(matrix[start : start+int(vSize)])
		}
	}

	return &Vectors{
		blas:    cblas.Implementation{},
		matrix:  matrix,
		vecSize: int(vSize),
		indices: indices,
		words:   words,
	}, nil
}

// Write vectors to a binary file accepted by word2vec
func (v *Vectors) Write(w *bufio.Writer) error {
	nWords := len(v.words)
	if nWords == 0 {
		return nil
	}

	if v.vecSize == 0 {
		return nil
	}

	if _, err := fmt.Fprintf(w, "%d %d\n", nWords, v.vecSize); err != nil {
		return err
	}

	for idx, word := range v.words {
		if _, err := w.WriteString(word + " "); err != nil {
			return err
		}

		if err := binary.Write(w, binary.LittleEndian, v.lookupIdx(idx)); err != nil {
			return err
		}
	}

	return nil
}

// Analogy performs word analogy queries.
//
// Consider an analogy of the form 'word1' is to 'word2' as 'word3' is to
// 'word4'. This method returns candidates for 'word4' based on 'word1..3'.
//
// If 'v1' is the vector of 'word1', etc., then the vector
// 'v4 = (v2 - v1) + v3' is computed. Then the words with vectors that are
// the most similar to v4 are returned.
//
// The query words are never returned as a result.
func (v *Vectors) Analogy(word1, word2, word3 string, limit int) ([]WordSimilarity, error) {
	idx1, ok := v.indices[word1]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word1)
	}

	idx2, ok := v.indices[word2]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word2)
	}

	idx3, ok := v.indices[word3]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word3)
	}

	v1 := v.lookupIdx(idx1)
	v2 := v.lookupIdx(idx2)
	v3 := v.lookupIdx(idx3)

	v4 := plus(minus(v2, v1), v3)

	skips := map[int]interface{}{
		idx1: nil,
		idx2: nil,
		idx3: nil,
	}

	return v.similarity(v4, skips, limit)
}

// SetBLAS sets the BLAS implementation to use (default: C BLAS).
func (v *Vectors) SetBLAS(impl blas.Float32Level2) {
	v.blas = impl
}

// Iterate applies the provided iteration function to all word embeddings.
func (v *Vectors) Iterate(f IterFunc) {
	for idx, word := range v.words {
		if !f(word, v.lookupIdx(idx)) {
			break
		}
	}
}

// Put adds a word embedding to the word embeddings. The new word can be
// queried after the call returns.
func (v *Vectors) Put(word string, vector []float32) error {
	if len(vector) != v.vecSize {
		return fmt.Errorf("Expected vector size: %d, got: %d", v.vecSize, len(vector))
	}

	if idx, ok := v.indices[word]; ok {
		// The word is already known, replace its vector.
		copy(v.matrix[idx*v.vecSize:], vector)
	} else {
		// The word is not known, add it and allocate memory.
		v.indices[word] = len(v.words)
		v.words = append(v.words, word)
		v.matrix = append(v.matrix, vector...)
	}

	return nil
}

// Similarity finds words that have embeddings that are similar to that of
// the given word. The 'limit' argument specifis how many words should be
// returned. The returned slice is ordered by similarity.
//
// The query word is never returned as a result.
func (v Vectors) Similarity(word string, limit int) ([]WordSimilarity, error) {
	idx, ok := v.indices[word]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word)
	}

	skips := map[int]interface{}{
		idx: nil,
	}

	return v.similarity(v.lookupIdx(idx), skips, limit)
}

// Size returns the number of words in the embeddings.
func (v *Vectors) Size() int {
	return len(v.indices)
}

// Vector returns the embedding for a particular word. If the word is
// unknown, the second return value will be false.
func (v *Vectors) Vector(word string) ([]float32, bool) {
	if idx, ok := v.indices[word]; ok {
		return v.lookupIdx(idx), true
	}

	return nil, false
}

// VectorSize returns the embedding size.
func (v *Vectors) VectorSize() int {
	return v.vecSize
}

// WordIdx returns the index of the word within an embedding.
func (v *Vectors) WordIdx(word string) (int, bool) {
	if idx, ok := v.indices[word]; ok {
		return idx, ok
	}

	return 0, false
}

func (v Vectors) similarity(vec Vector, skips map[int]interface{}, limit int) ([]WordSimilarity, error) {
	dps := make([]float32, v.Size())
	v.blas.Sgemv(blas.NoTrans, int(v.Size()), int(v.VectorSize()),
		1, v.matrix, int(v.VectorSize()), vec, 1, 0, dps, 1)

	var results []WordSimilarity
	for idx, sim := range dps {
		// Skip words in the skip set.
		if _, ok := skips[idx]; ok {
			continue
		}

		ip := sort.Search(len(results), func(i int) bool {
			return results[i].Similarity <= sim
		})
		if ip < limit {
			results = insertWithLimit(results, limit, ip, WordSimilarity{v.words[idx], sim})
		}
	}

	return results, nil
}

func dotProduct(v, w []float32) float32 {
	sum := float32(0)

	for idx, val := range v {
		sum += val * w[idx]
	}

	return sum
}

func insertWithLimit(slice []WordSimilarity, limit, index int, value WordSimilarity) []WordSimilarity {
	if len(slice) < limit {
		slice = append(slice, WordSimilarity{})
	}

	copy(slice[index+1:], slice[index:len(slice)-1])
	slice[index] = value
	return slice
}

// Look up the vector at the given index.
func (v *Vectors) lookupIdx(idx int) Vector {
	start := idx * v.vecSize
	return v.matrix[start : start+v.vecSize]
}

func minus(v, w []float32) []float32 {
	result := make([]float32, len(v))

	for idx, val := range v {
		result[idx] = val - w[idx]
	}

	return result
}

// Normalize a vector using its l2-norm.
func normalizeVectors(vec []float32) {
	// Normalize
	vecLen := float32(0)
	for _, val := range vec {
		vecLen += val * val
	}

	vecLen = float32(math.Sqrt(float64(vecLen)))

	for idx, val := range vec {
		vec[idx] = val / vecLen
	}
}

func plus(v, w []float32) []float32 {
	result := make([]float32, len(v))

	for idx, val := range v {
		result[idx] = val + w[idx]
	}

	return result
}
