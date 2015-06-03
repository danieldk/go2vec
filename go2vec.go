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
)

type IterFunc func(word string, vector []float32) bool

type WordSimilarity struct {
	Word       string
	Similarity float32
}

type Vector []float32

type Vectors struct {
	matrix  []float32
	vecSize uint64
	indices map[string]uint64
	words   []string
}

// Create new vectors from scratch, to be used in combination
// with 'PutVector'.
func NewVectors(vecSize uint64) *Vectors {
	return &Vectors{
		matrix:  make([]float32, 0),
		vecSize: vecSize,
		indices: make(map[string]uint64),
		words:   make([]string, 0),
	}
}

// Read vectors from a binary file produced by word2vec.
func ReadVectors(r *bufio.Reader) (*Vectors, error) {
	var nWords uint64
	if _, err := fmt.Fscanf(r, "%d", &nWords); err != nil {
		return nil, err
	}

	var vSize uint64
	if _, err := fmt.Fscanf(r, "%d", &vSize); err != nil {
		return nil, err
	}

	matrix := make([]float32, nWords*vSize)
	indices := make(map[string]uint64)
	words := make([]string, nWords)

	for idx := uint64(0); idx < nWords; idx++ {
		word, err := r.ReadString(' ')
		word = strings.TrimSpace(word)
		indices[word] = idx
		words[idx] = word

		start := idx * vSize
		if err = binary.Read(r, binary.LittleEndian, matrix[start:start+vSize]); err != nil {
			return nil, err
		}

		normalize(matrix[start : start+vSize])
	}

	return &Vectors{
		matrix:  matrix,
		vecSize: vSize,
		indices: indices,
		words:   words,
	}, nil
}

// Write vectors to a binary file accepted by word2vec
func (vectors *Vectors) Write(w *bufio.Writer) error {
	nWords := len(vectors.words)
	if nWords == 0 {
		return nil
	}

	if vectors.vecSize == 0 {
		return nil
	}

	if _, err := fmt.Fprintf(w, "%d %d\n", nWords, vectors.vecSize); err != nil {
		return err
	}

	for word, idx := range vectors.indices {
		if _, err := w.WriteString(word + " "); err != nil {
			return err
		}

		if err := binary.Write(w, binary.LittleEndian, vectors.lookupIdx(idx)); err != nil {
			return err
		}
	}

	return nil
}

// Consider an analogy of the form 'word1' is to 'word2' as 'word3' is to
// 'word4'. This method returns candidates for 'word4' based on 'word1..3'.
//
// If 'v1' is the vector of 'word1', etc., then the vector
// 'v4 = (v2 - v1) + v3' is computed. Then the words with vectors that are
// the most similar to v4 are returned.
//
// The query words are never returned as a result.
func (vecs *Vectors) Analogy(word1, word2, word3 string, limit int) ([]WordSimilarity, error) {
	idx1, ok := vecs.indices[word1]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word1)
	}

	idx2, ok := vecs.indices[word2]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word2)
	}

	idx3, ok := vecs.indices[word3]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word3)
	}

	v1 := vecs.lookupIdx(idx1)
	v2 := vecs.lookupIdx(idx2)
	v3 := vecs.lookupIdx(idx3)

	v4 := plus(minus(v2, v1), v3)

	skips := map[uint64]interface{}{
		idx1: nil,
		idx2: nil,
		idx3: nil,
	}

	return vecs.similarity(v4, skips, limit)
}

func (v *Vectors) Iterate(f IterFunc) {
	for idx, word := range v.words {
		if !f(word, v.lookupIdx(uint64(idx))) {
			break
		}
	}
}

func (v *Vectors) Put(word string, vector []float32) error {
	if uint64(len(vector)) != v.vecSize {
		return fmt.Errorf("Expected vector size: %d, got: %d", v.vecSize, len(vector))
	}

	if idx, ok := v.indices[word]; ok {
		// The word is already known, replace its vector.
		copy(v.matrix[idx*v.vecSize:], vector)
	} else {
		// The word is not known, add it and allocate memory.
		idx = uint64(len(v.words))
		v.indices[word] = idx
		v.words = append(v.words, word)
		v.matrix = append(v.matrix, vector...)
	}

	return nil
}

// Find words that have vectors that are similar to that of the given word.
// The 'limit' argument specifis how many words should be returned. The
// returned slice is ordered by similarity.
//
// The query word is never returned as a result.
func (vecs Vectors) Similarity(word string, limit int) ([]WordSimilarity, error) {
	idx, ok := vecs.indices[word]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word)
	}

	skips := map[uint64]interface{}{
		idx: nil,
	}

	return vecs.similarity(vecs.lookupIdx(idx), skips, limit)
}

func (v *Vectors) Size() uint64 {
	return uint64(len(v.indices))
}

func (v *Vectors) Vector(word string) ([]float32, bool) {
	if idx, ok := v.indices[word]; ok {
		return v.lookupIdx(idx), true
	}

	return nil, false
}

func (v *Vectors) VectorSize() uint64 {
	return v.vecSize
}

func (v *Vectors) WordIdx(word string) (uint64, bool) {
	if idx, ok := v.indices[word]; ok {
		return idx, ok
	}

	return 0, false
}

func (vecs Vectors) similarity(vec Vector, skips map[uint64]interface{}, limit int) ([]WordSimilarity, error) {
	results := make([]WordSimilarity, 0)

	for idx := uint64(0); idx < uint64(len(vecs.words)); idx++ {
		// Skip words in the skip set.
		if _, ok := skips[idx]; ok {
			continue
		}

		sim := dotProduct(vec, vecs.lookupIdx(idx))

		ip := sort.Search(len(results), func(i int) bool {
			return results[i].Similarity <= sim
		})
		if ip < limit {
			results = insertWithLimit(results, limit, ip, WordSimilarity{vecs.words[idx], sim})
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
func (v *Vectors) lookupIdx(idx uint64) Vector {
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
func normalize(vec []float32) {
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
