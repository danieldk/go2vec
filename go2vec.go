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

func dotProduct(v, w []float32) float32 {
	sum := float32(0)

	for idx, val := range v {
		sum += val * w[idx]
	}

	return sum
}

func (v *Vectors) lookupIdx(idx uint64) Vector {
	start := idx * v.vecSize
	return v.matrix[start : start+v.vecSize]
}

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

func insertWithLimit(slice []WordSimilarity, limit, index int, value WordSimilarity) []WordSimilarity {
	if len(slice) < limit {
		slice = append(slice, WordSimilarity{})
	}

	copy(slice[index+1:], slice[index:len(slice)-1])
	slice[index] = value
	return slice
}

func minus(v, w []float32) []float32 {
	result := make([]float32, len(v))

	for idx, val := range v {
		result[idx] = val - w[idx]
	}

	return result
}

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
