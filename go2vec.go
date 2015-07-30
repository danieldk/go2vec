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
type IterFunc func(word string, embedding []float32) bool

// WordSimilarity stores the similarity of a word compared to a query word.
type WordSimilarity struct {
	Word       string
	Similarity float32
}

// Embedding stores word representations in continuous space.
type Embedding []float32

// Embeddings is used to store a set of word embeddings, such that common
// operations can be performed on these embeddings (such as retrieving
// similar words).
type Embeddings struct {
	blas      blas.Float32Level2
	matrix    []float32
	embedSize int
	indices   map[string]int
	words     []string
}

// NewEmbeddings creates a set of word embeddings from scratch. This constructor
// should be used in conjunction with 'Put' to populate the embeddings.
func NewEmbeddings(embedSize int) *Embeddings {
	return &Embeddings{
		blas:      cblas.Implementation{},
		matrix:    make([]float32, 0),
		embedSize: embedSize,
		indices:   make(map[string]int),
		words:     make([]string, 0),
	}
}

// ReadWord2VecBinary reads word embeddings from a binary file that is produced
// by word2vec. The embeddings can be normalized using their L2 norms.
func ReadWord2VecBinary(r *bufio.Reader, normalize bool) (*Embeddings, error) {
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
			normalizeEmbeddings(matrix[start : start+int(vSize)])
		}
	}

	return &Embeddings{
		blas:      cblas.Implementation{},
		matrix:    matrix,
		embedSize: int(vSize),
		indices:   indices,
		words:     words,
	}, nil
}

// Write embeddings to a binary file accepted by word2vec
func (e *Embeddings) Write(w *bufio.Writer) error {
	nWords := len(e.words)
	if nWords == 0 {
		return nil
	}

	if e.embedSize == 0 {
		return nil
	}

	if _, err := fmt.Fprintf(w, "%d %d\n", nWords, e.embedSize); err != nil {
		return err
	}

	for idx, word := range e.words {
		if _, err := w.WriteString(word + " "); err != nil {
			return err
		}

		if err := binary.Write(w, binary.LittleEndian, e.lookupIdx(idx)); err != nil {
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
// If 'e1' is the embedding of 'word1', etc., then the embedding
// 'e4 = (e2 - e1) + e3' is computed. Then the words with embeddings that are
// the most similar to e4 are returned.
//
// The query words are never returned as a result.
func (e *Embeddings) Analogy(word1, word2, word3 string, limit int) ([]WordSimilarity, error) {
	idx1, ok := e.indices[word1]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word1)
	}

	idx2, ok := e.indices[word2]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word2)
	}

	idx3, ok := e.indices[word3]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word3)
	}

	v1 := e.lookupIdx(idx1)
	v2 := e.lookupIdx(idx2)
	v3 := e.lookupIdx(idx3)

	v4 := plus(minus(v2, v1), v3)

	skips := map[int]interface{}{
		idx1: nil,
		idx2: nil,
		idx3: nil,
	}

	return e.similarity(v4, skips, limit)
}

// SetBLAS sets the BLAS implementation to use (default: C BLAS).
func (e *Embeddings) SetBLAS(impl blas.Float32Level2) {
	e.blas = impl
}

// Iterate applies the provided iteration function to all word embeddings.
func (e *Embeddings) Iterate(f IterFunc) {
	for idx, word := range e.words {
		if !f(word, e.lookupIdx(idx)) {
			break
		}
	}
}

// Put adds a word embedding to the word embeddings. The new word can be
// queried after the call returns.
func (e *Embeddings) Put(word string, embedding []float32) error {
	if len(embedding) != e.embedSize {
		return fmt.Errorf("Expected embedding size: %d, got: %d", e.embedSize, len(embedding))
	}

	if idx, ok := e.indices[word]; ok {
		// The word is already known, replace its embedding.
		copy(e.matrix[idx*e.embedSize:], embedding)
	} else {
		// The word is not known, add it and allocate memory.
		e.indices[word] = len(e.words)
		e.words = append(e.words, word)
		e.matrix = append(e.matrix, embedding...)
	}

	return nil
}

// Similarity finds words that have embeddings that are similar to that of
// the given word. The 'limit' argument specifis how many words should be
// returned. The returned slice is ordered by similarity.
//
// The query word is never returned as a result.
func (e Embeddings) Similarity(word string, limit int) ([]WordSimilarity, error) {
	idx, ok := e.indices[word]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word)
	}

	skips := map[int]interface{}{
		idx: nil,
	}

	return e.similarity(e.lookupIdx(idx), skips, limit)
}

// Size returns the number of words in the embeddings.
func (e *Embeddings) Size() int {
	return len(e.indices)
}

// Embedding returns the embedding for a particular word. If the word is
// unknown, the second return value will be false.
func (e *Embeddings) Embedding(word string) ([]float32, bool) {
	if idx, ok := e.indices[word]; ok {
		return e.lookupIdx(idx), true
	}

	return nil, false
}

// EmbeddingSize returns the embedding size.
func (e *Embeddings) EmbeddingSize() int {
	return e.embedSize
}

// WordIdx returns the index of the word within an embedding.
func (e *Embeddings) WordIdx(word string) (int, bool) {
	if idx, ok := e.indices[word]; ok {
		return idx, ok
	}

	return 0, false
}

func (e Embeddings) similarity(embed Embedding, skips map[int]interface{}, limit int) ([]WordSimilarity, error) {
	dps := make([]float32, e.Size())
	e.blas.Sgemv(blas.NoTrans, int(e.Size()), int(e.EmbeddingSize()),
		1, e.matrix, int(e.EmbeddingSize()), embed, 1, 0, dps, 1)

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
			results = insertWithLimit(results, limit, ip, WordSimilarity{e.words[idx], sim})
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

// Look up the embedding at the given index.
func (e *Embeddings) lookupIdx(idx int) Embedding {
	start := idx * e.embedSize
	return e.matrix[start : start+e.embedSize]
}

func minus(v, w []float32) []float32 {
	result := make([]float32, len(v))

	for idx, val := range v {
		result[idx] = val - w[idx]
	}

	return result
}

// Normalize an embedding using its l2-norm.
func normalizeEmbeddings(embedding []float32) {
	// Normalize
	embedLen := float32(0)
	for _, val := range embedding {
		embedLen += val * val
	}

	embedLen = float32(math.Sqrt(float64(embedLen)))

	for idx, val := range embedding {
		embedding[idx] = val / embedLen
	}
}

func plus(v, w []float32) []float32 {
	result := make([]float32, len(v))

	for idx, val := range v {
		result[idx] = val + w[idx]
	}

	return result
}
