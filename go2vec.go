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

type Vectors map[string]Vector

func ReadVectors(r *bufio.Reader) (Vectors, error) {
	var nWords uint64
	if _, err := fmt.Fscanf(r, "%d", &nWords); err != nil {
		return nil, err
	}

	var vSize uint64
	if _, err := fmt.Fscanf(r, "%d", &vSize); err != nil {
		return nil, err
	}

	vecs := make(Vectors)

	for w := uint64(0); w < nWords; w++ {
		word, err := r.ReadString(' ')
		word = strings.TrimSpace(word)
		vec := make([]float32, vSize)

		if err = binary.Read(r, binary.LittleEndian, vec); err != nil {
			return nil, err
		}

		normalize(vec)

		vecs[word] = vec
	}

	return vecs, nil
}

func dotProduct(v, w []float32) float32 {
	sum := float32(0)

	for idx, val := range v {
		sum += val * w[idx]
	}

	return sum
}

func (vecs Vectors) Analogy(word1, word2, word3 string, limit int) ([]WordSimilarity, error) {
	v1, ok := vecs[word1]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word1)
	}

	v2, ok := vecs[word2]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word2)
	}

	v3, ok := vecs[word3]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word3)
	}

	v4 := plus(minus(v2, v1), v3)

	skips := map[string]interface{}{
		word1: nil,
		word2: nil,
		word3: nil,
	}

	return vecs.similarity(v4, skips, limit)
}

func (vecs Vectors) Similarity(word string, limit int) ([]WordSimilarity, error) {
	v, ok := vecs[word]
	if !ok {
		return nil, fmt.Errorf("Unknown word: %s", word)
	}

	skips := map[string]interface{}{
		word: nil,
	}

	return vecs.similarity(v, skips, limit)
}

func (vecs Vectors) similarity(vec Vector, skips map[string]interface{}, limit int) ([]WordSimilarity, error) {
	results := make([]WordSimilarity, 0)

	for vecWord, w := range vecs {
		// Skip words in the skip set.
		if _, ok := skips[vecWord]; ok {
			continue
		}

		sim := dotProduct(vec, w)

		ip := sort.Search(len(results), func(i int) bool {
			return results[i].Similarity <= sim
		})
		if ip < limit {
			results = insertWithLimit(results, limit, ip, WordSimilarity{vecWord, sim})
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
