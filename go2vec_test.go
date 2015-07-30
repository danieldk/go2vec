package go2vec

import (
	"bufio"
	"os"
	"testing"
)

var analogyOrder = []string{
	"Deutschland",
	"Westdeutschland",
	"Sachsen",
	"Mitteldeutschland",
	"Brandenburg",
	"Polen",
	"Norddeutschland",
	"Dänemark",
	"Schleswig-Holstein",
	"Österreich",
	"Bayern",
	"Thüringen",
	"Bundesrepublik",
	"Ostdeutschland",
	"Preußen",
	"Deutschen",
	"Hessen",
	"Potsdam",
	"Mecklenburg",
	"Niedersachsen",
	"Hamburg",
	"Süddeutschland",
	"Bremen",
	"Russland",
	"Deutschlands",
	"BRD",
	"Litauen",
	"Mecklenburg-Vorpommern",
	"DDR",
	"West-Berlin",
	"Saarland",
	"Lettland",
	"Hannover",
	"Rostock",
	"Sachsen-Anhalt",
	"Pommern",
	"Schweden",
	"Deutsche",
	"deutschen",
	"Westfalen"}

var similarityOrder = []string{
	"Potsdam",
	"Hamburg",
	"Leipzig",
	"Dresden",
	"München",
	"Düsseldorf",
	"Bonn",
	"Stuttgart",
	"Weimar",
	"Berlin-Charlottenburg",
	"Rostock",
	"Karlsruhe",
	"Chemnitz",
	"Breslau",
	"Wiesbaden",
	"Hannover",
	"Mannheim",
	"Kassel",
	"Köln",
	"Danzig",
	"Erfurt",
	"Dessau",
	"Bremen",
	"Charlottenburg",
	"Magdeburg",
	"Neuruppin",
	"Darmstadt",
	"Jena",
	"Wien",
	"Heidelberg",
	"Dortmund",
	"Stettin",
	"Schwerin",
	"Neubrandenburg",
	"Greifswald",
	"Göttingen",
	"Braunschweig",
	"Berliner",
	"Warschau",
	"Berlin-Spandau"}

func readEmbeddingsOrFail(t *testing.T, filename string) *Embeddings {
	f, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	embeds, err := ReadWord2VecBinary(bufio.NewReader(f), true)
	if err != nil {
		t.Fatal(err)
	}

	return embeds
}

func TestAnalogy(t *testing.T) {
	embeds := readEmbeddingsOrFail(t, "analogy.bin")
	answers, err := embeds.Analogy("Paris", "Frankreich", "Berlin", 40)
	if err != nil {
		t.Fatalf("Analogy error should be nil, was: %s", err)
	}

	for idx, wordSimilarity := range answers {
		if wordSimilarity.Word != analogyOrder[idx] {
			t.Errorf("Word at position %d should be '%s', was '%s'", idx, wordSimilarity.Word, analogyOrder[idx])
		}
	}
}

func TestBasicEmpty(t *testing.T) {
	embeds := NewEmbeddings(2)

	if embeds.Size() != 0 {
		t.Errorf("Embeddings should have size 0, was %d", embeds.Size())
	}

	if embeds.EmbeddingSize() != 2 {
		t.Errorf("Embedding size should be 2, was %d", embeds.Size())
	}

	embeds.Put("apple", []float32{1.0, 0.0})
	embeds.Put("pear", []float32{0.8, 0.1})
	embeds.Put("banana", []float32{0.2, 1.0})

	if embeds.Size() != 3 {
		t.Errorf("Embeddings should have size 3, was %d", embeds.Size())
	}

	if embeds.EmbeddingSize() != 2 {
		t.Errorf("Embeddings size should be 2, was %d", embeds.Size())
	}
}

func TestBasicFromFile(t *testing.T) {
	embeds := readEmbeddingsOrFail(t, "similarity.bin")

	if embeds.Size() != 41 {
		t.Errorf("Embeddings should have size 41, was %d", embeds.Size())
	}

	if embeds.EmbeddingSize() != 100 {
		t.Errorf("Embeddings size should be 100, was %d", embeds.Size())
	}

	if _, ok := embeds.Embedding("Bogus"); ok {
		t.Error("An unknown word should return ok==false")
	}

	if _, ok := embeds.Embedding("Berlin"); !ok {
		t.Error("An unknown word should return ok==true")
	}

	if _, ok := embeds.Embedding("Berlin-Spandau"); !ok {
		t.Error("An unknown word should return ok==true")
	}
}

func TestSimilarity(t *testing.T) {
	embeds := readEmbeddingsOrFail(t, "similarity.bin")
	answers, err := embeds.Similarity("Berlin", 40)
	if err != nil {
		t.Fatalf("Similarity error should be nil, was: %s", err)
	}

	for idx, wordSimilarity := range answers {
		if wordSimilarity.Word != similarityOrder[idx] {
			t.Errorf("Word at position %d should be '%s', was '%s'", idx, wordSimilarity.Word, analogyOrder[idx])
		}
	}
}
