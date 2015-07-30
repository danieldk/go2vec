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

func readVectorsOrFail(t *testing.T, filename string) *Vectors {
	f, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	vecs, err := ReadVectors(bufio.NewReader(f), true)
	if err != nil {
		t.Fatal(err)
	}

	return vecs
}

func TestAnalogy(t *testing.T) {
	vecs := readVectorsOrFail(t, "analogy.bin")
	answers, err := vecs.Analogy("Paris", "Frankreich", "Berlin", 40)
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
	vecs := NewVectors(2)

	if vecs.Size() != 0 {
		t.Errorf("Vectors should have size 0, was %d", vecs.Size())
	}

	if vecs.VectorSize() != 2 {
		t.Errorf("Vector size should be 2, was %d", vecs.Size())
	}

	vecs.Put("apple", []float32{1.0, 0.0})
	vecs.Put("pear", []float32{0.8, 0.1})
	vecs.Put("banana", []float32{0.2, 1.0})

	if vecs.Size() != 3 {
		t.Errorf("Vectors should have size 3, was %d", vecs.Size())
	}

	if vecs.VectorSize() != 2 {
		t.Errorf("Vector size should be 2, was %d", vecs.Size())
	}
}

func TestBasicFromFile(t *testing.T) {
	vecs := readVectorsOrFail(t, "similarity.bin")

	if vecs.Size() != 41 {
		t.Errorf("Vectors should have size 41, was %d", vecs.Size())
	}

	if vecs.VectorSize() != 100 {
		t.Errorf("Vector size should be 100, was %d", vecs.Size())
	}

	if _, ok := vecs.Vector("Bogus"); ok {
		t.Error("An unknown word should return ok==false")
	}

	if _, ok := vecs.Vector("Berlin"); !ok {
		t.Error("An unknown word should return ok==true")
	}

	if _, ok := vecs.Vector("Berlin-Spandau"); !ok {
		t.Error("An unknown word should return ok==true")
	}
}

func TestSimilarity(t *testing.T) {
	vecs := readVectorsOrFail(t, "similarity.bin")
	answers, err := vecs.Similarity("Berlin", 40)
	if err != nil {
		t.Fatalf("Similarity error should be nil, was: %s", err)
	}

	for idx, wordSimilarity := range answers {
		if wordSimilarity.Word != similarityOrder[idx] {
			t.Errorf("Word at position %d should be '%s', was '%s'", idx, wordSimilarity.Word, analogyOrder[idx])
		}
	}
}
