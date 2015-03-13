package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/danieldk/go2vec"
)

func main() {
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "Usage: distance vectors.bin")
		os.Exit(1)
	}

	f, err := os.Open(flag.Arg(0))
	defer f.Close()
	if err != nil {
		log.Fatal(err)
	}

	vecs, err := go2vec.ReadVectors(bufio.NewReader(io.Reader(f)))
	if err != nil {
		log.Fatal(err)
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		token := scanner.Text()
		results, err := vecs.Similarity(token, 10)
		if err != nil {
			fmt.Println(os.Stderr, err.Error())
			os.Exit(1)
		}

		for _, wordSimilarity := range results {
			fmt.Println(wordSimilarity.Word, wordSimilarity.Similarity)
		}
	}
}
