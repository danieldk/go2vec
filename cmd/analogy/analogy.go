package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/danieldk/go2vec"
)

func main() {
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "Usage: analogy vectors.bin")
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

	for {
		reader := bufio.NewReader(os.Stdin)
		line, err := reader.ReadString('\n')
		if err != nil {
			return
		}

		line = strings.TrimSpace(line)

		parts := strings.Split(line, " ")
		if len(parts) != 3 {
			fmt.Fprintf(os.Stderr, "Skipping line that does not have three words: %s\n", line)
			continue
		}

		results, err := vecs.Analogy(parts[0], parts[1], parts[2], 10)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			continue
		}

		for _, wordSimilarity := range results {
			fmt.Println(wordSimilarity.Word, wordSimilarity.Similarity)
		}
	}
}
