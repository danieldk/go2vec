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

	embeds, err := go2vec.ReadWord2VecBinary(bufio.NewReader(io.Reader(f)), true)
	if err != nil {
		log.Fatal(err)
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()

		parts := strings.Split(line, " ")
		if len(parts) != 3 {
			fmt.Fprintf(os.Stderr, "Skipping line that does not have three words: %s\n", line)
			continue
		}

		results, err := embeds.Analogy(parts[0], parts[1], parts[2], 10)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			continue
		}

		for _, wordSimilarity := range results {
			fmt.Println(wordSimilarity.Word, wordSimilarity.Similarity)
		}
	}
}
