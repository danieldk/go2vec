package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/danieldk/go2vec/v2"
	"github.com/danieldk/go2vec/v2/cmd/common"
)

func main() {
	flag.Parse()

	if flag.NArg() != 1 {
		os.Exit(1)
	}

	f, err := os.Open(flag.Arg(0))
	common.ExitIfError("Cannot open file: ", err)
	defer f.Close()

	embeds, err := go2vec.ReadWord2VecBinary(bufio.NewReader(f), true)
	common.ExitIfError("Cannot read vectors: ", err)

	embeds.Iterate(func(word string, embedding []float32) bool {
		fmt.Print(word + " ")

		fmt.Println(floatSliceToString(embedding))

		return true
	})
}

func floatSliceToString(floats []float32) string {
	stringFloats := make([]string, len(floats))

	for idx, float := range floats {
		stringFloats[idx] = strconv.FormatFloat(float64(float), 'f', 6, 32)
	}

	return strings.Join(stringFloats, " ")
}
