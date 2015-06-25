package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/danieldk/dparnn/cmd/common"
	"github.com/danieldk/go2vec"
)

func main() {
	flag.Parse()

	if flag.NArg() != 1 {
		os.Exit(1)
	}

	f, err := os.Open(flag.Arg(0))
	common.ExitIfError("Cannot open file: ", err)
	defer f.Close()

	vectors, err := go2vec.ReadVectors(bufio.NewReader(f), true)
	common.ExitIfError("Cannot read vectors: ", err)

	//fmt.Printf("%d %d\n", vectors.Size(), vectors.VectorSize())

	vectors.Iterate(func(word string, vector []float32) bool {
		fmt.Print(word + " ")

		fmt.Println(floatSliceToString(vector))

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
