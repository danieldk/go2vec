package common

import "log"

func ExitIfError(prefix string, err error) {
	if err != nil {
		log.Fatal(prefix, err.Error())
	}
}
