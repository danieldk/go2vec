package common

import "log"

// ExitIfError exits a program with a fatal error message, if
// the supplied error is not nil.
func ExitIfError(prefix string, err error) {
	if err != nil {
		log.Fatal(prefix, err.Error())
	}
}
