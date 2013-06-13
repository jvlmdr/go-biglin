package biglinalg

import "github.com/jackvalmadre/vector"

type Matrix interface {
	// Type checking is done at runtime to make the code more readable.
	// Otherwise the abstraction has to be a vector with a pointer to a constant
	// matrix, which knows how to construct a zero vector of itself.
	//
	// This function returns an error if the types or dimensions are wrong.
	Times(x vec.ConstTyped, transpose bool) (vec.MutableTyped, error)
}
