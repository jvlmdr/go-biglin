package biglinalg

import "github.com/jackvalmadre/vector"

//
//
//
type Objective interface {
	VectorType() vec.Type
	Evaluate(x vec.ConstTyped, y *float64, dfdx *vec.MutableTyped) error
	LineSearch(pos, dir vec.ConstTyped) (float64, error)
}

//
//
//
type Regression struct {
	A Matrix
	B vec.ConstTyped
}

func (f Regression) VectorType() vec.Type {
	return f.B.Type()
}

func (f Regression) Evaluate(x vec.ConstTyped, y *float64, dfdx *vec.MutableTyped) error {
	// r = A x - b
	r, err := f.A.Times(x, false)
	if err != nil {
		return err
	}
	r = vec.Subtract(r, f.B)
	if y != nil {
		*y = 0.5 * vec.SqrNorm(r)
	}
	if dfdx != nil {
		// d/dx 1/2 ||r||^2 = A' r
		*dfdx, err = f.A.Times(r, true)
		if err != nil {
			return err
		}
	}
	return nil
}

func (f Regression) LineSearch(x, v vec.ConstTyped) (float64, error) {
	// r = A x - b
	r, err := f.A.Times(x, false)
	if err != nil {
		return 0, err
	}
	r = vec.Subtract(r, f.B)
	// q = A v
	q, err := f.A.Times(v, false)
	if err != nil {
		return 0, err
	}
	// ||A (x + k v) - b|| = ||r + k q||
	alpha := quadraticLineSearch(r, q)
	return alpha, nil
}

// Finds k which minimizes ||r + k q||
func quadraticLineSearch(r, q vec.ConstTyped) float64 {
	return -vec.Dot(r, q) / vec.SqrNorm(q)
}

//
//
//
type RidgeRegression struct {
	A      Matrix
	B      vec.ConstTyped
	Lambda float64
}

func (f RidgeRegression) VectorType() vec.Type {
	return f.B.Type()
}

func (f RidgeRegression) Evaluate(x vec.ConstTyped, y *float64, dfdx *vec.MutableTyped) error {
	// r = A x - b
	r, err := f.A.Times(x, false)
	if err != nil {
		return err
	}
	r = vec.Subtract(r, f.B)
	if y != nil {
		*y = 0.5*vec.SqrNorm(r) + 0.5*f.Lambda*vec.SqrNorm(x)
	}
	if dfdx != nil {
		// d/dx 1/2 ||r||^2 = A' r
		*dfdx, err = f.A.Times(r, true)
		if err != nil {
			return err
		}
		// df/dx = A' (A x - b) + lambda x
		*dfdx = vec.CombineLinear(1, *dfdx, f.Lambda, x)
	}
	return nil
}

func (f RidgeRegression) LineSearch(x, v vec.ConstTyped) (float64, error) {
	// r = A x - b
	r, err := f.A.Times(x, false)
	if err != nil {
		return 0, err
	}
	r = vec.Subtract(r, f.B)
	// q = A v
	q, err := f.A.Times(v, false)
	if err != nil {
		return 0, err
	}
	// ||A (x + k v) - b|| = ||r + k q||
	alpha := biquadraticLineSearch(r, q, x, v, f.Lambda)
	return alpha, nil
}

// Finds k which minimizes ||r + k q||^2 + lambda ||x + k s||^2
func biquadraticLineSearch(r, q, x, s vec.ConstTyped, lambda float64) float64 {
	a := vec.Dot(r, q) + lambda*vec.Dot(x, s)
	b := vec.SqrNorm(q) + lambda*vec.SqrNorm(s)
	return -a / b
}
