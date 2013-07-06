package biglin

import (
	"fmt"
	"github.com/jackvalmadre/go-vec"
	"math"
)

type NonlinearConjugateGradient struct{}

func (NonlinearConjugateGradient) Solve(f Objective, x0 vec.ConstTyped,
	crit TerminationCriteria, callback Callback, verbose bool) (vec.MutableTyped, error) {
	k := 0
	f_x := math.Inf(1)
	x := vec.Clone(x0)
	space := x0.Type()
	var (
		g_x    vec.MutableTyped
		delta  vec.MutableTyped
		s      vec.MutableTyped
		g_x0   vec.MutableTyped
		x_prev vec.ConstTyped
	)

	for {
		f_x_prev := f_x
		err := f.Evaluate(x, &f_x, &g_x)
		if err != nil {
			return nil, err
		}
		if k == 0 {
			g_x0 = vec.Clone(g_x)
		}

		summary := Summarize(k, f_x_prev, f_x, g_x0, g_x, x_prev, x)
		if verbose {
			fmt.Println(summary)
		}
		if callback != nil {
			callback(summary)
		}
		converged := crit.Evaluate(summary)
		if converged {
			break
		}

		// Get gradient direction.
		delta_prev := delta
		delta = space.New()
		vec.CopyTo(delta, vec.Scale(-1, g_x))
		if k == 0 {
			// Use gradient direction.
			// TODO: Replace with shallow copy?
			s = vec.Clone(delta)
		} else {
			// Get conjugate direction.
			beta := vec.SqrNorm(delta) / vec.SqrNorm(delta_prev)
			vec.CopyTo(s, vec.Plus(delta, vec.Scale(beta, s)))
		}
		// Perform exact line search.
		alpha, err := f.LineSearch(x, s)
		if err != nil {
			return nil, err
		}
		x_prev = vec.Clone(x)
		vec.CopyTo(x, vec.Plus(x, vec.Scale(alpha, s)))

		k += 1
	}

	return x, nil
}
