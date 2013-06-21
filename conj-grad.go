package biglin

import (
	"fmt"
	"github.com/jackvalmadre/go-vec"
	"math"
)

type NonlinearConjugateGradient struct{}

func (NonlinearConjugateGradient) Solve(f Objective, x vec.ConstTyped,
		crit TerminationCriteria, callback Callback, verbose bool) (vec.MutableTyped, error) {
	k := 0
	f_x := math.Inf(1)
	var g_x vec.MutableTyped
	var delta vec.MutableTyped
	var s vec.MutableTyped
	var g_x0 vec.MutableTyped
	var x_prev vec.ConstTyped

	for {
		f_x_prev := f_x
		err := f.Evaluate(x, &f_x, &g_x)
		if err != nil {
			return nil, err
		}
		if k == 0 {
			g_x0 = g_x
		}

		summary := Summarize(k, f_x_prev, f_x, g_x0, g_x, x_prev, x)
		fmt.Println(summary)
		if callback != nil {
			callback(summary)
		}
		converged := crit.Evaluate(summary)
		if converged {
			break
		}

		// Get gradient direction.
		delta_prev := delta
		delta = vec.Scale(-1, g_x)
		if k == 0 {
			// Use gradient direction.
			s = delta
		} else {
			// Get conjugate direction.
			beta := vec.SqrNorm(delta) / vec.SqrNorm(delta_prev)
			s = vec.CombineLinear(1, delta, beta, s)
		}
		// Perform exact line search.
		alpha, err := f.LineSearch(x, s)
		if err != nil {
			return nil, err
		}
		x_prev = x
		x = vec.CombineLinear(1, x, alpha, s)

		k += 1
	}

	return vec.Copy(x), nil
}
