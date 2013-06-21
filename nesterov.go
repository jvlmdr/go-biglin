package biglin

import (
	"fmt"
	"github.com/jackvalmadre/go-vec"
	"math"
)

type AcceleratedGradientDescent struct {
	LineSearch bool
	StepSize   float64
	Backtrack  bool
}

func (opts AcceleratedGradientDescent) Solve(f Objective, x vec.ConstTyped,
		crit TerminationCriteria, callback Callback, verbose bool) (vec.MutableTyped, error) {
	k := 0
	alpha := opts.StepSize
	var t float64 = 1
	f_x := math.Inf(1)
	var y vec.ConstTyped
	var g_x0 vec.MutableTyped
	var x_prev vec.ConstTyped

	for {
		f_x_prev := f_x
		var g_x vec.MutableTyped
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

		if k == 0 {
			y = x
		}
		var f_y float64
		var g_y vec.MutableTyped
		err = f.Evaluate(y, &f_y, &g_y)
		if err != nil {
			return nil, err
		}

		x_prev = x
		if opts.LineSearch {
			// Update step size by exact line search.
			alpha, err = f.LineSearch(y, g_y)
			if err != nil {
				return nil, err
			}
			x = vec.CombineLinear(1, y, alpha, g_y)
		} else {
			if !opts.Backtrack {
				// No search involved.
				x = vec.CombineLinear(1, y, -alpha, g_y)
			} else {
				var z vec.MutableTyped
				for satisfied := false; !satisfied; {
					// Update with current alpha.
					z = vec.CombineLinear(1, y, -alpha, g_y)
					var f_z float64
					err = f.Evaluate(z, &f_z, nil)
					if err != nil {
						return nil, err
					}
					// Check if inequality is satisfied.
					lhs := f_y - f_z
					rhs := 0.5 * alpha * vec.SqrNorm(g_y)
					satisfied = (lhs >= rhs)
					if !satisfied {
						// Backtrack.
						alpha /= 2
					}
				}
				x = z
			}
		}

		t_next := (1+math.Sqrt(4*t*t+1)) / 2
		y = vec.CombineLinear(1, x, (t-1)/t_next, vec.Subtract(x, x_prev))
		t = t_next
		k += 1
	}

	return vec.Copy(x), nil
}
