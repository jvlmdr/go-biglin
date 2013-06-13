package biglinalg

import (
	"fmt"
	"github.com/jackvalmadre/vector"
	"math"
)

type Callback func(summary IterationSummary)

type TerminationCriteria struct {
	MaxNumIterations   int
	FunctionTolerance  float64
	FunctionEpsilon    float64
	GradientTolerance  float64
	GradientEpsilon    float64
	ParameterTolerance float64
	ParameterEpsilon   float64
}

func (crit TerminationCriteria) Evaluate(it IterationSummary) bool {
	if it.Iteration >= crit.MaxNumIterations {
		return true
	}
	df := math.Abs(it.CostChange) / (it.Cost+crit.FunctionEpsilon)
	if df <= crit.FunctionTolerance {
		return true
	}
	g := math.Abs(it.GradientNorm) / (it.GradientNormInit+crit.GradientEpsilon)
	if g <= crit.GradientTolerance {
		return true
	}
	dx := it.StepNorm / (it.ParameterNorm+crit.ParameterEpsilon)
	if dx <= crit.ParameterTolerance {
		return true
	}
	return false
}

func DefaultTerminationCriteria() TerminationCriteria {
	var crit TerminationCriteria
	crit.MaxNumIterations = 50
	crit.FunctionTolerance = 1e-6
	crit.FunctionEpsilon = 0
	crit.GradientTolerance = 1e-10
	crit.GradientEpsilon = 0
	crit.ParameterTolerance = 1e-8
	crit.ParameterEpsilon = 1e-8
	return crit
}

type IterationSummary struct {
	Iteration        int
	Cost             float64
	CostChange       float64
	GradientNorm     float64
	GradientNormInit float64
	ParameterNorm    float64
	StepNorm         float64
}

func (it IterationSummary) String() string {
	k := it.Iteration
	f := it.Cost
	df := math.Abs(it.CostChange) / it.Cost
	g := it.GradientNorm / it.GradientNormInit
	dx := it.StepNorm / it.ParameterNorm
	return fmt.Sprintf("%5d  f:%13.6e  df:%10.3e  g:%10.3e  dx:%10.3e", k, f, df, g, dx)
}

func Summarize(k int, f_prev, f float64, g_init, g, x_prev, x vec.ConstTyped) IterationSummary {
	df := f_prev - f
	dx := math.Inf(1)
	if k > 0 {
		dx = vec.Distance(x, x_prev)
	}
	return IterationSummary{k, f, df, vec.InfNorm(g), vec.InfNorm(g_init), vec.Norm(x), dx}
}

type Solver interface {
	Solve(f Objective, x vec.ConstTyped, crit TerminationCriteria,
		callback Callback, verbose bool) (vec.MutableTyped, error)
}
