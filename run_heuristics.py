from heuristics.heuristics_intervals import Heuristics
import timeit

if __name__ == '__main__':

    #### Search
    search = True
    n_comp = 22
    freq_col = [1.e-2, 1.e-3, 5.e-4, 1.e-5, 1.e-6]
    discount_reward = 0.95
    eval_size = 2000

    #### Evaluation
    insp_int = 10
    insp_comp = 5
    pf_brace_rep = 0.01

    h1 = Heuristics(n_comp,
                     # Number of structure
                     freq_col,
                     discount_reward)

    if search:
        starting_time = timeit.default_timer()
        h1.search(eval_size)
        print("Time (s):", timeit.default_timer() - starting_time)
    else:
        h1.eval(eval_size, insp_int, insp_comp, pf_brace_rep)
