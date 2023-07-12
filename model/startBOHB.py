#!/usr/bin/env python

# Script to start BOHB training process.
# Auto-handles all workers, processes. Ideally, only the worker class
# definition (in MyWorker.py) should be necessary to modify to adapt
# to different problems at hand.

import logging
import os
import time

import hpbandster.core.nameserver as hpns
from threading import Thread
from hpbandster.optimizers import BOHB as BOHB
from MyWorker import MyWorker


# Increase logging so that interactions between BOHB server
# and workers is visible.
logging.basicConfig(level=logging.INFO)

# This uniquely identifies a run of any HpBandSter optimizer.
run_id = "hp_BOHB"


def start_BOHB(min_budget, max_budget, Xfile, Yfile=None, n_workers=1, n_iterations=10):
    """
    Perform BOHB on the local system to find hyper-parameters

    Parameters
    ----------
    min_budget : Minimum computing resources to be used, in seconds.
        At the early iterations, the models will be trained only this much time.

    max_budget : Maximum computing resources to be used, in seconds.
        Limits how much the well-performing models will get towards the end.

    Xfile : Training data, array-like of shape (n_features, n_samples).
        First column and rows are assumed to be labels and ignored.

    Yfile : Training labels, array-like of shape (n_samples, 1).
        First column should contain sample identifiers and match the header column
        of Xfile.

    n_workers : Number of parallel processes to start for BOHB. For best performance,
        should be set to number of cores. Defaults to 1.

    n_iterations : Number of iterations for BOHB, the higher the better but would
        increase resources consumed. Defaults to 10.

    Returns
    -------
    best_configuration : A dict of hyperparams that give the best performance.
    """

    # Start a nameserver
    # Every run needs a nameserver. It could be a 'static' server with a
    # permanent address, but here it will be started for the local machine with the default port.
    # The nameserver manages the concurrent running workers across all possible threads or clusternodes.
    NS = hpns.NameServer(run_id=run_id, host='127.0.0.1', port=None)
    NS.start()

    # Parameter validation
    if min_budget <= 0 or max_budget < min_budget:
        raise ValueError("0 < Min budget < max budgets violated.")

    if n_iterations <= 0:
        raise ValueError("Number of iterations cannot be negative")

    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.
    if n_workers <= 0:
        raise ValueError("Number of workers (%d) invalid." % n_workers)
    elif n_workers > 1:
        worker_threads = []
        for wid in range(n_workers):
            # Time buffer for nameserver to properly start and bind.
            time.sleep(1)

            print('Starting worker wid=%d' % wid)
            t = Thread(target=os.system, daemon=False,
                       args=['python startBOHB.py --worker --input_X %s --input_Y %s' % (Xfile, Yfile)])
            t.start()
            worker_threads.append(t)

        # Check if all workers, i.e. threads are functional
        for t in worker_threads:
            if not t.is_alive():
                raise Exception("Worker(s) failed to initialise")
    else:
        # 1 worker only, i.e. single threaded via current shell
        print('Starting the worker...')
        w = MyWorker(Xfile=Xfile, Yfile=Yfile, nameserver='127.0.0.1', run_id=run_id)
        w.run(background=True)

    # Run an optimizer BOHB
    # Now we can create an optimizer object and start the run.
    # The run method will return the `Result` that contains all runs performed.
    bohb = BOHB(configspace=MyWorker.get_configspace(), run_id=run_id, min_budget=min_budget, max_budget=max_budget)
    res = bohb.run(n_iterations=n_iterations, min_n_workers=n_workers)

    # Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    all_runs = res.get_all_runs()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations were sampled.' % len(id2config.keys()))
    print('A total of %i runs were executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (sum([r.budget for r in all_runs]) / max_budget))
    print('The run took  %.1f seconds to complete.' % (all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

    best_configuration = id2config[incumbent]['config']
    return best_configuration


if __name__ == '__main__':
    import argparse

    # Parse input arguments.
    parser = argparse.ArgumentParser(description='BOHB - Local or parallel execution')
    parser.add_argument('--input_X', type=str, help='Data table to be used for the training.')
    parser.add_argument('--input_Y', type=str, help='Data labels to be used for the training.', required=False)
    parser.add_argument('--min_budget', type=int, help='Minimum budget (wall time, s) used during the optimization.', default=30)
    parser.add_argument('--max_budget', type=int, help='Maximum budget (wall time, s) used during the optimization.', default=10 * 60)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=10)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=1)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    args = parser.parse_args()

    # In case n_workers>1 and starting a brand new worker only...
    # Workers need to be separately initialised by calling "this_script.py --worker"
    # They will hang in the run state till the end.
    if args.worker:
        print('Entering worker mode')
        w = MyWorker(Xfile=args.input_X, Yfile=args.input_Y, nameserver='127.0.0.1', run_id=run_id)
        w.run(background=False)
        exit(0)
    else:
        start_BOHB(min_budget=args.min_budget, Xfile=args.input_X, Yfile=args.input_Y, max_budget=args.max_budget,
                   n_workers=1, n_iterations=10)
