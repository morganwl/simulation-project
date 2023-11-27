## Sat Nov 25

I want to slightly rework my simulation pipeline from previous work for
modeling. Prior work was centered around outputting an estimated mean
for a single random variable, possibly with confidence intervals. For
this go-around, I would like to center around data collection, i.e.
collecting data from trials into an array and writing the contents of
that array to disk. Analysis of trials can be done with a separate tool. 

MAIN(experiment: structure of parameters, n: integer, output: str path)
writes the results of n trials, performed with parameters from
experiment, to disk.

SIMULATE-BATCH(experiment: structure of parameters, n: integer, size of
batch) returns an ndarray of n sets of random variables, extracted from
n trials.

SIMULATE(experiment: structure of parameters) returns an array of simulated
events.

MEASURE(events: an array of simulated events) returns a set of
variables, extracted from a set of events.

Measured Variables:
- Loading time
- Moving time
- Holding time (waiting at a stop in order to stay on schedule)
- Total number of passengers

Experimental parameters:
- Routes (number of stops per route, optionally labeling stops)
- Distance(r,s) (for any given (r, s) pair, the travel distance from (r, s-1)
  to (r, s) in miles.
- Traffic(r,s,t) the traffic along the route to (r,s) at time t.
- Passenger-load-demand(r,s,t) the number of passengers boarding at
  (r,s) at time t.
- Passenger-unload-demand(r,s,t) the number of passengers unloading at
  (r,s) at time t.
- Loading-time(p) the loading time for a single passenger onto a bus
  with p passengers.
- Unloading-time(p) the unloading time for a single passenger onto a bus
  with p passengers.
