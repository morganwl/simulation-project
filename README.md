# Freebus

Freebus performs simulations of bus performance under varying passenger
boarding times.

The program was submitted as a final project for a Masters course in
Stochastic Modeling and Simulation at Hunter College.

## Features

- Passenger demand varies over the course of a day according to one or
  more arrival distributions.
- Traffic varies over the course of the day.
- Includes support for multiple bus-lines, with transfers.
- Saves results to disk and generates plots for a number of
  measurements.
- Reports confidence intervals based on recorded data.
- Fully configurable experimental parameters with Python configuration
  file.

## Requirements

- Python3, tested on 3.10.12

## Installation

### Automated

A provided setup script will create an isolated virtual Python
environment and install the freebus, with all of its dependencies,
within this environment. Requires a MacOS or Unix-like environment.

In a terminal application of your choice:

```
git clone https://github.com/morganwl/simulation-project
cd simulation-project
./setup.sh
```

### Manual

The package can be built by hand and installed into an environment of
your choosing. In a terminal application of your choice:
```
git clone https://github.com/morganwl/simulation-project
cd simulation-project
python3 -m build
```

This will generate a package file in `dist`.

## Usage

If you have installed `freebus` into a virtual environment, you will
want to *activate* that environment.
```
cd simulation-project
source venv/bin/activate
```

You can now run the command `fbsimulate` to perform a series of trials:
```
fbsimulate -x brooklyn -n 5
```
This will run 5 trials of the *brooklyn* experiment, saving the results
in the `results` directory. Results will continue to be appended to the
same csv file for all future invocations of `fbsimulate`, *unless you
change the builtin experimental parameters*.

Once you have generated a satisfactory amount of results, you can
generate plots using the `fbplot` command:
```
fbplot brooklyn --name brooklyn
```
This will look for the latest version of the *brooklyn* experiment, and
generate figures labaled `brooklyn_<figure name>.png`.

A third script, `fbseries` will run trials with a range of passenger
boarding time distributions.

## Assumptions

Our overall model makes the following assumptions and simplifications:
- Passengers arrive irrespective of bus schedule
- Passenger unloading time does not contribute to that passenger's
  overall travel time.
- Traffic is independent of bus behavior.
- Passengers will wait indefinitely until a bus arrives.

## To-Do

- [X] create basic deployment workflow
  - [X] python project
  - [X] Docker container
  - [X] deployment script
- [ ] paper outline of simulation model
  - [X] what am I measuring
  - [X] what are the fixed parameters
  - [X] what are the random parameters
  - [ ] how are the random parameters distributed
- [ ] 0.1.0 features
  - [X] two-stop, fixed parameter bus system
  - [X] generate csv document for each command-line invocation
    containing results for all trials
    - [X] test
    - [X] feature
  - [X] calculate and report variance
  - [X] calculate and report 95% confidence intervals
    - [X] test
    - [X] feature
  - [X] report simulation parameters
    - [X] test
    - [X] feature
- [X] 0.2.0 features
  - [X] non-deterministic trials
  - [X] test bus leapfrog events
  - [X] print results as distribution
    - very rudimentary
  - [X] PERT distribution
  - [X] single line with multiple stops
    - need a servicable time function
  - [X] calculate and report benchmarks
    - [X] events processed
    - [X] runtime per trial
    - [X] memory usage per trial (via memray)
    - [-] rns used per trial
- [X] 0.3.0 features
  - [X] multiple lines
    - [ ] lines should have separate queues
  - [X] transfers
    - [ ] in large system
    - [ ] properly switch between queues to handle transfers
  - [X] random daily scaling of traffic and passenger rates
- [X] 0.4.0 improved reporting
  - [X] fix plotting of more than two experiments
  - [X] find way to demonstrate cascading effect
  - [X] generate plot of the PERT distribution
  - [-] collect events from each trial directly (compressed?)
  - [X] plot traffic
  - [X] automatically detect current experiment hash and appropriate
  - [X] wide range of Pert distribution samples
    results file
- [X] 0.5.0 submission-ready
  - [X] fix passenger inter-arrival times
      - [ ] include lamb in recorded data
      - [ ] tweak minimum between shorter and longer
      - [ ] think about the area of middle 50%?
  - [X] one more bus line with transfers
  - [X] rework traffic decay model
  - [X] generate schedule from parameters
  - [X] non-uniform rate for each stop
  - [X] fix unloading rate percentage calculations
  - [-] plot bus line variances
  - [-] scheduled departure time and intermediate stops
