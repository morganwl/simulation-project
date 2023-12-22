# Modeling and Simulation Final Project

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
- [-] 0.2.0 features
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
- [ ] 0.3.0 features
  - [X] multiple lines
    - [ ] lines should have separate queues
  - [X] transfers
    - [ ] in large system
    - [ ] properly switch between queues to handle transfers
  - [X] random daily scaling of traffic and passenger rates
- [ ] 0.4.0 improved reporting
  - [X] fix plotting of more than two experiments
  - [X] find way to demonstrate cascading effect
  - [X] generate plot of the PERT distribution
  - [ ] collect events from each trial directly (compressed?)
  - [X] plot traffic
  - [X] automatically detect current experiment hash and appropriate
  - [X] wide range of Pert distribution samples
    results file
- [ ] 0.5.0 submission-ready
  - [X] fix passenger inter-arrival times
      - [ ] include lamb in recorded data
      - [ ] tweak minimum between shorter and longer
      - [ ] think about the area of middle 50%?
  - [X] one more bus line with transfers
  - [ ] rework traffic decay model
  - [ ] generate schedule from parameters
  - [ ] non-uniform rate for each stop
  - [ ] fix unloading rate percentage calculations
  - [ ] plot bus line variances
  - [ ] scheduled departure time and intermediate stops
