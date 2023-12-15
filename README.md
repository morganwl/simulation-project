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
- [ ] 0.2.0 features
  - [X] non-deterministic trials
  - [X] test bus leapfrog events
  - [X] print results as distribution
    - very rudimentary
  - [X] PERT distribution
  - [X] single line with multiple stops
    - need a servicable time function
  - [ ] scheduled departure time and intermediate stops
  - [ ] calculate and report benchmarks
    - [ ] events processed
    - [X] runtime per trial
    - [ ] memory usage per trial
    - [ ] rns used per trial
- [ ] 0.3.0 features
  - [ ] multiple lines
  - [ ] transfers
  - [ ] input parameters via YAML
