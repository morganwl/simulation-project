trials=$1
fbsimulate -x b35-short -n $1
fbsimulate -x b35-long -n $1
fbsimulate -x b35-short-busy -n $1
fbsimulate -x b35-long-busy -n $1
