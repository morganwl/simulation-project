trials=$1
while true; do
    fbsimulate -x b35-short -n $trials
    fbsimulate -x b35-long -n $trials
    fbsimulate -x b35-short-busy -n $trials
    fbsimulate -x b35-long-busy -n $trials
    fbtabulate b35-short b35-long b35-short-busy b35-long-busy -n experiment
done
