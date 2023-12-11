import csv
from datetime import datetime
import sys

baseline = datetime.strptime('00:00', '%H:%M')
with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    deltas = [datetime.strptime(line[0], '%H:%M') - baseline
              for line in reader]
schedule = [d.seconds // 60 for d in deltas]
print(schedule)
