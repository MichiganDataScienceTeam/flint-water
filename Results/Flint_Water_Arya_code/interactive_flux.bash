#!/bin/bash
qsub \
-I \
-N interactive_job \
-m abe \
-A mdatascienceteam_flux \
-q flux \
-l qos=preempt,procs=2,pmem=8gb,walltime=01:00:00:00 \
-j oe \
-V \

