#! /usr/bin/env python
import sys
sys.path.insert(0,sys.path[0]+'/source')
sys.path.insert(0,sys.path[0]+'/model')
sys.path.insert(0,sys.path[3]+'/makePlots')
sys.path.insert(0,sys.path[4]+'/libraries')
sys.path.insert(0,sys.path[5]+'/readData')
sys.path.insert(0,sys.path[6]+'/crossValidation')
sys.path.insert(0,sys.path[7]+'/analysis_tool')


#import matplotlib
#matplotlib.use('Agg')

from mainPipeline import mainPipeline
#from visualization import mainPipeline

mainPipeline()


