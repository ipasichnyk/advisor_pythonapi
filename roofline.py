from __future__ import print_function

import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict

# This style requires $DISPLAY available.
# Use it instead of matplotlib.use('Agg') if you have GUI environment
#matplotlib.style.use('ggplot')

pd.options.display.max_rows = 20

advisor_path='c:/Program Files (x86)/IntelSWTools/Advisor 2020/pythonapi'
sys.path.append(advisor_path)
print(sys.path)

gflops_roof_names=['Scalar Add Peak','DP Vector Add Peak','DP Vector FMA Peak']

try:
     import advisor
except ImportError:
    print('Import error: Python could not load advisor python library. Possible reasons:\n'
          '1. Python cannot resolve path to Advisor\'s pythonapi directory. '
          'To fix, either manually add path to the pythonapi directory into PYTHONPATH environment variable,'
          ' or use advixe-vars.* scripts to set up product environment variables automatically.\n'
          '2. Incompatible runtime versions used by advisor python library and other packages '
          '(such as matplotlib or pandas). To fix, either try to change import order or update other package '
          'version if possible.')
    sys.exit(1)

if len(sys.argv) < 2:
    print('Usage: "python {} path_to_project_dir"'.format(__file__))
    sys.exit(2)

project = advisor.open_project(sys.argv[1])
data = project.load(advisor.SURVEY)
#take elapsed time from the topdown stack
tot_elapsed_time=float(next(data.topdown).total_elapsed_time)
print(tot_elapsed_time)

rows = [{col: row[col] for col in row} for row in data.bottomup]
#temp_row=rows[0]
#for key in temp_row:
#    print(key, temp_row[key])

#roofs = data.get_roofs()
roofs = data.get_roofs(1, advisor.RoofsStrategy.SINGLE_THREAD)

df = pd.DataFrame(rows).replace('', np.nan)
df.self_elapsed_time=df.self_elapsed_time.astype(float)
df.self_gflop = df.self_gflop.astype(float)

#aggregate by function_call_sites_and_loops
aggregation_functions = {'function_call_sites_and_loops': 'first', 'self_elapsed_time': 'sum', 
    'self_gflop': 'sum', 'self_ai': 'first'}
df = df.groupby(['function_call_sites_and_loops','self_ai']).aggregate(aggregation_functions)

#df['weight']= df.self_elapsed_time.astype(float)/tot_elapsed_time*100
df.self_ai = df.self_ai.astype(float)
#df.self_gflops = df.self_gflops.astype(float)

#filter out NaN inplace
df.dropna(subset=['self_ai'],inplace=True)
df=df.loc[df['self_ai'] > 1.e-8]

df['weight']= df.self_elapsed_time/tot_elapsed_time*100
df['marker_size'] = df.apply(lambda row: max(30,row.weight*120.), axis = 1) 
df['self_gflops']= df.self_gflop/df.self_elapsed_time

# take only weight > 5%
df=df[df['weight']>0.5]

df=df.sort_values(by=['weight'], ascending=False)
print(df[['weight','self_ai','self_gflops']])

#print(df[['function_call_sites_and_loops','self_elapsed_time','weight','self_ai', 'self_gflop','self_gflops']].dropna())

#width = df.self_ai.max() * 1.2
width=100

fig,ax = plt.subplots()

max_compute_roof = max(roofs, key=lambda roof: roof.bandwidth if 'bandwidth' not in roof.name.lower() else 0)
max_compute_bandwidth = max_compute_roof.bandwidth // math.pow(10, 9)  # converting to GByte/s

for roof in roofs:
    # by default drawing multi threaded roofs only
    #remove '(single-threaded)'
    roof_trunc = roof.name.replace(' (single-threaded)','')
 #  if 'single-thread' not in roof.name:
    if 1==1:
        # memory roofs
        #if 'bandwidth' in roof.name.lower():
        if roof_trunc == 'DRAM Bandwidth':
            bandwidth = roof.bandwidth / math.pow(10, 9) # converting to GByte/s
            bw_label = '{} {:.0f} GB/s'.format(roof_trunc, bandwidth)
        # compute roofs
        if roof_trunc in gflops_roof_names:
            gflops = roof.bandwidth / math.pow(10, 9)  # converting to GFlOPS
            if roof_trunc=='DP Vector FMA Peak':
                gflops_dp_fma=gflops
            x1, x2 = gflops/bandwidth, width
            y1, y2 = gflops, gflops
            label = '{} {:.0f} GFLOPS'.format(roof_trunc, gflops)
            ax.annotate(label, xy=(width, y1), xytext=(-5,4), textcoords="offset points",horizontalalignment='right')
            ax.plot([x1, x2], [y1, y2], '-', label=label,color='black')


#plot BW roofline
# y = banwidth * x
#x1, x2 = 0, int(min(width, max_compute_bandwidth / bandwidth))
x1, x2 = 0, gflops_dp_fma/bandwidth
#y1, y2 = 0, int(x2 * bandwidth)
y1, y2 = 0, gflops_dp_fma
angle_data = np.rad2deg(np.arctan2(y2-y1, x2-x1))

ax.plot([x1, x2], [y1, y2], '-',color='red')

ax.tick_params(which='major',labelsize=14,length=6)
ax.tick_params(which='minor',labelsize=14,length=3) 
# drawing points using the same ax
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set(ylim=(1e-2, 5e+2), xlim=(1e-3, width))
ax.set_xlabel('Arithmetic intensity [FLOP/byte]',fontsize=14)
ax.set_ylabel('Performance [GFLOP/sec]',fontsize=14)

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')

sc=ax.scatter(df.self_ai, df.self_gflops, c=df.weight, s=df.marker_size, alpha=0.5, 
    linewidths=0.7,edgecolors='black',
    cmap='rainbow')
#sc=ax.scatter(df.self_ai, df.self_gflops, s=10)
#sc.set_clim(vmin=min(df.weight),vmax=max(df.weight))
plt.colorbar(sc,label='Self elapsed time [%]')

plt.tight_layout()

#bw label - order is important, i.e. AFTER applying xlimit, ylimit
bw_label_loc = np.array((0.001, bandwidth*0.001))
angle_screen = ax.transData.transform_angles(np.array((angle_data,)), 
                                 bw_label_loc.reshape((1, 2)))[0]
# using `annotate` allows to specify an offset in units of points
ax.annotate(bw_label, xy=(bw_label_loc[0], bw_label_loc[1]), xytext=(10,20), textcoords="offset points", 
                rotation_mode='anchor', rotation=angle_screen)

#plt.legend(loc='lower right', fancybox=True, prop={'size': 6})

# saving the chart as PNG image
plt.savefig('roofline.png')
# saving the chart in SVG vector format
plt.savefig('roofline.svg')

print('Roofline chart has been generated and saved into roofline.png and roofline.svg files in the current directory')
