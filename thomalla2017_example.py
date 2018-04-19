from glob import glob
from matplotlib import pyplot as plt
import thomalla2017_quenching as qch
import pandas as pd
import numpy as np


flist = glob('./example_data/data*.csv')
print('Reading example data with pandas')
data = {}
for fname in flist:
    print('  ', fname)
    var_name = fname.split('/')[-1].split('.')[0].split('_')[1]
    df = pd.read_csv(fname, index_col=0, parse_dates=True).astype(float)
    df.columns = map(np.datetime64, df.columns.values)  # column names must be datetime64[ns]
    data[var_name] = df
data['photiclayer'] = data['photiclayer'].astype(bool)  # photic layer is a boolean mask

print('\nPrint information about data format')
# print information about the formatting of the data
for key in data:
    df = data[key]
    txt = '   {: <16} {}   '.format(key, str(df.shape))
    txt += 'index: {} [{}]  '.format(df.index.name, df.index.dtype)
    txt += 'columns: time [{}]'.format(df.columns.dtype)
    print(txt)


#####################################
#  RUN QUENCHING CORRECTION SCRIPT  #
#####################################
# Run the quenching corrections. Output is saved to the dictionary
data['fluorescence_corrected'], data['quenching_layer'], data['dives_per_night'] = qch.quenching_correction(
    data['fluorescence'],
    data['backscatter'],
    data['latitude'],
    data['longitude'],
    data['photiclayer'])

#####################
#  PLOT
#####################
x = data['fluorescence'].columns.values
y = data['fluorescence'].index.values

variables = 'photiclayer', 'backscatter', 'fluorescence', 'fluorescence_corrected', 'quenching_layer'
fig, ax = plt.subplots(len(variables), 1, figsize=[8, 11], sharex=True)
for i in range(ax.size):
    im = ax[i].pcolormesh(x, y, data[variables[i]].values.astype(float), rasterized=True)
    ax[i].set_ylim(200, 0)
    ax[i].set_ylabel('Depth (m)')
    try:
        cb = plt.colorbar(mappable=im, ax=ax[i], pad=0.02)
        cb.set_label(variables[i].capitalize())
    except TypeError:
        pass

plt.setp(ax[-1].xaxis.get_majorticklabels(), rotation=30)

fig.tight_layout()
fig.suptitle('Example for quenching correction method from Thomalla et al. (2018)', x=0.45, y=1.01)
fig.savefig('./fig_example.pdf', bbox_inches='tight')
plt.show()
