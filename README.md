# Thomalla et al. (2017) quenching correction

Quenching correction method used in Thomalla et al. (2017) with example data and script.

CITE AS:

	Thomalla, S. J., Moutier, W., Ryan-Keogh, T. J., Gregor, L., & Schütt, J. (2017).
	An optimized method for correcting fluorescence quenching using optical
	backscattering on autonomous platforms. Limnology and Oceanography: Methods,
	(Lorenzen 1966). [https://doi.org/10.1002/lom3.10234](https://doi.org/10.1002/lom3.10234)

To run this script you need to format your data into pandas DataFrames.

The index (row labels) of the dataFrames must be depth and the column lables
must be date (dtype=np.datetime64).
