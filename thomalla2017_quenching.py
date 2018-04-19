__doc__ = """
This is the code for the quenching correction used in Thomalla et al. (2017).

CITE AS:
Thomalla, S. J., Moutier, W., Ryan-Keogh, T. J., Gregor, L., & Schütt, J. (2017).
An optimized method for correcting fluorescence quenching using optical
backscattering on autonomous platforms. Limnology and Oceanography: Methods,
(Lorenzen 1966). https://doi.org/10.1002/lom3.10234

To run this script you need to format your data into pandas DataFrames.
The index (row labels) of the dataFrames must be depth and the column lables
must be date (dtype=np.datetime64).

REQUIRES:
    - numpy (np)
    - pandas (pd)
    - astral

For enquiries about the code contact:
Luke Gregor (luke.gregor@uct.ac.za)
"""


def quenching_correction(fluorescence, backscatter, lat, lon, photic_layer,
                         night_day_group=True,
                         surface_layer=7,
                         rolling_window=3):
    import numpy as np
    import pandas as pd
    import astral

    """
    Calculates the quenching depth and performs the quenching correction
    based on backscatter. The compulsory inputs must all be

    INPUT:
        fluorescence - pandas.DataFrame(index=depth, columns=surface_time, dtype=float) despiked
        backscatter  - pandas.DataFrame(index=depth, columns=surface_time, dtype=float) despiked
        lat          - pandas.DataFrame(index=depth, columns=surface_time, dtype=float)
        lon          - pandas.DataFrame(index=depth, columns=surface_time, dtype=float)
        photic_layer - pandas.DataFrame(index=depth, columns=surface_time, dtype=bool)
                       1% surface PAR True/False mask.
        night_day_group - True: quenching corrected with preceding night
                          False: quenching corrected with following night
        rolling_window  - We use a rolling window to find the quenching depth
                          the data may be spikey, this smooths it a little.
        surface_layer   - The surface fluorescence data may be quite noisy/spikey
                          hence we leave out the surface layer (meters)

    OUTPUT:
        corrected fluorescence
        quenching layer - boolean mask of quenching depth
        number of profiles per night used to correct quenching.

    METHOD:
        Correct for difference between night and daytime fluorescence.

        QUENCHING DEPTH
        ===============
        The default setting is for the preceding night to be used to
        correct the following day's quenching (`night_day_group=True`).
        This can be changed so that the following night is used to
        correct the preceding day. The quenching depth is then found
        from the differnece between the night and daytime fluorescence.
        We use the steepest gradient of the {5 minimum differences and
        the points the differnece changes sign (+ve/-ve)}.

        BACKSCATTER / CHLOROPHYLL RATIO
        ===============================
        1. Get the ratio between quenching depth and fluorescence
        2. Find the mean nighttime ratio for each night
        3. Get the ratio between nighttime and daytime quenching
        4. Apply the day/night ratio to the fluorescence
        5. If the corrected value is less than raw return to raw
    """

    def sunset_sunrise(time, lat, lon):
        """
        Uses the Astral package to find sunset and sunrise times.
        The times are returned rather than day or night indicies.
        More flexible for quenching corrections.
        """

        ast = astral.Astral()

        df = pd.DataFrame.from_items([
            ('time', time),
            ('lat', lat),
            ('lon', lon),
        ])
        # set days as index
        df = df.set_index(df.time.values.astype('datetime64[D]'))

        # groupby days and find sunrise for unique days
        grp = df.groupby(df.index).mean()
        date = grp.index.to_pydatetime()

        grp['sunrise'] = list(map(ast.sunrise_utc, date, df.lat, df.lon))
        grp['sunset'] = list(map(ast.sunset_utc, date, df.lat, df.lon))

        # reindex days to original dataframe as night
        df[['sunrise', 'sunset']] = grp[['sunrise', 'sunset']].reindex(df.index)

        # set time as index again
        df = df.set_index('time', drop=False)
        cols = ['time', 'sunset', 'sunrise']
        return df[cols]

    def quench_nmin_grad(diff_ser, window, surface_layer):
        """
        Quenching depth for a day/night fluorescence difference

        INPUT:   pandas.Series indexed by depth
                 window [4] is a rolling window size to remove spikes
                 skip_n_meters [5] skips the top layer that is often 0
        OUPUT:   estimated quenching depth as a float or int
                 note that this can be NaN if no fluorescence measurement
                 OR if the average difference is less than 0
        """

        # When the average is NaN or less than 0 don't give a depth
        # Average difference of < 0 is an artefact
        if not (diff_ser.loc[:surface_layer].mean() > 0):
            return np.NaN

        # The rolling window removes spikes creating fake shallow QD
        x = diff_ser.rolling(window, center=True).mean()
        # We also skip the first N meters as fluorescence is often 0
        x_abs = x.loc[surface_layer:].abs()

        # 5 smallest absolute differences included
        x_small = x_abs.nsmallest(5)
        # Points that cross the 0 difference and make nans False
        sign_change = (x.loc[surface_layer:] > 0).astype(int).diff(1).abs()
        sign_change.iloc[[0, -1]] = False
        x_sign_change = x_abs[sign_change]
        # subset of x to run gradient on
        x_subs = pd.concat([x_small, x_sign_change])

        # find the steepest gradient from largest difference
        x_ref = x_subs - x.loc[:surface_layer].max()
        x_ref = x_ref[x_ref.notnull()]
        x_grad = (x_ref / x_ref.index.values).astype(float)
        # index of the largest negative gradient
        x_grad_min = x_grad.idxmin()

        return x_grad_min

    # ######################## #
    # DAY / NIGHT TIME BATCHES #
    # ######################## #
    # get the coordinates of the top 20 meters of the dives (surface)
    surf_lat = lat.loc[:20].mean()
    surf_lon = lon.loc[:20].mean()
    surf_time = fluorescence.columns.values

    # get the sunrise sunset times
    sun = sunset_sunrise(surf_time, surf_lat, surf_lon).astype('datetime64[ns]')
    # calculate day night times
    day = (sun.time > sun.sunrise) & (sun.time < sun.sunset)

    # creating quenching correction batches, where a batch is a
    # night and the following day
    if type(night_day_group) is not bool:
        raise TypeError("`night_day_group` must be boolean.")
    batch = (day.astype(int).diff().abs().cumsum() + night_day_group) // 2
    batch[0] = 0

    # Group the fluorescence by daytime and quenching batch
    grouped = fluorescence.groupby([day.values, batch.values], axis=1)
    fluo_night_median = grouped.median()[False]  # get the night values
    dives_per_night = grouped.count()[False].iloc[0]

    # Calculate the nighttime fluorescence and extrapolate this to day
    # so that the difference between night and day can be calculated
    fluo_night = fluo_night_median.reindex(columns=batch.values)
    fluo_night.columns = fluorescence.columns

    # #################################################### #
    # QUENCHING DEPTH BASED ON GRADIENT OF MIN DIFFERENCES #
    # #################################################### #
    # find the depth at which mean-nighttime and daytime fluorescence cross
    diff = (fluo_night - fluorescence).where(photic_layer)
    quench_depth = diff.apply(quench_nmin_grad, args=(rolling_window, surface_layer))

    quench_layer = diff.copy()
    idx = quench_layer.index.values
    quench_layer = quench_layer.apply(lambda s: idx < quench_depth[s.name])

    # #################################################################### #
    # QUENCHING CORRECTION FROM BACKSCATTER / FLUORESCENCE NIGHTTIME RATIO #
    # #################################################################### #
    # find mean fluorescence to backscatter ratio for nigttime
    flr_bb_ratio = backscatter / fluorescence
    flr_bb_grps = flr_bb_ratio.groupby([day.values, batch.values], axis=1)
    flr_bb_night = flr_bb_grps.mean()[False]
    flr_bb_night = flr_bb_night.reindex(columns=batch.values)
    flr_bb_night.columns = fluorescence.columns

    # quenching ratio for nighttime
    quench_ratio = (flr_bb_night * fluorescence / backscatter)
    quench_ratio = quench_ratio.where(quench_layer)

    # apply the quenching ratio to the fluorescence
    quench_correction = fluorescence / quench_ratio
    mask = quench_correction.notnull().values
    fluorescence_corrected = fluorescence.copy()
    fluorescence_corrected.values[mask] = quench_correction.values[mask]

    # if corrected fluorescence is lower than raw, return to raw
    mask = (fluorescence_corrected < fluorescence).values
    fluorescence_corrected.values[mask] = fluorescence.values[mask]

    return fluorescence_corrected, quench_layer, dives_per_night


if __name__ == '__main__':
    print(__doc__)
