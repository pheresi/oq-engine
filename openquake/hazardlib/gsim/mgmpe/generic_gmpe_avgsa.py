# The Hazard Library
# Copyright (C) 2012-2019 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Module :mod:`openquake.hazardlib.mgmp.generic_gmpe_avgsa` implements
:class:`~openquake.hazardlib.mgmpe.GenericGmpeAvgSA`
"""

import copy
import numpy as np
from openquake.hazardlib.gsim.base import GMPE, registry
from openquake.hazardlib import const
from openquake.hazardlib.imt import SA
from openquake.hazardlib.gsim.mgmpe import akkar_coeff_table as act

class GenericGmpeAvgSA(GMPE):
    """
    Implements a modified GMPE class that can be used to compute average
    ground motion over several spectral ordinates. The mean and standard
    deviation are computed according to:
    Kohrangi M., Reddy Kotha S. and Bazzurro P., 2018, Ground-motion models
    for average spectral acceleration in a period range: direct and indirect
    methods, Bull. Earthquake. Eng., 16, pp. 45–65.
    Note that only the Total Standard Deviation is supported.

    :param string gmpe_name:
        The name of a GMPE class used for the calculation.

    :param list avg_periods:
        List of averaging periods (must be a subset of the periods allowed
        in the selected GMPE)

    :param string corr_func:
        Handle of the function to compute correlation coefficients between
        different spectral acceleration ordinates. Valid options are:
        'baker_jayaram', 'akkar', 'none'. Default is none.
    """

    # Parameters
    REQUIRES_SITES_PARAMETERS = set()
    REQUIRES_DISTANCES = set()
    REQUIRES_RUPTURE_PARAMETERS = set()
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = ''
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([SA])
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([const.StdDev.TOTAL])
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''
    DEFINED_FOR_REFERENCE_VELOCITY = None

    def __init__(self, gmpe_name, avg_periods, corr_func='none'):
        super().__init__(gmpe_name=gmpe_name)
        self.gmpe = registry[gmpe_name]()
        self.set_parameters()
        self.avg_periods = [float(t) for t in avg_periods.split(',')]
        self.tnum = len(self.avg_periods)

        correlation_function_handles = {
            'baker_jayaram': baker_jayaram_correlation,
            'akkar': akkar_correlation,
            'none': dummy_correlation
        }

        self.corr_func = correlation_function_handles[corr_func]

        # Check if this GMPE has the necessary requirements
        # [TO DO]

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stds_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """

        mean_list = []
        stddvs_list = []

        # Loop over averaging periods
        for period in self.avg_periods:
            imt_local = SA(float(period))
            # compute mean and standard deviation
            mean, stddvs = self.gmpe.get_mean_and_stddevs(sites, rup, dists,
                                                          imt_local,
                                                          stds_types)
            mean_list.append(mean)
            stddvs_list.append(stddvs[0]) # Support only for total!

        mean_avgsa = 0.
        stddvs_avgsa = 0.

        for i1 in range(self.tnum):
            mean_avgsa += mean_list[i1]
            for i2 in range(self.tnum):
                rho = self.corr_func(self.avg_periods[i1],
                                     self.avg_periods[i2])
                stddvs_avgsa += rho * stddvs_list[i1] * stddvs_list[i2]

        mean_avgsa *= (1./self.tnum)
        stddvs_avgsa = (1./self.tnum) * np.sqrt(stddvs_avgsa)

        return mean_avgsa, [stddvs_avgsa]


def baker_jayaram_correlation(t1, t2):
    """
    Produce inter-period correlation for any two spectral periods.
    Subroutine taken from: https://usgs.github.io/shakemap/shakelib
    Based upon:
    Baker, J.W. and Jayaram, N., "Correlation of spectral acceleration
    values from NGA ground motion models," Earthquake Spectra, (2007).

    :param float t1:
        First period of interest.

    :param float t2:
        Second period of interest.

    :return float rho:
        The predicted correlation coefficient.
    """

    t_min = min(t1, t2)
    t_max = max(t1, t2)

    c1 = 1.0 - np.cos(np.pi / 2.0 - np.log(t_max / max(t_min, 0.109)) * 0.366)

    if t_max < 0.2:
        c2 = 0.105 * (1.0 - 1.0 / (1.0 + np.exp(100.0 * t_max - 5.0)))
        c2 = 1.0 - c2 * (t_max - t_min) / (t_max - 0.0099)
    else:
        c2 = 0

    if t_max < 0.109:
        c3 = c2
    else:
        c3 = c1

    c4 = c1 + 0.5 * (np.sqrt(c3) - c3) * (1.0 + np.cos(np.pi * t_min / 0.109))

    if t_max <= 0.109:
        rho = c2
    elif t_min > 0.109:
        rho = c1
    elif t_max < 0.2:
        rho = min(c2, c4)
    else:
        rho = c4

    return rho


def akkar_correlation(t1, t2):
    """
    Read the period-dependent correlation coefficient matrix as in:
    Akkar S., Sandikkaya MA., Ay BO., 2014, Compatible ground-motion
    prediction equations for damping scaling factors and vertical to
    horizontal spectral amplitude ratios for the broader Europe region,
    Bull Earthquake Eng, 12, pp. 517-547.

    :param float t1:
        First period of interest.

    :param float t2:
        Second period of interest.

    :return float:
        The predicted correlation coefficient.
    """

    return act.coeff_table[act.periods.index(t1)][act.periods.index(t2)]


def dummy_correlation(t1, t2):
    """
    Dummy function returning just 1 (used as default function handle)

    :param float t1:
        First period of interest.

    :param float t2:
        Second period of interest.

    :return 1:
    """

    return 1.