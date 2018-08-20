# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2017 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import FIV3
from openquake.hazardlib import imt as imt_module

class DavalosMiranda2018FIV3(GMPE):
    """
    Implements GMPE developed by Hector Davalos and Eduardo Miranda
    Coded by Pablo Heresi
    """
    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure type is spectral inelastic displacement
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        FIV3
    ])

    #: Supported intensity measure component is random horizontal
    #: :attr:`~openquake.hazardlib.const.IMC.RANDOM_HORIZONTAL`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RANDOM_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: No sites parameters are required
    REQUIRES_SITES_PARAMETERS = set()

    #: Required rupture parameters are magnitude
    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    #: Required distance measure is Rjb
    REQUIRES_DISTANCES = set(('rjb',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        
        #: imt_aux is for searching inside the coefficient table
        imt_aux = imt_module.SA(damping=5, period=imt.period)
        CoeffsTable = self.Coeffs
        C = CoeffsTable[imt_aux]

        mean = (self._get_magnitude_scaling_term(C, rup.mag) +
                self._get_path_scaling(C, dists, rup.mag))
        stddevs = self._get_stddevs(C, stddev_types, dists)
        return mean, stddevs

    def _get_magnitude_scaling_term(self, C, mag):
        """
        Returns the magnitude scaling term
        """
        dmag = mag - C['Mh']
        if mag < C['Mh']:
            mag_term = C['e2'] * dmag + C['e3'] * dmag ** 2
        else:
            mag_term = C['e4'] * dmag
        return C['e1'] + mag_term

    def _get_path_scaling(self, C, dists, mag):
        """
        Returns the path scaling term
        """
        R = np.sqrt((dists.rjb ** 2.0) + (C['h'] ** 2.0))
        scaling = (C['c1'] + C['c2'] * mag) * \
            np.log(R / self.CONSTS['Rref'])
        return scaling

    def _get_stddevs(self, C, stddev_types, dists):
        """
        Returns the aleatory uncertainty terms
        """
        stddevs = []
        tau = C['tau']
        phi = C['phi']
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(np.sqrt((tau ** 2.0)+(phi ** 2.0)) + 
                                                             dists.rjb * 0)
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(phi + dists.rjb * 0)
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(tau + dists.rjb * 0)
        return stddevs

    CONSTS = {"Rref": 1.0}

    Coeffs = CoeffsTable(sa_damping=5, table='''
imt e1  e2  e3  e4  Mh  c1  c2  h   phi   tau
0.1 4.7258  1.0319  -0.0986 -0.0640 6.3 -1.5689 0.1197  3.9785  0.41    0.34
0.2 5.2930  1.1064  -0.0579 -0.0656 6.3 -1.5618 0.1198  3.8224  0.41    0.34
0.3 5.6266  1.1948  -0.0224 -0.0553 6.3 -1.5363 0.1168  3.7031  0.42    0.35
0.4 5.7926  1.2394  -0.0283 -0.0338 6.3 -1.5003 0.1121  3.6084  0.42    0.35
0.5 5.9304  1.3677  0.0319  -0.0100 6.3 -1.4608 0.1067  3.5262  0.43    0.35
0.6 5.9960  1.4705  0.0830  0.0303  6.3 -1.3945 0.0975  3.4334  0.44    0.36
0.7 6.0333  1.5777  0.1460  0.0779  6.3 -1.3228 0.0873  3.4014  0.44    0.37
0.8 6.0528  1.7006  0.2382  0.1197  6.3 -1.2691 0.0795  3.3985  0.45    0.37
0.9 6.0435  1.8357  0.3006  0.1866  6.3 -1.1851 0.0676  3.3492  0.45    0.37
1.0 6.0461  1.9148  0.3322  0.2317  6.3 -1.1421 0.0611  3.3656  0.46    0.38
1.1 6.0464  2.0053  0.3814  0.2804  6.3 -1.0972 0.0541  3.3816  0.46    0.38
1.2 6.0488  2.0165  0.3553  0.3320  6.3 -1.0607 0.0481  3.4510  0.46    0.38
1.3 6.0425  2.0108  0.3426  0.3731  6.2 -1.0300 0.0430  3.4639  0.46    0.39
1.4 6.0208  1.9982  0.3043  0.4208  6.2 -0.9868 0.0366  3.5088  0.47    0.39
1.5 5.9913  2.0440  0.3233  0.4692  6.2 -0.9479 0.0306  3.5468  0.47    0.39
1.6 5.9719  2.0255  0.2893  0.5018  6.2 -0.9355 0.0285  3.6266  0.47    0.38
1.7 5.9446  2.0223  0.2603  0.5523  6.2 -0.8846 0.0207  3.7155  0.47    0.38
1.8 5.9374  2.0002  0.2395  0.5710  6.2 -0.8742 0.0191  3.7306  0.47    0.39
1.9 5.9393  2.0042  0.2265  0.5874  6.2 -0.8663 0.0175  3.7797  0.47    0.38
2.0 5.9432  2.0010  0.2217  0.5969  6.2 -0.8887 0.0204  3.8395  0.47    0.38
2.1 5.9771  2.1049  0.3138  0.5672  6.2 -0.9305 0.0262  3.8854  0.47    0.38
2.2 5.9591  2.0340  0.2466  0.5935  6.2 -0.9323 0.0263  3.9044  0.47    0.38
2.3 5.9996  2.1710  0.3657  0.5575  6.2 -0.9446 0.0280  3.8999  0.47    0.37
2.4 5.9824  2.1495  0.3494  0.5709  6.2 -0.9304 0.0262  3.8455  0.47    0.37
2.5 5.9917  2.1708  0.3558  0.5634  6.2 -0.9327 0.0268  3.8493  0.47    0.37
2.6 5.9886  2.1637  0.3471  0.5679  6.2 -0.9277 0.0261  3.9329  0.47    0.37
2.7 5.9716  2.1641  0.3471  0.5844  6.2 -0.9043 0.0230  3.9282  0.47    0.37
2.8 5.9549  2.1927  0.3717  0.6065  6.2 -0.8755 0.0190  3.9297  0.47    0.37
2.9 5.9358  2.2145  0.3961  0.6238  6.2 -0.8632 0.0172  3.9647  0.47    0.34
3.0 5.9399  2.2777  0.4598  0.6107  6.2 -0.8644 0.0176  3.9806  0.47    0.37
        ''')


    


