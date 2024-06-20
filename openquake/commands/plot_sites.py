# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2024 GEM Foundation
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
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

from openquake.baselib import hdf5, sap
from openquake.hazardlib.geo.utils import cross_idl
from openquake.calculators.postproc.plots import add_borders


def main(files_csv):
    """
    Plot the sites contained in the file
    """

    # NB: matplotlib is imported inside since it is a costly import
    import matplotlib.pyplot as p

    csvfiles = hdf5.sniff(files_csv)
    dfs = [csvfile.read_df()[['lon', 'lat']]
           for csvfile in csvfiles]
    
    fig = p.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    markersize = 5
    for csvfile, df in zip(csvfiles, dfs):
        lons, lats = df.lon.to_numpy(), df.lat.to_numpy()
        if len(lons) > 1 and cross_idl(*lons):
            lons %= 360
        p.scatter(lons, lats, marker='o',
                  label=csvfile.fname, s=markersize)
    ax = add_borders(ax)
    p.show()
    return p


main.files_csv = dict(help='a path to a CSV file with lon, lat fields',
                      nargs='+')

if __name__ == '__main__':
    sap.run(main)
