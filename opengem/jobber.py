# -*- coding: utf-8 -*-
"""
Main jobber module.
"""

import math

from opengem import config
from opengem import flags
from opengem import hazard
from opengem import logs
from opengem import kvs
from opengem import producer
from opengem import risk
from opengem import shapes
from opengem import settings

from opengem.risk import tasks

from opengem.output.risk import RiskXMLWriter

from opengem.parser import exposure
from opengem.parser import hazard
from opengem.parser import vulnerability

FLAGS = flags.FLAGS
LOGGER = logs.LOG
SITES_PER_BLOCK = 100

# TODO (ac): This class is not covered by unit tests...
class Jobber(object):
    """The Jobber class is responsible to evaluate the configuration settings
    and to execute the computations in parallel tasks (using the celery
    framework and the message queue RabbitMQ).
    """

    def __init__(self, job, partition):
        self.memcache_client = None
        self.partition = partition
        self.job = job
        
        self._init()

    def run(self):
        """Core method of Jobber. It splits the requested computation
        in blocks and executes these as parallel tasks.
        """

        LOGGER.debug("running jobber, job_id = %s" % self.job.id)

        for block_id in self._partition():
            self._preload(self.job.id, block_id)
            self._execute(self.job.id, block_id)
            self._write_output_for_block(self.job.id, block_id)

        LOGGER.debug("Jobber run ended")

    def _partition(self):
        """Split the set of sites to compute in blocks and store
        the in the underlying kvs system.
        """

        sites = []
        blocks_keys = []
        region_constraint = self._read_region_constraint()
        
        # we use the exposure, if specified,
        # otherwise we use the input region
        if self.job.has(config.EXPOSURE):
            sites = self._read_sites_from_exposure()
        else:
            sites = shapes.Region.from_file(
                    self.job[config.INPUT_REGION]).sites

        if self.partition:
            for block in BlockSplitter(sites, constraint=region_constraint):
                blocks_keys.append(block.id)
                block.to_kvs()
        else:
            block = Block(sites)
            blocks_keys.append(block.id)
            block.to_kvs()
        
        return blocks_keys

    def _read_region_constraint(self):
        """Read the region constraint, if present, from the job definition."""

        if self.job.has(config.INPUT_REGION):
            return shapes.RegionConstraint.from_file(
                    self.job[config.INPUT_REGION])
        else:
            return None

    def _read_sites_from_exposure(self):
        """Read the set of sites to compute from the exposure file specified
        in the job definition."""

        sites = []
        reader = exposure.ExposurePortfolioFile(self.job[config.EXPOSURE])
        
        for asset_data in reader:
            sites.append(asset_data[0])

        return sites

    def _execute(self, job_id, block_id):
        """ Execute celery task for risk given block with sites """
        
        LOGGER.debug("starting task block, block_id = %s" % block_id)

        # task compute_risk has return value 'True' (writes its results to
        # memcache).
        result = tasks.compute_risk.apply_async(args=[job_id, block_id])

        # TODO(fab): Wait until result has been computed. This has to be
        # changed if we run more tasks in parallel.
        result.get()

    def _write_output_for_block(self, job_id, block_id):
        """note: this is usable only for one block"""
        
        # produce output for one block
        loss_curves = []

        sites = kvs.get_sites_from_memcache(self.memcache_client, job_id, 
            block_id)

        for (gridpoint, (site_lon, site_lat)) in sites:
            key = kvs.generate_product_key(job_id, 
                risk.LOSS_CURVE_KEY_TOKEN, block_id, gridpoint)
            loss_curve = self.memcache_client.get(key)
            loss_curves.append((shapes.Site(site_lon, site_lat), 
                                loss_curve))

        LOGGER.debug("serializing loss_curves")
        output_generator = RiskXMLWriter(settings.LOSS_CURVES_OUTPUT_FILE)
        output_generator.serialize(loss_curves)
        
        #output_generator = output.SimpleOutput()
        #output_generator.serialize(ratio_results)
        
        #output_generator = geotiff.GeoTiffFile(output_file, 
        #    region_constraint.grid)
        #output_generator.serialize(losses_one_perc)

    def _init(self):
        """ Initialize memcached_client. This should move into a Singleton """
        
        # TODO(fab): find out why this works only with binary=False
        self.memcache_client = kvs.get_client(binary=False)
        self.memcache_client.flush_all()

    def _preload(self, job_id, block_id):
        """ preload configuration for job """

        # set region
        region_constraint = shapes.RegionConstraint.from_file(
                self.job[config.INPUT_REGION])

        # TODO(fab): the cell size has to be determined from the configuration 
        region_constraint.cell_size = 1.0

        # load hazard curve file and write to memcache_client
        nrml_parser = hazard.NrmlFile(self.job[config.HAZARD_CURVES])
        attribute_constraint = \
            producer.AttributeConstraint({'IMT' : 'MMI'})

        sites_hash_list = []

        for site, hazard_curve_data in nrml_parser.filter(
                region_constraint, attribute_constraint):

            gridpoint = region_constraint.grid.point_at(site)

            # store site hashes in memcache
            # TODO(fab): separate this from hazard curves. Regions of interest
            # should not be taken from hazard curve input, should be 
            # idependent from the inputs (hazard, exposure)
            sites_hash_list.append((str(gridpoint), 
                                   (site.longitude, site.latitude)))

            hazard_curve = shapes.Curve(zip(hazard_curve_data['IML'], 
                                                hazard_curve_data['Values']))

            memcache_key_hazard = kvs.generate_product_key(job_id, 
                hazard.HAZARD_CURVE_KEY_TOKEN, block_id, gridpoint)

            LOGGER.debug("Loading hazard curve %s at %s, %s" % (
                        hazard_curve, site.latitude,  site.longitude))

            success = self.memcache_client.set(memcache_key_hazard, 
                hazard_curve.to_json())

            if success is not True:
                raise ValueError(
                    "jobber: cannot write hazard curve to memcache")

        # write site hashes to memcache (JSON)
        memcache_key_sites = kvs.generate_sites_key(job_id, block_id)

        success = kvs.set_value_json_encoded(self.memcache_client, 
                memcache_key_sites, sites_hash_list)
        if not success:
            raise ValueError(
                "jobber: cannot write sites to memcache")
        
        # load assets and write to memcache
        exposure_parser = exposure.ExposurePortfolioFile(self.exposure_file)
        for site, asset in exposure_parser.filter(region_constraint):
            gridpoint = region_constraint.grid.point_at(site)

            memcache_key_asset = kvs.generate_product_key(
                job_id, risk.EXPOSURE_KEY_TOKEN, block_id, gridpoint)

            LOGGER.debug("Loading asset %s at %s, %s" % (asset,
                site.longitude,  site.latitude))

            success = kvs.set_value_json_encoded(self.memcache_client, 
                memcache_key_asset, asset)
            if not success:
                raise ValueError(
                    "jobber: cannot write asset to memcache")

        # load vulnerability and write to memcache
        vulnerability.load_vulnerability_model(job_id,
            self.vulnerability_model_file)


class Block(object):
    """A block is a collection of sites to compute."""

    def __init__(self, sites):
        self.sites = tuple(sites)
        self.block_id = kvs.generate_block_id()

    def __eq__(self, other):
        return self.sites == other.sites

    @classmethod
    def from_kvs(cls, block_id):
        """Return the block in the underlying kvs system with the given id."""

        raw_sites = kvs.get_value_json_decoded(block_id)

        sites = []

        for raw_site in raw_sites:
            sites.append(shapes.Site(raw_site[0], raw_site[1]))

        return Block(sites)

    def to_kvs(self):
        """Store this block into the underlying kvs system."""

        raw_sites = []

        for site in self.sites:
            raw_sites.append(site.coords)

        kvs.set_value_json_encoded(self.id, raw_sites)

    @property
    def id(self):
        """Return the id of this block."""
        return self.block_id


class BlockSplitter(object):
    """Split the sites into a set of blocks."""

    def __init__(self, sites, sites_per_block=SITES_PER_BLOCK, constraint=None):
        self.sites = sites
        self.constraint = constraint
        self.sites_per_block = sites_per_block
    
        if not self.constraint:
            class AlwaysTrueConstraint():
                def match(self, point):
                    return True
            
            self.constraint = AlwaysTrueConstraint()
    
    def __iter__(self):
        if not len(self.sites):
            return

        number_of_blocks = int(math.ceil(len(self.sites) /
                float(self.sites_per_block)))

        for idx in range(number_of_blocks):
            filtered_sites = []
            offset = idx * self.sites_per_block
            sites = self.sites[offset:offset + self.sites_per_block]

            # TODO (ac): Can be done better using shapely.intersects,
            # but after the shapes.Site refactoring...
            for site in sites:
                if self.constraint.match(site):
                    filtered_sites.append(site)
                
            yield(Block(filtered_sites))
