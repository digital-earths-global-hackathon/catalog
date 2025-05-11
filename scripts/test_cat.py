#!/usr/bin/env python

import intake
import intake_tools
import sys
import logging
import warnings
import numpy as np

logger = logging.getLogger("test_cat")

logging.basicConfig()
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning) 

def log_pos(cat, child, position):
    logger.info(f" {'.'.join(position)}.{child}")


def test_entry(cat, child, position):
    for params in intake_tools.iterate_user_parameters(cat[child]):
        logger.info(f" testing {child} with {params}")
        ds = cat[child](**params).to_dask()
        good = []
        for k, v in ds.items():
            dims = v.dims
            selection = {dn: min(1,len(ds[dn])) for dn in dims}
            val = v.isel(selection).values
            if val == 0 or not np.isfinite(val):
                logger.warning(f"{k}[{selection}] = {val}")
            else:
                good.append(k)
        logger.info(f"{good} look good.")


def test_cat(cat):
    intake_tools.traverse_tree(
        cat,
        subcat_callback=log_pos,
        entry_callback=test_entry,
    )


if __name__ == "__main__":
    cat = intake.open_catalog(sys.argv[1])
    test_cat(cat)
    print (sys.argv[1])
