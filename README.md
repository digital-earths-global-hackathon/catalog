# catalog

This repo contains data catalogs for the [Digital Earths Global Hackathon](https://digital-earths-global-hackathon.github.io/hk25/).

From a user point of view, it's best to think of [**one** single catalog](https://digital-earths-global-hackathon.github.io/catalog/), covering all datasets available for the hackathon.
Now, depending on where you want to access data from, a given dataset might be better reachable using a local path, instead of a global URL.
In order to facilitate usage, we technically provide one catalog per *usage* site.
Users should select the most appropriate catalog (either manually or automatically), based on where they execute an analysis.

## source catalogs and usable catalogs

We distinguish between two sets of catalogs:

* provider catalogs
* usable catalogs

**Provider** catalogs are the "source code": it's once catalog per data providing location, and should be edited by the people providing a given dataset (instance).

**Usable** catalogs are the "compiled code": based on the file `availability_zones.yaml`, multiple provider catalogs are combined into catalogs showing all the entries which can be reached from a given site. E.g. at `EU`, we might have access to all data stored locally at the `EU` node, as well as all data available `online`. Thus these two catalogs are combined.

## adding to the catalog

Datasets should be added to, changed in or removed from the catalog whenever they are added, moved or removed from a storage location.
There's one catalog per *provider*, and the dataset should be added to the catalog where the data is stored.
A dataset *must* be identified by a unique id, which *must* be the key of the `sources` dictionary in the catalog.
If a dataset is available at multiple storage locations, the unique id must be identical across all storage locations.
`metadata` *should* be specified with each entry.
**If a dataset is available at multiple storage locations, metadata should only be provided in one file**. The others can have a comment like `# metadata in <location>`. The scripts will take care of copying the metadata to the other catalogs.
Metadata corresponds to the dataset's global attributes.
It *should* be the same, but *may* be extended in the catalog.
We recommend to fill at least the following metadata attributes (following [CF](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html) and [ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3) conventions):

Key | Value
--- | ---
`title` | Short phrase or sentence describing the dataset
`summary` | Paragraph describing the dataset
`references` | Comma-separated list of URL/DOI to extended information <br/>Published or web-based references that describe the data or methods used to produce it.
`license` | [SPDX-ID](https://spdx.org/licenses/)
`Conventions` | A comma-separated list of the conventions that are followed by the dataset, e.g., 'ACDD-1.3, CF-1.12'.

You may want to think additionally of the following:

Key | Value
--- | ---
`creator_name` | Comma-separated list of names
`creator_email` | Comma-separated list of emails
`creator_id` | Comma-separated list of identifiers (e.g. ORCID)
`source` | Method of production of the original data
`history` | Audit trail for modifications to the original data
`keywords` | Comma-separated list of keywords
`institution` | Institution responsible for the dataset