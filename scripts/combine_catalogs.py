import argparse
from dataclasses import asdict, dataclass, field
from collections import defaultdict
from functools import reduce
import datetime
from pathlib import Path
from typing import Iterable
import yaml
import json
from jinja2 import Template
from pydantic import BaseModel, Field


class RawDataSource(BaseModel):
    driver: str
    args: dict
    allowed_parameters: dict[str, list]


class ParameterDescription(BaseModel):
    default: str | int | float
    type: str
    description: str = Field(default_factory=str)

    @staticmethod
    def from_entry(entry):
        return ParameterDescription(
            default=entry["default"], type=entry["type"], description=entry["description"]
        )


class DataSource(BaseModel):
    raw: RawDataSource
    description: str = Field(default_factory=str)
    parameter_descriptions: dict[str, ParameterDescription] = Field(
        default_factory=dict
    )
    metadata: dict = Field(default_factory=dict)

    @staticmethod
    def from_entry(entry):
        raw = RawDataSource(
            driver=entry["driver"],
            args=entry["args"],
            allowed_parameters={
                k: v["allowed"] for k, v in entry.get("parameters", {}).items()
            },
        )
        return DataSource(
            raw=raw,
            description=entry.get("description", ""),
            parameter_descriptions={
                k: ParameterDescription.from_entry(v)
                for k, v in entry.get("parameters", {}).items()
            },
            metadata=entry.get("metadata", {}),
        )

    @property
    def parameters(self):
        return {
            k: {
                **v.model_dump(),
                "allowed": self.raw.allowed_parameters[k],
            }
            for k, v in self.parameter_descriptions.items()
        }

    def __or__(self, other: "DataSource | None") -> "DataSource":
        if other is None:
            return self
        assert isinstance(other, DataSource)
        return other

    def __ror__(self, other: None) -> "DataSource":
        assert other is None
        return self

    def to_entry(self, name) -> dict:
        return {
            "driver": self.raw.driver,
            "args": self.raw.args,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "description": self.description,
        }


class MultiDataSource(BaseModel):
    id: str
    raw: dict[str, RawDataSource]
    description: str = Field(default_factory=str)
    parameter_descriptions: dict[str, ParameterDescription] = Field(
        default_factory=dict
    )
    metadata: dict = Field(default_factory=dict)

    @staticmethod
    def from_datasource(id: str, provider: str, source: DataSource) -> "MultiDataSource":
        return MultiDataSource(id=id, raw = {provider: source.raw}, description=source.description, parameter_descriptions=source.parameter_descriptions, metadata=source.metadata)

    @property
    def parameters(self):
        return {
            k: {
                **v.model_dump(),
                "allowed": list(sorted(set(allowed for r in self.raw.values() for allowed in r.allowed_parameters[k]))),
            }
            for k, v in self.parameter_descriptions.items()
        }

    def __or__(self, other: "MultiDataSource | None") -> "MultiDataSource":
        if other is None:
            return self
        assert isinstance(other, MultiDataSource)
        if self.id != other.id:
            raise ValueError(f"trying to merge datasources with different ids: {self.id} != {other.id}, this is likely a BUG in the code.")

        attrs = {}
        for attr in ["description", "parameter_descriptions", "metadata"]:
            s = getattr(self, attr)
            o = getattr(other, attr)
            if s:
                if o:
                    if s != o:
                        raise ValueError(f"{attr} of source {self.id} differ between providers {list(self.raw)} and {list(other.raw)}!\n"
                                         "Please ensure they are either identical or specify it only once and leave the other empty.")
                attrs[attr] = s
            else:
                attrs[attr] = o

        return MultiDataSource(id=self.id, raw={**self.raw, **other.raw}, **attrs)

    def __ror__(self, other: None) -> "MultiDataSource":
        assert other is None
        return self


@dataclass
class SimpleCat:
    sources: dict[str, "SimpleCat | DataSource"]
    location: Path | None = None
    description: str | None = None

    def __or__(self, other: "SimpleCat | None") -> "SimpleCat":
        if other is None:
            return self
        assert isinstance(other, SimpleCat)
        keys = set(self.sources) | set(other.sources)
        return SimpleCat({k: self.sources.get(k) | other.sources.get(k) for k in keys})

    def __ror__(self, other: None) -> "SimpleCat":
        assert other is None
        return self

    def to_entry(self, name) -> dict:
        res = {
            "driver": "yaml_file_cat",
            "args": {"path": f"{{{{CATALOG_DIR}}}}/{name}/catalog.yaml"},
        }
        if self.description:
            res["description"] = self.description
        return res


def parse_source(source: dict, path: Path) -> SimpleCat | DataSource:
    assert source["driver"] in ["yaml_file_cat", "zarr", "netcdf"]
    if source["driver"] == "yaml_file_cat":
        return read_cat(
            Path(Template(source["args"]["path"]).render(CATALOG_DIR=str(path.parent)))
        )
    else:
        return DataSource.from_entry(source)


def read_cat(path: Path) -> SimpleCat:
    cat = yaml.safe_load(open(path))
    assert set(cat) == {
        "sources"
    }, f"can't handle catalogs which not only have sources: '{ path }'"
    return SimpleCat(
        sources={k: parse_source(v, path) for k, v in cat["sources"].items()},
        location=path.parent,
    )


def write_cat(cat: SimpleCat, path: Path) -> None:
    sources = {
        k: v.to_entry(k)
        for k, v in cat.sources.items()
    }
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "catalog.yaml", "w") as outfile:
        yaml.dump({"sources": sources}, outfile)

    for k, v in cat.sources.items():
        if isinstance(v, SimpleCat):
            write_cat(v, path / k)


def iterate_sources(
    cat: SimpleCat, prefix: list[str] | None = None
) -> Iterable[tuple[str, DataSource]]:
    prefix = prefix or []
    for name, source in cat.sources.items():
        key = prefix + [name]
        if isinstance(source, SimpleCat):
            yield from iterate_sources(source, key)
        else:
            yield ".".join(key), source


def collect_mlds(catalogs: dict[str, SimpleCat]) -> dict[str, MultiDataSource]:
    sources: dict[str, dict[str, MultiDataSource]] = defaultdict(dict)
    for zone, cat in catalogs.items():
        for id, source in iterate_sources(cat):
            sources[id][zone] = MultiDataSource.from_datasource(id, zone, source)

    return {k: reduce(lambda a, b: a | b, s.values()) for k, s in sources.items()}

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

def join_mlds(mlds: MultiDataSource, visible_sources: list[str]) -> DataSource | None:
    raw = None
    for s in visible_sources:
        raw = mlds.raw.get(s, raw)
    if raw:
        return DataSource(raw=raw, description=mlds.description, parameter_descriptions=mlds.parameter_descriptions, metadata=mlds.metadata)
    return None

def deep_dict():
    return defaultdict(deep_dict)

def deep_insert(d, key, value):
    keys = key.split(".")
    for subkey in keys[:-1]:
        d = d[subkey]
    d[keys[-1]] = value

def deep_dict_to_simple_cat(dd):
    if isinstance(dd, dict):
        return SimpleCat({
            k: deep_dict_to_simple_cat(v)
            for k, v in dd.items()
        })
    return dd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "zones_file", type=Path, help="availability zone definition file"
    )
    parser.add_argument("outdir", type=Path, help="output folder")
    args = parser.parse_args()

    basedir = args.zones_file.absolute().parent
    outdir = args.outdir
    zones = yaml.safe_load(open(args.zones_file))
    cat = read_cat(basedir / "catalog.yaml")

    full_cat = collect_mlds(cat.sources)

    zoned_cats = deep_dict()
    ds_available_at = defaultdict(set)

    for zone, sources in zones.items():
        for id, mlds in full_cat.items():
            if ds := join_mlds(mlds, sources):
                deep_insert(zoned_cats[zone], id, ds)
                ds_available_at[id].add(zone)

    ds_available_at = {id: list(sorted(zones)) for id, zones in ds_available_at.items()}

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "mlds.json", "w") as outfile:
        json.dump(
            {
                id: {
                    **mlds.model_dump(),
                    "available_at": ds_available_at[id],
                }
                for id, mlds in full_cat.items()
            },
            outfile,
            cls=DateTimeEncoder,
            indent=2,
        )

    zoned_cats = {k: deep_dict_to_simple_cat(v) for k, v in zoned_cats.items()}

    for zone, zcat in zoned_cats.items():
        write_cat(zcat, outdir / zone)

    with open(outdir / "catalog.yaml", "w") as outfile:
        yaml.dump(
            {
                "sources": {
                    zone: {
                        "driver": "yaml_file_cat",
                        "args": {"path": f"{{{{CATALOG_DIR}}}}/{zone}/catalog.yaml"},
                        "description": f"catalog as visible from {zone}",
                    }
                    for zone in zones
                }
            },
            outfile,
        )


if __name__ == "__main__":
    main()
