"""
Ontology classes for defining reusable annotation and taxonomy structures.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from __future__ import annotations

import fnmatch
from datetime import datetime
from typing import Any, Optional

from fiftyone.core.annotation.attributes import Attribute
from fiftyone.core.odm.ontology import OntologyDocument, OntologyType


class Ontology:
    """Abstract base class for ontology types.

    Ontologies are global, named, versioned resources that define reusable
    annotation structures. They are not scoped to any single dataset.

    Subclasses must set :attr:`_TYPE` to the corresponding
    :class:`fiftyone.core.odm.ontology.OntologyType` value.

    Args:
        name: the ontology name
        description: optional description
    """

    _TYPE: Optional[str] = None

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self._doc = None

    @property
    def version(self) -> Optional[int]:
        """The version of this ontology, or ``None`` if not yet saved."""
        if self._doc is not None:
            return self._doc.version

        return None

    @property
    def created_at(self) -> Optional[datetime]:
        """The datetime this ontology was created, or ``None`` if not yet
        saved.
        """
        if self._doc is not None:
            return self._doc.created_at

        return None

    @property
    def last_modified_at(self) -> Optional[datetime]:
        """The datetime this ontology was last modified, or ``None`` if not
        yet saved.
        """
        if self._doc is not None:
            return self._doc.last_modified_at

        return None

    def save(self) -> None:
        """Saves this ontology to the database."""
        if self._doc is None:
            self._doc = OntologyDocument(
                name=self.name,
                type=self._TYPE,
                description=self.description,
                root=self._get_root(),
            )
        else:
            self._doc.name = self.name
            self._doc.description = self.description
            self._doc.root = self._get_root()

        self._doc.save()

    def reload(self) -> None:
        """Reloads this ontology from the database."""
        if self._doc is None:
            raise ValueError(
                "Cannot reload an ontology that has not been saved"
            )

        self._doc.reload()
        self._apply_doc(self._doc)

    def delete(self) -> None:
        """Deletes this ontology from the database."""
        if self._doc is None:
            raise ValueError(
                "Cannot delete an ontology that has not been saved"
            )

        self._doc.delete()
        self._doc = None

    def _get_root(self) -> Any:
        """Returns the serialized root data for storage.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def _apply_doc(self, doc: OntologyDocument) -> None:
        """Hydrates this instance from a loaded document.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serializes this ontology to a dict.

        Returns:
            a dict
        """
        d: dict[str, Any] = {
            "name": self.name,
            "type": self._TYPE,
        }

        if self.description is not None:
            d["description"] = self.description

        if self.version is not None:
            d["version"] = self.version

        if self.created_at is not None:
            d["created_at"] = self.created_at

        if self.last_modified_at is not None:
            d["last_modified_at"] = self.last_modified_at

        return d

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, version={self.version})"
        )


class AnnotationOntology(Ontology):
    """Ontology for defining annotation structures.

    Bundles typed attributes (with optional conditional display logic) and
    taxonomy references into a single document that gets connected to a
    label schema on a field.

    Args:
        name: the ontology name
        description: optional description
        taxonomies: list of taxonomy names referenced by this ontology
        attributes: list of :class:`Attribute` instances

    Example::

        AnnotationOntology(
            name="vehicle_damage_ontology",
            description="Vehicle damage annotation",
            taxonomies=["vehicle_classes"],
            attributes=[
                Attribute(
                    name="damage_present",
                    type="bool",
                    component="checkbox",
                ),
                Attribute(
                    name="damage_location",
                    type="str",
                    component="dropdown",
                    values=["front", "rear", "driver_side", "passenger_side"],
                    when=[When(WhenOperator.EQUALS, field="damage_present", value=True)],
                ),
            ],
        )
    """

    _TYPE = OntologyType.ANNOTATION_ONTOLOGY.value

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        taxonomies: Optional[list[str]] = None,
        attributes: Optional[list[Attribute]] = None,
    ):
        super().__init__(name=name, description=description)
        self.taxonomies = taxonomies or []
        self.attributes = attributes or []

    def _get_root(self) -> dict:
        return {
            "taxonomies": self.taxonomies,
            "attributes": [attr.to_dict() for attr in self.attributes],
        }

    def _apply_doc(self, doc: OntologyDocument) -> None:
        self.name = doc.name
        self.description = doc.description
        self.taxonomies = doc.root.get("taxonomies", [])
        self.attributes = [
            Attribute.from_dict(a) for a in doc.root.get("attributes", [])
        ]

    def to_dict(self) -> dict:
        """Serializes this annotation ontology to a dict.

        Returns:
            a dict
        """
        d = super().to_dict()
        d["root"] = self._get_root()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AnnotationOntology:
        """Creates an :class:`AnnotationOntology` from a dict.

        Args:
            d: an annotation ontology dict

        Returns:
            an :class:`AnnotationOntology`
        """
        root = d.get("root", {})
        return cls(
            name=d["name"],
            description=d.get("description"),
            taxonomies=root.get("taxonomies", []),
            attributes=[
                Attribute.from_dict(a) for a in root.get("attributes", [])
            ],
        )


class Node:
    """A node in a :class:`Taxonomy` hierarchy.

    .. note::

        Taxonomy support is not yet implemented. This class is a
        placeholder for phase 2.

    Raises:
        NotImplementedError: always
    """

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "Node is not yet implemented; taxonomy support is planned for "
            "phase 2"
        )


class Taxonomy(Ontology):
    """A named, versioned, hierarchical class taxonomy.

    .. note::

        Taxonomy support is not yet implemented. This class is a
        placeholder for phase 2.

    Raises:
        NotImplementedError: always
    """

    _TYPE = OntologyType.TAXONOMY.value

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "Taxonomy is not yet implemented; taxonomy support is planned "
            "for phase 2"
        )


# ---- Type dispatch --------------------------------------------------------

_TYPE_TO_CLS: dict[str, type[Ontology]] = {
    OntologyType.ANNOTATION_ONTOLOGY.value: AnnotationOntology,
}


def _from_doc(doc: OntologyDocument) -> Ontology:
    """Constructs the appropriate SDK class from an ODM document."""
    cls = _TYPE_TO_CLS.get(doc.type)
    if cls is None:
        raise ValueError(f"Unknown ontology type: {doc.type!r}")

    root = doc.root or {}
    instance = cls.__new__(cls)
    Ontology.__init__(instance, name=doc.name, description=doc.description)
    instance._doc = doc
    instance._apply_doc(doc)
    return instance


# ---- Module-level CRUD functions ------------------------------------------


def create_ontology(ontology: Ontology) -> None:
    """Saves an ontology to the database.

    Args:
        ontology: an :class:`Ontology` instance
    """
    ontology.save()


def load_ontology(name: str) -> Ontology:
    """Loads the latest version of an ontology by name.

    Args:
        name: the ontology name

    Returns:
        an :class:`Ontology`

    Raises:
        ValueError: if no ontology with the given name exists
    """
    doc = (
        OntologyDocument.objects(name=name)  # pylint: disable=no-member
        .order_by("-version")
        .first()
    )
    if doc is None:
        raise ValueError(f"Ontology '{name}' not found")

    return _from_doc(doc)


def list_ontologies(glob_patt: Optional[str] = None) -> list[str]:
    """Lists ontology names in the database.

    Args:
        glob_patt: an optional glob pattern to filter names

    Returns:
        a sorted list of ontology names
    """
    if glob_patt is not None:
        regex = fnmatch.translate(glob_patt)
        docs = OntologyDocument.objects(  # pylint: disable=no-member
            name__regex=regex
        )
    else:
        docs = OntologyDocument.objects()  # pylint: disable=no-member

    return sorted(docs.distinct("name"))


def ontology_exists(name: str) -> bool:
    """Checks if an ontology with the given name exists.

    Args:
        name: the ontology name

    Returns:
        True/False
    """
    return (
        OntologyDocument.objects(  # pylint: disable=no-member
            name=name
        ).count()
        > 0
    )


def delete_ontology(name: str, force: bool = False) -> None:
    """Deletes an ontology and all its versions from the database.

    Args:
        name: the ontology name
        force: whether to delete even if the ontology is in use.
            By default, raises an error if the ontology is referenced
            by a label schema. Not yet enforced — will be implemented
            with label schema integration.
    """
    # TODO: check if in use when force=False (requires label schema integration)
    count = OntologyDocument.objects(  # pylint: disable=no-member
        name=name
    ).delete()
    if count == 0:
        raise ValueError(f"Ontology '{name}' not found")


def rename_ontology(name: str, new_name: str) -> None:
    """Renames an ontology (all versions).

    Args:
        name: the current ontology name
        new_name: the new name
    """
    count = OntologyDocument.objects(  # pylint: disable=no-member
        name=name
    ).update(set__name=new_name)
    if count == 0:
        raise ValueError(f"Ontology '{name}' not found")


def clone_ontology(name: str, new_name: str) -> Ontology:
    """Clones an ontology under a new name.

    Loads the latest version and saves it as a new ontology.

    Args:
        name: the source ontology name
        new_name: the name for the clone

    Returns:
        the cloned :class:`Ontology`
    """
    source = load_ontology(name)
    source.name = new_name
    source._doc = None
    source.save()
    return source
