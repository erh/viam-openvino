"""``erh:openvino:diagnostics`` - a generic service exposing device discovery and benchmarking.

The mlmodel API has no DoCommand RPC, so this small service is the client-reachable entry point for
``list_devices``, ``get_compiled_properties``, ``benchmark`` and ``stats`` (see registry.py).
"""

from __future__ import annotations

from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.generic import Generic
from viam.utils import ValueTypes, struct_to_dict

from . import registry


class OpenVINODiagnostics(Generic, EasyResource):
    MODEL: ClassVar[Model] = "erh:openvino:diagnostics"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.mlmodel_name: Optional[str] = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        name = _mlmodel_name(struct_to_dict(config.attributes))
        # Depend on the mlmodel so it is built first, but keep it optional so list_devices works regardless.
        return [], [name] if name else []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        self.mlmodel_name = _mlmodel_name(struct_to_dict(config.attributes))

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        override = command.get("mlmodel_name")
        name = override if isinstance(override, str) and override else self.mlmodel_name
        return await registry.run_diagnostic(command, name)


def _mlmodel_name(attrs: Mapping[str, object]) -> Optional[str]:
    v = attrs.get("mlmodel_name")
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        raise ValueError("mlmodel_name must be a non-empty string when set")
    return v.strip()
