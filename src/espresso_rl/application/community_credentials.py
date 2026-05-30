from __future__ import annotations

from espresso_rl.domain.community import CommunityUploadCredentials
from espresso_rl.ports.community import CommunityCredentialRegistrar, CommunityCredentialStore


class CommunityCredentialService:
    def __init__(
        self,
        store: CommunityCredentialStore,
        registrar: CommunityCredentialRegistrar,
    ) -> None:
        self._store = store
        self._registrar = registrar

    def resolve_for_upload(
        self,
        configured: CommunityUploadCredentials | None = None,
        *,
        allow_registration: bool = True,
    ) -> CommunityUploadCredentials | None:
        if configured is not None:
            return configured

        stored = self._store.load()
        if stored is not None:
            return stored

        if not allow_registration:
            return None

        credentials = self._registrar.register_install()
        self._store.save(credentials)
        return credentials

    def rotate(
        self,
        current: CommunityUploadCredentials | None = None,
    ) -> CommunityUploadCredentials:
        credentials = current or self._store.load()
        if credentials is None:
            raise ValueError("no community upload credentials are available to rotate")
        rotated = self._registrar.rotate_credentials(credentials)
        self._store.save(rotated)
        return rotated

    def revoke(
        self,
        current: CommunityUploadCredentials | None = None,
    ) -> None:
        credentials = current or self._store.load()
        if credentials is None:
            return
        self._registrar.revoke_credentials(credentials)
        self._store.clear()
