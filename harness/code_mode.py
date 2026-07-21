import secrets
from collections.abc import Callable

_grants: dict[str, dict[str, Callable]] = {}


def issue_token(allowed: dict[str, Callable]) -> str:
    token = secrets.token_urlsafe(16)
    _grants[token] = allowed
    return token


def revoke(token: str) -> None:
    _grants.pop(token, None)


def call(token: str, name: str, args: list) -> object:
    allowed = _grants.get(token)
    if allowed is None or name not in allowed:
        raise PermissionError(f"not permitted: {name}")
    return allowed[name](*args)
