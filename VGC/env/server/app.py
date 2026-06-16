from __future__ import annotations

from openenv.core.env_server import create_app

from env.models import HardreadAction, HardreadObservation
from env.server.environment import HardreadEnvironment

app = create_app(
    HardreadEnvironment,
    HardreadAction,
    HardreadObservation,
    env_name="hardread",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

