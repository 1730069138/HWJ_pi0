import dataclasses
import enum
import logging
import socket
import time

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment used when serving a default policy.
    env: EnvMode = EnvMode.ALOHA_SIM

    # Used when the request doesn't contain a prompt and the model has no default.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
) -> _policy.Policy:
    """Create a default policy for an environment."""

    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create the policy requested on the command line."""

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
            )


def warmup_dummy_real_policy(policy: _policy.Policy) -> None:
    """Compile the DummyReal JAX inference path before accepting clients."""

    observation = {
        "cam_global": np.zeros((240, 424, 3), dtype=np.uint8),
        "cam_wrist": np.zeros((240, 424, 3), dtype=np.uint8),
        "state": np.asarray(
            [180.0, 80.0, -100.0, -120.0, 115.0, 140.0, -115.0],
            dtype=np.float32,
        ),
        "prompt": "Pick up the screwdriver and place it into the nearby express box.",
    }

    logging.info(
        "Warming up DummyReal policy; the first compilation may take about 30 seconds..."
    )
    start_time = time.monotonic()
    result = policy.infer(observation)
    elapsed = time.monotonic() - start_time

    actions = np.asarray(result["actions"])
    expected_shape = (40, 7)
    if actions.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected warmup action shape: {actions.shape}, expected {expected_shape}"
        )

    logging.info(
        "DummyReal policy warmup completed in %.3fs, actions shape=%s",
        elapsed,
        actions.shape,
    )


def warmup_dummyx_policy(policy: _policy.Policy) -> None:
    """Compile the DummyX simulation JAX inference path before accepting clients."""

    observation = {
        "observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float64),
        "prompt": "Pick up the screwdriver and drop it into the box.",
    }

    logging.info(
        "Warming up DummyX simulation policy; "
        "the first compilation may take about 30 seconds..."
    )
    start_time = time.monotonic()
    result = policy.infer(observation)
    elapsed = time.monotonic() - start_time

    actions = np.asarray(result["actions"])
    if actions.ndim != 2:
        raise RuntimeError(
            f"Unexpected DummyX warmup action rank: "
            f"shape={actions.shape}, expected a 2-D action chunk"
        )
    if actions.shape[0] == 0 or actions.shape[1] < 7:
        raise RuntimeError(
            f"Unexpected DummyX warmup action shape: {actions.shape}; "
            "expected a non-empty action chunk with at least 7 values per action"
        )

    logging.info(
        "DummyX simulation policy warmup completed in %.3fs, actions shape=%s",
        elapsed,
        actions.shape,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if isinstance(args.policy, Checkpoint):
        if args.policy.config == "pi0_dummy_real_lora":
            warmup_dummy_real_policy(policy)
        elif args.policy.config == "pi0_dummyx_lora":
            warmup_dummyx_policy(policy)

    # Record only real requests; don't include the synthetic warmup request.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))