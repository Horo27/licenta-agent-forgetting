"""Command-line entry point: ``uv run fademem <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from fademem.config import load_params, load_profile
from fademem.providers import Message, build_embedder, build_llm

DEFAULT_PARAMS = "configs/fademem/default.yaml"


def _show_config(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    params = load_params(args.params)
    out = {"profile": profile.model_dump(), "params": params.model_dump()}
    print(json.dumps(out, indent=2))
    return 0


def _check_providers(args: argparse.Namespace) -> int:
    """Send one tiny request to every role in the profile and report what answered."""
    profile = load_profile(args.profile)
    ok = True
    for role in ("memory_llm", "answer_llm", "judge_llm"):
        cfg = getattr(profile, role)
        try:
            result = build_llm(cfg).complete([Message("user", "Reply with the word: ok")])
            print(f"{role:11} {cfg.model:28} ok  -> {result.text.strip()[:40]!r}")
        except Exception as exc:
            ok = False
            print(f"{role:11} {cfg.model:28} FAIL {type(exc).__name__}: {exc}")
    try:
        vectors = build_embedder(profile.embedder).embed(["hello world"])
        print(f"{'embedder':11} {profile.embedder.model:28} ok  -> dim {vectors.shape[1]}")
    except Exception as exc:
        ok = False
        print(f"{'embedder':11} {profile.embedder.model:28} FAIL {type(exc).__name__}: {exc}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fademem")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show-config", help="print the resolved profile and hyperparameters")
    show.add_argument("--profile", default="test", help="profile name or YAML path")
    show.add_argument("--params", default=DEFAULT_PARAMS, help="hyperparameter YAML")
    show.set_defaults(func=_show_config)

    check = sub.add_parser("check-providers", help="ping every model in a profile")
    check.add_argument("--profile", default="test", help="profile name or YAML path")
    check.set_defaults(func=_check_providers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
