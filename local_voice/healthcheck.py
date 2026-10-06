import sys

from .client import EngineClient


def main():
    try:
        state = EngineClient().health()
        if state.get("status") != "ready":
            raise RuntimeError(state.get("error") or "Loading models")
        if state.get("device") != "cuda:0":
            raise RuntimeError("The CUDA GPU is not active")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

