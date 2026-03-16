print("Hello!\nI am Scrooge...")

import runpy


def main() -> None:
    runpy.run_module("bot.runtime", run_name="__main__")


if __name__ == "__main__":
    main()

