import os


REQUIRED_DIRS = {"test_waves", "test_datasets"}


def create_dirs():
    for d in REQUIRED_DIRS:
        if not os.path.isdir(d):
            os.makedirs(d)


def create_env():
    if not os.path.isfile(".env"):
        with open(".env", "w") as f:
            f.write(
                """
# ARCHITECTURE=C1
# EPOCHS=100
            """
            )


if __name__ == "__main__":
    create_dirs()
    create_env()
