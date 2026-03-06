from project_logging import configure_logging, get_logger


def main():
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Hello from sympy-rlvr!")


if __name__ == "__main__":
    main()
