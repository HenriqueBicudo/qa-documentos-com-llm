"""
config/logger.py — Configuração centralizada de logging do projeto

Uso:
    from config.logger import get_logger
    logger = get_logger(__name__)
    logger.info("mensagem")
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado com o nome do módulo que chamou.

    Formato: [LEVEL] nome_do_modulo: mensagem
    Nível padrão: INFO — mostra info, warnings e erros, ignora debug.
    """
    logger = logging.getLogger(name)

    # Evita adicionar handlers duplicados se get_logger for chamado várias vezes
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
