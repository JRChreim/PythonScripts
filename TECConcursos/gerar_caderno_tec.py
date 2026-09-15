from __future__ import annotations

import argparse
import os
import re
import traceback
from pathlib import Path

import yaml
from playwright.sync_api import Playwright, sync_playwright


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SITE_CONFIG_PATH = REPO_ROOT / "site.yaml"
DEFAULT_CADERNOS_CONFIG_PATH = REPO_ROOT / "cadernos.yaml"

LOGIN_EMAIL = os.environ.get("TEC_EMAIL")
LOGIN_PASSWORD = os.environ.get("TEC_PASSWORD")


def validar_credenciais() -> None:
    ausentes = []
    if not LOGIN_EMAIL:
        ausentes.append("TEC_EMAIL")
    if not LOGIN_PASSWORD:
        ausentes.append("TEC_PASSWORD")

    if ausentes:
        nomes = ", ".join(ausentes)
        raise RuntimeError(
            f"Defina {nomes} no ambiente antes de executar a automação. "
            "Consulte TECConcursos/README.md para os comandos do seu sistema."
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TecConcursos notebooks from TECConcursos/site.yaml and "
            "TECConcursos/cadernos.yaml. Site labels and option catalog live "
            "in the site file; notebook combinations live in the cadernos file."
        )
    )
    parser.add_argument(
        "--site-config",
        type=Path,
        default=DEFAULT_SITE_CONFIG_PATH,
        help="Path to site.yaml. Defaults to TECConcursos/site.yaml.",
    )
    parser.add_argument(
        "--caderno",
        help="Exact name of the notebook preset to generate. Defaults to the first preset in the file.",
    )
    parser.add_argument(
        "--config",
        "--cadernos-config",
        dest="cadernos_config",
        type=Path,
        default=DEFAULT_CADERNOS_CONFIG_PATH,
        help="Path to cadernos.yaml. Defaults to TECConcursos/cadernos.yaml.",
    )
    parser.add_argument(
        "--list-cadernos",
        action="store_true",
        help="List the available notebook presets and exit.",
    )
    return parser


def resolver_caminho_config(path: Path) -> Path:
    if path.is_absolute():
        return path

    relative_to_cwd = Path.cwd() / path
    if relative_to_cwd.exists():
        return relative_to_cwd

    relative_to_script = REPO_ROOT / path
    if relative_to_script.exists():
        return relative_to_script

    return relative_to_script


def carregar_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        dados = yaml.safe_load(f) or {}
    if not isinstance(dados, dict):
        raise ValueError(f"O arquivo {path} precisa conter um dicionário YAML.")
    return dados


def carregar_configuracao(site_path: Path, cadernos_path: Path) -> dict:
    site_dados = carregar_yaml(site_path)
    cadernos_dados = carregar_yaml(cadernos_path)

    if "site" not in site_dados:
        raise ValueError(
            f"A configuração em {site_path} precisa conter a chave 'site'."
        )
    if "cadernos" not in cadernos_dados:
        raise ValueError(
            f"A configuração em {cadernos_path} precisa conter a chave 'cadernos'."
        )

    return {"site": site_dados["site"], "cadernos": cadernos_dados["cadernos"]}


def selecionar_caderno(config: dict, nome_caderno: str | None) -> dict:
    cadernos = config.get("cadernos", [])
    if not cadernos:
        raise ValueError("Nenhum caderno foi definido em cadernos.yaml.")

    if nome_caderno is None:
        return cadernos[0]

    for caderno in cadernos:
        if caderno.get("nome") == nome_caderno:
            return caderno

    disponiveis = ", ".join(caderno.get("nome", "<sem nome>") for caderno in cadernos)
    raise ValueError(
        f"Caderno '{nome_caderno}' não encontrado. Disponíveis: {disponiveis}"
    )


def listar_cadernos(config: dict):
    cadernos = config.get("cadernos", [])
    if not cadernos:
        print("Nenhum caderno cadastrado.")
        return

    print("Cadernos disponíveis:")
    for caderno in cadernos:
        nome = caderno.get("nome", "<sem nome>")
        print(f"- {nome}")


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def esperar_usuario(mensagem: str = "Pressione ENTER para continuar..."):
    input(f"\n{mensagem}\n")


def abrir_filtro(page, nome_filtro: str):
    page.get_by_role("listitem", name=nome_filtro, exact=True).click()


def valores_disponiveis_para_filtro(site: dict, nome_filtro: str) -> list[str]:
    filtro_cfg = site.get("filters", {}).get(nome_filtro)
    if filtro_cfg is None:
        return []
    return as_list(filtro_cfg.get("values"))


def validar_valor_de_filtro(site: dict, nome_filtro: str, valor: str):
    valores_disponiveis = valores_disponiveis_para_filtro(site, nome_filtro)
    if valores_disponiveis and valor not in valores_disponiveis:
        disponiveis = ", ".join(valores_disponiveis)
        raise ValueError(
            f"O valor '{valor}' não está definido em site.yaml para o filtro "
            f"'{nome_filtro}'. Valores disponíveis: {disponiveis}"
        )


def valores_disponiveis_para_opcao(site: dict, nome_opcao: str) -> list:
    options_cfg = site.get("options", {})
    available_cfg = options_cfg.get("available", {})
    return as_list(available_cfg.get(nome_opcao))


def validar_opcoes(site: dict, opcoes: dict):
    options_cfg = site.get("options", {})
    labels = options_cfg.get("labels", {})
    available_cfg = options_cfg.get("available", {})
    chaves_conhecidas = set(labels) | set(available_cfg) | {"somente_comentadas", "remover"}

    for chave, valor in opcoes.items():
        if chave not in chaves_conhecidas:
            disponiveis = ", ".join(sorted(chaves_conhecidas))
            raise ValueError(
                f"Opção desconhecida em cadernos.yaml: '{chave}'. "
                f"Opções disponíveis: {disponiveis}"
            )

        if chave == "somente_comentadas":
            if not isinstance(valor, bool):
                raise ValueError(
                    "A opção 'somente_comentadas' precisa ser true ou false."
                )
            continue

        valores_disponiveis = valores_disponiveis_para_opcao(site, chave)
        if not valores_disponiveis:
            continue

        valores_escolhidos = as_list(valor)
        for valor_escolhido in valores_escolhidos:
            if valor_escolhido not in valores_disponiveis:
                disponiveis = ", ".join(map(str, valores_disponiveis))
                raise ValueError(
                    f"O valor '{valor_escolhido}' não está definido em site.yaml "
                    f"para a opção '{chave}'. Valores disponíveis: {disponiveis}"
                )


def clicar_texto(page, texto: str, *, exact: bool = True, occurrence: str | None = None):
    locator = page.get_by_text(texto, exact=exact)
    if occurrence == "first":
        locator = locator.first
    elif occurrence == "last":
        locator = locator.last
    locator.click()


def pesquisar_no_filtro_aberto(page, site: dict, termo: str):
    search_cfg = site["search"]
    trigger_text = search_cfg["trigger_text"]
    textbox_name = search_cfg["textbox_name"]

    campo = page.get_by_role("textbox", name=textbox_name).last
    if not campo.is_visible():
        pesquisar = page.locator(
            "a.link-limpo:visible",
            has_text=re.compile(re.escape(trigger_text)),
        ).last

        pesquisar.wait_for(state="visible", timeout=5000)
        pesquisar.click()

    campo.wait_for(state="visible", timeout=5000)
    campo.fill(termo)


def selecionar_valor_em_filtro(page, site: dict, nome_filtro: str, valor: str):
    filtros_cfg = site["filters"]
    filtro_cfg = filtros_cfg.get(nome_filtro)
    if filtro_cfg is None:
        raise ValueError(f"Filtro '{nome_filtro}' não está configurado em site.filters.")

    mode = filtro_cfg.get("mode", "click")
    exact = filtro_cfg.get("exact", True)
    occurrence = filtro_cfg.get("occurrence")

    if mode == "search":
        pesquisar_no_filtro_aberto(page, site, valor)
        clicar_texto(page, valor, exact=exact, occurrence=occurrence)
        return

    if mode == "click":
        clicar_texto(page, valor, exact=exact, occurrence=occurrence)
        return

    raise ValueError(
        f"Modo de seleção desconhecido para o filtro '{nome_filtro}': {mode!r}"
    )


def aplicar_filtros(page, filtros: dict, site: dict):
    filtros_cfg = site.get("filters", {})

    for nome_filtro, valores in filtros.items():
        filtro_cfg = filtros_cfg.get(nome_filtro)
        if filtro_cfg is None:
            raise ValueError(
                f"Filtro '{nome_filtro}' não está configurado em site.filters."
            )

        valores = as_list(valores)
        for valor in valores:
            validar_valor_de_filtro(site, nome_filtro, valor)

        abrir_filtro(page, nome_filtro)
        for valor in valores:
            selecionar_valor_em_filtro(page, site, nome_filtro, valor)


def aplicar_opcoes(page, site: dict, opcoes: dict | None):
    if not opcoes:
        return

    options_cfg = site["options"]
    labels = options_cfg["labels"]
    validar_opcoes(site, opcoes)

    page.get_by_role("listitem", name=options_cfg["button"], exact=True).click()

    if opcoes.get("somente_comentadas"):
        clicar_texto(page, labels["somente_comentadas"])

    for chave in ("dificuldade", "tipo_questao", "gabarito"):
        texto = opcoes.get(chave)
        if texto:
            clicar_texto(page, labels.get(chave, texto))

    first_match = set(options_cfg.get("first_match", []))
    for texto in as_list(opcoes.get("remover")):
        occurrence = "first" if texto in first_match else None
        clicar_texto(page, texto, occurrence=occurrence)


def preencher_login(page, site: dict):
    login_cfg = site["login"]
    page.goto(site["login_url"])
    page.get_by_role("textbox", name=login_cfg["email"]).fill(LOGIN_EMAIL)
    page.get_by_role("textbox", name=login_cfg["password"]).fill(LOGIN_PASSWORD)
    page.get_by_role("button", name=login_cfg["submit"]).click()


def print_caderno_resumo(caderno: dict):
    print("\n" + "=" * 70)
    print(f"Caderno: {caderno.get('nome', '<sem nome>')}")
    print("=" * 70)

    filtros = caderno.get("filtros", {})
    if filtros:
        print("Filtros:")
        for nome_filtro, valores in filtros.items():
            valores_formatados = ", ".join(as_list(valores))
            print(f"  - {nome_filtro}: {valores_formatados}")

    if caderno.get("opcoes"):
        print("Opções específicas:")
        for chave, valor in caderno["opcoes"].items():
            if isinstance(valor, list):
                valor = ", ".join(valor)
            print(f"  - {chave}: {valor}")
    else:
        print("Opções específicas: nenhuma")


def gerar_caderno(page, site: dict, caderno: dict):
    controls = site["notebook_controls"]

    page.get_by_role("link", name=controls["create_button"]).click()
    aplicar_filtros(page, caderno.get("filtros", {}), site)
    aplicar_opcoes(page, site, caderno.get("opcoes"))

    page.get_by_role("textbox", name=controls["name_field"]).fill(caderno["nome"])
    page.get_by_role("button", name=controls["generate_button"]).click()


def run(playwright: Playwright, config: dict, caderno: dict):
    site = config["site"]
    browser = playwright.chromium.launch(
        headless=False,
        slow_mo=300,
    )

    context = browser.new_context()
    page = context.new_page()

    try:
        preencher_login(page, site)
        print_caderno_resumo(caderno)
        gerar_caderno(page, site, caderno)

        print("\nMacro executada até o fim sem erro.")

    except Exception:
        print("\nO script encontrou um erro:\n")
        traceback.print_exc()

        print(
            "\nA página será mantida aberta. "
            "Você pode inspecionar o estado atual no navegador."
        )

        try:
            page.pause()
        except Exception:
            pass

    finally:
        esperar_usuario("Pressione ENTER para fechar o navegador.")
        context.close()
        browser.close()


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    site_path = resolver_caminho_config(args.site_config)
    cadernos_path = resolver_caminho_config(args.cadernos_config)
    config = carregar_configuracao(site_path, cadernos_path)

    if args.list_cadernos:
        listar_cadernos(config)
        return

    validar_credenciais()
    caderno = selecionar_caderno(config, args.caderno)

    with sync_playwright() as playwright:
        run(playwright, config, caderno)


if __name__ == "__main__":
    main()
