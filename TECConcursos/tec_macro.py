from pathlib import Path

from playwright.sync_api import sync_playwright
import yaml
import time


def pausa(segundos=1.0):
    time.sleep(segundos)


def carregar_cadernos(arquivo="cadernos.yaml"):
    caminho = Path(arquivo)
    if not caminho.is_absolute():
        caminho = Path(__file__).resolve().parent / caminho

    with open(caminho, "r", encoding="utf-8") as f:
        dados = yaml.safe_load(f)

    return dados["cadernos"]


def aguardar_usuario(mensagem):
    input(f"\n{mensagem}\nPressione ENTER para continuar...")


def preencher_caderno(page, cad):
    print("\n" + "=" * 70)
    print(f"Caderno: {cad['nome']}")
    print("=" * 70)

    filtros = cad.get("filtros", {})
    if filtros:
        print("Filtros selecionados:")
        for nome_filtro, valores in filtros.items():
            if isinstance(valores, list):
                valores_formatados = ", ".join(valores)
            else:
                valores_formatados = str(valores)
            print(f"  - {nome_filtro}: {valores_formatados}")

    opcoes = cad.get("opcoes", {})
    if opcoes:
        print("Opções adicionais:")
        for chave, valor in opcoes.items():
            if isinstance(valor, list):
                valor_formatado = ", ".join(valor)
            else:
                valor_formatado = str(valor)
            print(f"  - {chave}: {valor_formatado}")

    aguardar_usuario(
        "Agora deixe aberta no navegador a tela onde você quer aplicar os filtros "
        "para este caderno."
    )

    # ------------------------------------------------------------------
    # A partir daqui entram os comandos específicos do TecConcursos.
    # Por enquanto deixamos comentado até descobrirmos os seletores corretos.
    # ------------------------------------------------------------------


    #
    # page.get_by_label("Nome").fill(cad["nome"])
    #
    # for nome_filtro, valores in cad.get("filtros", {}).items():
    #     page.get_by_text(nome_filtro).click()
    #     for valor in valores:
    #         page.get_by_text(valor).click()

    aguardar_usuario(
        "Revise manualmente este caderno no navegador. "
        "Por enquanto, o script ainda não vai clicar em salvar/criar."
    )


def main():
    cadernos = carregar_cadernos()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=300,
        )

        context = browser.new_context()
        page = context.new_page()

        page.goto("https://www.tecconcursos.com.br/login")

        aguardar_usuario(
            "Faça login manualmente no TecConcursos no navegador que abriu."
        )

        for cad in cadernos:
            preencher_caderno(page, cad)

        aguardar_usuario("Processo finalizado.")
        browser.close()


if __name__ == "__main__":
    main()
