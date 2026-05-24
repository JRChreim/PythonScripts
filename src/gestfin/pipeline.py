from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import json
import re
import subprocess
import unicodedata
from pathlib import Path

from src.gestfin.xlsx_writer import (
    Cell,
    SheetData,
    header_cell,
    integer_cell,
    money_cell,
    text_cell,
)

DEFAULT_OUTPUT_FILENAME = "gestfin_monthly_report.xlsx"
DEFAULT_RULES_FILENAME = "category_rules.json"
DEFAULT_INPUT_DIRNAME = "Extratos"

DEFAULT_CATEGORIES = [
    "Receitas",
    "Transferencias internas",
    "Pessoas e terceiros",
    "Impostos e contribuicoes",
    "Contas fixas",
    "Saude e seguros",
    "Transporte e veiculo",
    "Mercado e alimentacao",
    "Compras e casa",
    "Assinaturas e digital",
    "Educacao",
    "Lazer",
    "Pets",
    "Negocio e fornecedores",
    "Tarifas, juros e emprestimos",
    "Pagamentos financeiros",
    "Saque em dinheiro",
    "Nao categorizado",
]

DEFAULT_RULES_TEMPLATE = {
    "version": 1,
    "notes": (
        "Edit rules in uppercase without accents. The parser normalizes the PDF text "
        "before matching. Rules are evaluated from highest priority to lowest."
    ),
    "categories": DEFAULT_CATEGORIES,
    "household_names": [
        "FABIO DE SANTIS CAMPOS",
        "VERONICA PINEIRO BOUZAS DO ESPIRITO SANTO",
        "VERONICA PINEIRO BOUZAS DO ESPIRITO SANT",
        "VERONICA PINEIRO B E SANT",
    ],
    "skip_patterns": [
        "SALDO DO DIA",
        "SALDO ANTERIOR",
    ],
    "stop_patterns": [
        "LANCAMENTOS FUTUROS",
    ],
    "rules": [
        {"pattern": "DAS MEI", "category": "Impostos e contribuicoes", "priority": 100, "nature": "tax"},
        {"pattern": "DA DAS MEI", "category": "Impostos e contribuicoes", "priority": 100, "nature": "tax"},
        {"pattern": "EST DAS MEI", "category": "Impostos e contribuicoes", "priority": 100, "nature": "refund"},
        {"pattern": "RECEITA FEDERAL", "category": "Impostos e contribuicoes", "priority": 95, "nature": "tax"},
        {"pattern": "MUNICIPIO DE PERUIBE", "category": "Impostos e contribuicoes", "priority": 90, "nature": "tax"},
        {"pattern": "SABESP", "category": "Contas fixas", "priority": 90, "nature": "expense"},
        {"pattern": "NEOENERGIA", "category": "Contas fixas", "priority": 90, "nature": "expense"},
        {"pattern": "CLARO", "category": "Contas fixas", "priority": 90, "nature": "expense"},
        {"pattern": "PLANO SANTA SAUDE", "category": "Saude e seguros", "priority": 95, "nature": "expense"},
        {"pattern": "DROGARIA SAO PAULO", "category": "Saude e seguros", "priority": 95, "nature": "expense"},
        {"pattern": "DROGARIA POPMED", "category": "Saude e seguros", "priority": 95, "nature": "expense"},
        {"pattern": "VIACAO PIRACICABANA", "category": "Transporte e veiculo", "priority": 95, "nature": "expense"},
        {"pattern": "AUTOPASS", "category": "Transporte e veiculo", "priority": 95, "nature": "expense"},
        {"pattern": "ECOVIAS", "category": "Transporte e veiculo", "priority": 95, "nature": "expense"},
        {"pattern": "SEM PARAR", "category": "Transporte e veiculo", "priority": 95, "nature": "expense"},
        {"pattern": "MOBIFACIL", "category": "Transporte e veiculo", "priority": 95, "nature": "expense"},
        {"pattern": "AUTO POSTO", "category": "Transporte e veiculo", "priority": 95, "nature": "expense"},
        {"pattern": "CENTRO D S AUTO NET", "category": "Transporte e veiculo", "priority": 95, "nature": "expense"},
        {"pattern": "KRILL PERUIBE", "category": "Mercado e alimentacao", "priority": 95, "nature": "expense"},
        {"pattern": "MERCADO EXTRA", "category": "Mercado e alimentacao", "priority": 95, "nature": "expense"},
        {"pattern": "MINI MERCADO", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "PANIFICADORA", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "FRUTAS", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "COCADASELIAS", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "SABORES DA EMPADA", "category": "Lazer", "priority": 90, "nature": "expense"},
        {"pattern": "BAR LOUNGE", "category": "Lazer", "priority": 90, "nature": "expense"},
        {"pattern": "QUIOSQUE", "category": "Lazer", "priority": 90, "nature": "expense"},
        {"pattern": "TUTTO BUONA", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "COMPANHIA BRASILEIRA DE D", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "ELECEBE COMERCIO DE ALIME", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "GRSA VIA MUND", "category": "Mercado e alimentacao", "priority": 90, "nature": "expense"},
        {"pattern": "LOJAS AMERICANAS", "category": "Compras e casa", "priority": 95, "nature": "expense"},
        {"pattern": "LOJAS AMERIC", "category": "Compras e casa", "priority": 85, "nature": "expense"},
        {"pattern": "COMPRE FACIL", "category": "Compras e casa", "priority": 90, "nature": "expense"},
        {"pattern": "V Z YAN COSMETICOS", "category": "Compras e casa", "priority": 90, "nature": "expense"},
        {"pattern": "MP.PARAISOCAPAS", "category": "Compras e casa", "priority": 90, "nature": "expense"},
        {"pattern": "RZ EMBALAGENS", "category": "Compras e casa", "priority": 90, "nature": "expense"},
        {"pattern": "C PERNAMBUCA", "category": "Compras e casa", "priority": 90, "nature": "expense"},
        {"pattern": "PETSHOPGRIMALDI", "category": "Pets", "priority": 95, "nature": "expense"},
        {"pattern": "AGROPERUIBE", "category": "Pets", "priority": 90, "nature": "expense"},
        {"pattern": "SPOTIFY", "category": "Assinaturas e digital", "priority": 95, "nature": "expense"},
        {"pattern": "DM SPOTIFY", "category": "Assinaturas e digital", "priority": 95, "nature": "expense"},
        {"pattern": "COMPLEXO EDUCACIONAL CAST", "category": "Educacao", "priority": 95, "nature": "expense"},
        {"pattern": "FATURA PAGA ITAU UNICLAS", "category": "Pagamentos financeiros", "priority": 100, "nature": "settlement"},
        {"pattern": "PAGTO A VISTA SALDO DEV", "category": "Pagamentos financeiros", "priority": 100, "nature": "settlement"},
        {"pattern": "PAGAMENTO EMPRESTIMO CDC", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "loan"},
        {"pattern": "PAGTO CDC", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "loan"},
        {"pattern": "BB CREDITO AUTOMATICO", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "loan"},
        {"pattern": "JUROS SALDO UTILIZ", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "fee"},
        {"pattern": "JUROS ATRASO LIM CONTA", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "fee"},
        {"pattern": "TARIFA MSG", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "fee"},
        {"pattern": "SEGURO CARTAO", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "fee"},
        {"pattern": "MENSALIDADE DE SEGURO", "category": "Tarifas, juros e emprestimos", "priority": 100, "nature": "fee"},
        {"pattern": "ESCRITORIO DAS", "category": "Negocio e fornecedores", "priority": 80, "nature": "income"},
        {"pattern": "FIOTEC", "category": "Receitas", "priority": 70, "nature": "income"},
        {"pattern": "PIX MARKETPLACE", "category": "Compras e casa", "priority": 80, "nature": "expense"},
        {"pattern": "KAOZ BEER", "category": "Lazer", "priority": 80, "nature": "expense"},
        {"pattern": "PIRIQUITO", "category": "Lazer", "priority": 80, "nature": "expense"},
        {"pattern": "TUZINO FLORES", "category": "Lazer", "priority": 80, "nature": "expense"},
        {"pattern": "MARCELOLOPES", "category": "Pessoas e terceiros", "priority": 80, "nature": "expense"},
        {"pattern": "SP20 LAV 60 MINUTO", "category": "Transporte e veiculo", "priority": 90, "nature": "expense"},
        {"pattern": "62307711GERS", "category": "Tarifas, juros e emprestimos", "priority": 80, "nature": "fee"},
        {"pattern": "FABIO DE SANTIS CAMPOS", "category": "Transferencias internas", "priority": 120, "nature": "transfer"},
        {"pattern": "VERONICA PINEIRO BOUZAS DO ESPIRITO SANTO", "category": "Transferencias internas", "priority": 120, "nature": "transfer"},
        {"pattern": "VERONICA PINEIRO BOUZAS DO ESPIRITO SANT", "category": "Transferencias internas", "priority": 120, "nature": "transfer"},
        {"pattern": "VERONICA PINEIRO B E SANT", "category": "Transferencias internas", "priority": 120, "nature": "transfer"},
    ],
}


@dataclass
class Rule:
    pattern: str
    category: str
    priority: int = 0
    nature: str | None = None


@dataclass
class RulesConfig:
    categories: list[str]
    household_names: set[str]
    skip_patterns: list[str]
    stop_patterns: list[str]
    rules: list[Rule]


@dataclass
class Transaction:
    transaction_date: date
    amount: Decimal
    raw_description: str
    display_description: str
    normalized_description: str
    source_file: str
    statement_owner: str
    bank: str
    statement_format: str
    transaction_tag: str = ""
    category: str = ""
    nature: str = ""
    classification_method: str = ""
    classification_note: str = ""
    matched_rule: str = ""


@dataclass
class StatementInfo:
    source_file: str
    bank: str
    owner_name: str
    statement_format: str
    transaction_count: int = 0
    first_date: date | None = None
    last_date: date | None = None


def ensure_rules_file(rules_path: Path | str) -> None:
    rules_path = Path(rules_path)
    if rules_path.exists():
        return
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text(json.dumps(DEFAULT_RULES_TEMPLATE, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rules_config(rules_path: Path | str) -> RulesConfig:
    rules_path = Path(rules_path)
    ensure_rules_file(rules_path)
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    categories = list(payload.get("categories") or DEFAULT_CATEGORIES)
    household_names = {normalize_text(name) for name in payload.get("household_names", [])}
    skip_patterns = [normalize_text(pattern) for pattern in payload.get("skip_patterns", [])]
    stop_patterns = [normalize_text(pattern) for pattern in payload.get("stop_patterns", [])]
    rules = []
    for item in payload.get("rules", []):
        pattern = normalize_text(item.get("pattern", ""))
        if not pattern:
            continue
        rules.append(
            Rule(
                pattern=pattern,
                category=item.get("category", "Nao categorizado"),
                priority=int(item.get("priority", 0)),
                nature=item.get("nature"),
            )
        )
    rules.sort(key=lambda rule: (-rule.priority, -len(rule.pattern), rule.pattern))
    return RulesConfig(
        categories=categories,
        household_names=household_names,
        skip_patterns=skip_patterns,
        stop_patterns=stop_patterns,
        rules=rules,
    )


def parse_pdf_directory(input_dir: Path | str, rules_config: RulesConfig | None = None) -> list[Transaction]:
    input_dir = Path(input_dir)
    pdf_paths = sorted(path for path in input_dir.glob("*.pdf") if path.is_file())
    transactions: list[Transaction] = []
    for pdf_path in pdf_paths:
        transactions.extend(parse_pdf_statement(pdf_path, rules_config))
    return transactions


def parse_pdf_statement(pdf_path: Path, rules_config: RulesConfig | None = None) -> list[Transaction]:
    text = extract_pdf_text(pdf_path)
    lines = [line.rstrip("\n") for line in text.splitlines()]
    owner_name = detect_statement_owner(lines, pdf_path)
    bank_name = detect_bank_name(lines, pdf_path)
    statement_format = detect_statement_format(lines, pdf_path)

    skip_patterns, stop_patterns = build_parser_patterns(rules_config)

    if statement_format == "santander":
        return parse_santander_statement(lines, pdf_path, owner_name, bank_name, statement_format, skip_patterns)
    if statement_format == "itau":
        return parse_itau_statement(lines, pdf_path, owner_name, bank_name, statement_format, skip_patterns)
    return parse_multiline_statement(lines, pdf_path, owner_name, bank_name, statement_format, skip_patterns, stop_patterns)


def parse_multiline_statement(
    lines: list[str],
    pdf_path: Path,
    owner_name: str,
    bank_name: str,
    statement_format: str,
    skip_patterns: list[str],
    stop_patterns: list[str],
) -> list[Transaction]:
    transactions: list[Transaction] = []
    current: Transaction | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        normalized_line = normalize_text(line)
        if any(pattern in normalized_line for pattern in stop_patterns):
            break
        if any(pattern in normalized_line for pattern in skip_patterns):
            continue

        if _looks_like_full_date_transaction(line):
            parts = re.split(r"\s{2,}", line.strip())
            if not parts:
                continue
            amount_text = parts[-1]
            amount = parse_amount(amount_text)
            if amount is None:
                continue

            description_parts = parts[1:-1] if len(parts) > 2 else parts[1:2]
            raw_description = " ".join(part.strip() for part in description_parts if part.strip())
            raw_description = raw_description.strip()
            if _is_balance_like(normalize_text(raw_description)):
                continue

            current = Transaction(
                transaction_date=parse_date(parts[0]),
                amount=amount,
                raw_description=raw_description,
                display_description=build_display_description(raw_description),
                normalized_description=normalize_text(raw_description),
                source_file=pdf_path.name,
                statement_owner=owner_name,
                bank=bank_name,
                statement_format=statement_format,
            )
            transactions.append(current)
            continue

        if current is not None and _looks_like_continuation(line):
            current.raw_description = f"{current.raw_description} {line}".strip()
            current.display_description = build_display_description(current.raw_description)
            current.normalized_description = normalize_text(current.raw_description)

    return [tx for tx in transactions if not _is_balance_like(tx.normalized_description)]


def parse_santander_statement(
    lines: list[str],
    pdf_path: Path,
    owner_name: str,
    bank_name: str,
    statement_format: str,
    skip_patterns: list[str],
) -> list[Transaction]:
    transactions: list[Transaction] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        normalized_line = normalize_text(line)
        if any(pattern in normalized_line for pattern in skip_patterns):
            continue
        if not re.match(r"^\d{2}/\d{2}/\d{4}\s+", line):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            continue
        amount = parse_amount(parts[-2])
        if amount is None:
            continue
        raw_description = parts[0][10:].strip()
        if _is_balance_like(normalize_text(raw_description)):
            continue
        transactions.append(
            Transaction(
                transaction_date=parse_date(parts[0][:10]),
                amount=amount,
                raw_description=raw_description,
                display_description=build_display_description(raw_description),
                normalized_description=normalize_text(raw_description),
                source_file=pdf_path.name,
                statement_owner=owner_name,
                bank=bank_name,
                statement_format=statement_format,
            )
        )
    return transactions


def parse_itau_statement(
    lines: list[str],
    pdf_path: Path,
    owner_name: str,
    bank_name: str,
    statement_format: str,
    skip_patterns: list[str],
) -> list[Transaction]:
    transactions: list[Transaction] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        normalized_line = normalize_text(line)
        if any(pattern in normalized_line for pattern in skip_patterns):
            continue
        if not re.match(r"^\d{2}/\d{2}/\d{4}\s+", line):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 2:
            continue
        amount = parse_amount(parts[-1])
        if amount is None:
            continue
        raw_description = parts[0][10:].strip()
        if _is_balance_like(normalize_text(raw_description)):
            continue
        transactions.append(
            Transaction(
                transaction_date=parse_date(parts[0][:10]),
                amount=amount,
                raw_description=raw_description,
                display_description=build_display_description(raw_description),
                normalized_description=normalize_text(raw_description),
                source_file=pdf_path.name,
                statement_owner=owner_name,
                bank=bank_name,
                statement_format=statement_format,
            )
        )
    return transactions


def build_report(
    transactions: list[Transaction],
    rules_config: RulesConfig,
    input_dir: Path,
    rules_path: Path,
    output_path: Path,
) -> list[SheetData]:
    internal_names = set(rules_config.household_names)
    for transaction in transactions:
        if transaction.statement_owner:
            internal_names.add(normalize_text(transaction.statement_owner))
    for owner_name in _extract_owner_names(transactions):
        internal_names.add(owner_name)

    classify_transactions(transactions, rules_config, internal_names)

    statements = summarize_statements(transactions)
    months = group_transactions_by_month(transactions)
    overview_rows = build_overview_sheet_rows(
        statements=statements,
        months=months,
        transactions=transactions,
        rules_config=rules_config,
        input_dir=input_dir,
        rules_path=rules_path,
        output_path=output_path,
    )
    sheets = [SheetData(name="Overview", rows=overview_rows)]

    for month_key in sorted(months):
        sheets.append(SheetData(name=month_key, rows=build_month_sheet_rows(month_key, months[month_key], rules_config)))

    review_rows = build_review_sheet_rows(transactions)
    sheets.append(SheetData(name="Review", rows=review_rows))
    sheets.append(SheetData(name="Rules", rows=build_rules_sheet_rows(rules_config)))
    return sheets


def classify_transactions(transactions: list[Transaction], rules_config: RulesConfig, internal_names: set[str]) -> None:
    for transaction in transactions:
        classify_transaction(transaction, rules_config, internal_names)


def classify_transaction(transaction: Transaction, rules_config: RulesConfig, internal_names: set[str]) -> None:
    normalized_text = transaction.normalized_description
    transaction.transaction_tag = detect_transaction_tag(normalized_text)

    matched_rule = match_rule(normalized_text, rules_config.rules)
    if matched_rule is not None:
        transaction.category = matched_rule.category
        transaction.nature = matched_rule.nature or infer_nature(transaction, matched_rule.category)
        transaction.classification_method = "rule"
        transaction.classification_note = f"pattern={matched_rule.pattern}"
        transaction.matched_rule = matched_rule.pattern
        return

    if _is_balance_like(normalized_text):
        transaction.category = "Nao categorizado"
        transaction.nature = "ignore"
        transaction.classification_method = "ignored"
        transaction.classification_note = "balance line"
        return

    if transaction.transaction_tag == "withdrawal":
        transaction.category = "Saque em dinheiro"
        transaction.nature = "withdrawal"
        transaction.classification_method = "heuristic"
        transaction.classification_note = "cash withdrawal"
        return

    if transaction.transaction_tag in {"settlement", "loan"}:
        transaction.category = "Pagamentos financeiros" if transaction.transaction_tag == "settlement" else "Tarifas, juros e emprestimos"
        transaction.nature = transaction.transaction_tag
        transaction.classification_method = "heuristic"
        transaction.classification_note = transaction.transaction_tag
        return

    if transaction.transaction_tag in {"tax", "tax_refund"}:
        transaction.category = "Impostos e contribuicoes"
        transaction.nature = "refund" if transaction.transaction_tag == "tax_refund" else "tax"
        transaction.classification_method = "heuristic"
        transaction.classification_note = transaction.transaction_tag
        return

    if transaction.transaction_tag in {"fee", "interest"}:
        transaction.category = "Tarifas, juros e emprestimos"
        transaction.nature = "fee"
        transaction.classification_method = "heuristic"
        transaction.classification_note = transaction.transaction_tag
        return

    if transaction.amount > 0:
        if transaction.transaction_tag == "pix_received":
            if _contains_any(normalized_text, internal_names):
                transaction.category = "Transferencias internas"
                transaction.nature = "transfer"
                transaction.classification_method = "heuristic"
                transaction.classification_note = "internal transfer received"
                return
            transaction.category = "Receitas"
            transaction.nature = "income"
            transaction.classification_method = "heuristic"
            transaction.classification_note = "received pix"
            return

        if _contains_any(normalized_text, internal_names):
            transaction.category = "Transferencias internas"
            transaction.nature = "transfer"
            transaction.classification_method = "heuristic"
            transaction.classification_note = "internal positive transfer"
            return

        transaction.category = "Receitas"
        transaction.nature = "income"
        transaction.classification_method = "fallback"
        transaction.classification_note = "positive default"
        return

    if transaction.amount < 0:
        if _contains_any(normalized_text, internal_names):
            transaction.category = "Transferencias internas"
            transaction.nature = "transfer"
            transaction.classification_method = "heuristic"
            transaction.classification_note = "internal transfer"
            return

        merchant_category = infer_merchant_category(normalized_text)
        if merchant_category is not None:
            transaction.category = merchant_category
            transaction.nature = infer_nature(transaction, merchant_category)
            transaction.classification_method = "heuristic"
            transaction.classification_note = "merchant keyword"
            return

        if transaction.transaction_tag == "pix_received" and _contains_any(
            normalized_text,
            {"ROD PERUIBE II", "BANCO 24H", "BANCO24H", "ATM", "SAQUE"},
        ):
            transaction.category = "Saque em dinheiro"
            transaction.nature = "withdrawal"
            transaction.classification_method = "heuristic"
            transaction.classification_note = "withdrawal fallback"
            return

        if _looks_like_person(normalized_text):
            transaction.category = "Pessoas e terceiros"
            transaction.nature = "expense"
            transaction.classification_method = "heuristic"
            transaction.classification_note = "person-to-person payment"
            return

        if transaction.transaction_tag == "pix_sent":
            transaction.category = "Pagamentos financeiros" if _contains_any(normalized_text, {"FATURA PAGA", "PAGTO A VISTA", "SALDO DEV"}) else "Nao categorizado"
            transaction.nature = "expense"
            transaction.classification_method = "fallback"
            transaction.classification_note = "pix sent fallback"
            return

        transaction.category = "Nao categorizado"
        transaction.nature = "expense"
        transaction.classification_method = "fallback"
        transaction.classification_note = "default expense"
        return

    transaction.category = "Nao categorizado"
    transaction.nature = "unknown"
    transaction.classification_method = "fallback"
    transaction.classification_note = "zero amount"


def build_overview_sheet_rows(
    statements: list[StatementInfo],
    months: dict[str, list[Transaction]],
    transactions: list[Transaction],
    rules_config: RulesConfig,
    input_dir: Path,
    rules_path: Path,
    output_path: Path,
) -> list[list[Cell | object]]:
    rows: list[list[Cell | object]] = []
    rows.append([header_cell("GestFin Monthly Expense Analysis")])
    rows.append([text_cell(f"Input folder: {input_dir}")])
    rows.append([text_cell(f"Rules file: {rules_path}")])
    rows.append([text_cell(f"Output file: {output_path}")])
    rows.append([])

    rows.append([header_cell("Statement Inventory")])
    rows.append([
        header_cell("Source file"),
        header_cell("Bank"),
        header_cell("Owner"),
        header_cell("Format"),
        header_cell("Transactions"),
        header_cell("First date"),
        header_cell("Last date"),
    ])
    for statement in statements:
        rows.append([
            text_cell(statement.source_file),
            text_cell(statement.bank),
            text_cell(statement.owner_name),
            text_cell(statement.statement_format),
            integer_cell(statement.transaction_count),
            text_cell(statement.first_date.isoformat() if statement.first_date else ""),
            text_cell(statement.last_date.isoformat() if statement.last_date else ""),
        ])
    rows.append([])

    rows.append([header_cell("Monthly Summary")])
    rows.append([
        header_cell("Month"),
        header_cell("Income"),
        header_cell("Spend"),
        header_cell("Transfers"),
        header_cell("Settlements"),
        header_cell("Net cashflow"),
        header_cell("Needs review"),
    ])
    for month_key in sorted(months):
        month_transactions = months[month_key]
        metrics = summarize_transactions(month_transactions)
        rows.append([
            text_cell(month_key),
            money_cell(metrics["income"]),
            money_cell(metrics["spend"]),
            money_cell(metrics["transfer"]),
            money_cell(metrics["settlement"]),
            money_cell(metrics["net"]),
            integer_cell(metrics["needs_review"]),
        ])
    rows.append([])

    rows.append([header_cell("Category Totals Across All Months")])
    rows.append([
        header_cell("Category"),
        header_cell("Credits"),
        header_cell("Debits"),
        header_cell("Net"),
        header_cell("Count"),
    ])
    totals = summarize_category_totals(transactions, rules_config.categories)
    for category in rules_config.categories:
        category_totals = totals.get(category, {"credits": Decimal("0"), "debits": Decimal("0"), "count": 0})
        rows.append([
            text_cell(category),
            money_cell(category_totals["credits"]),
            money_cell(category_totals["debits"]),
            money_cell(category_totals["credits"] - category_totals["debits"]),
            integer_cell(category_totals["count"]),
        ])

    return rows


def build_month_sheet_rows(month_key: str, transactions: list[Transaction], rules_config: RulesConfig) -> list[list[Cell | object]]:
    rows: list[list[Cell | object]] = []
    metrics = summarize_transactions(transactions)
    rows.append([header_cell(f"Month: {month_key}")])
    rows.append([
        text_cell(f"Transactions: {len(transactions)}"),
        text_cell(f"Income: {format_money(metrics['income'])}"),
        text_cell(f"Spend: {format_money(metrics['spend'])}"),
        text_cell(f"Transfers: {format_money(metrics['transfer'])}"),
        text_cell(f"Settlements: {format_money(metrics['settlement'])}"),
        text_cell(f"Net cashflow: {format_money(metrics['net'])}"),
        text_cell(f"Needs review: {metrics['needs_review']}"),
    ])
    rows.append([])

    rows.append([header_cell("Category Summary")])
    rows.append([
        header_cell("Category"),
        header_cell("Credits"),
        header_cell("Debits"),
        header_cell("Net"),
        header_cell("Count"),
    ])
    category_totals = summarize_category_totals(transactions, rules_config.categories)
    for category in rules_config.categories:
        totals = category_totals.get(category, {"credits": Decimal("0"), "debits": Decimal("0"), "count": 0})
        rows.append([
            text_cell(category),
            money_cell(totals["credits"]),
            money_cell(totals["debits"]),
            money_cell(totals["credits"] - totals["debits"]),
            integer_cell(totals["count"]),
        ])
    rows.append([])

    rows.append([header_cell("Transactions")])
    rows.append([
        header_cell("Date"),
        header_cell("Amount"),
        header_cell("Category"),
        header_cell("Nature"),
        header_cell("Method"),
        header_cell("Tag"),
        header_cell("Display description"),
        header_cell("Normalized description"),
        header_cell("Owner"),
        header_cell("Bank"),
        header_cell("Source file"),
        header_cell("Rule"),
        header_cell("Note"),
    ])
    for transaction in transactions:
        rows.append([
            text_cell(transaction.transaction_date.isoformat()),
            money_cell(transaction.amount),
            text_cell(transaction.category),
            text_cell(transaction.nature),
            text_cell(transaction.classification_method),
            text_cell(transaction.transaction_tag),
            text_cell(transaction.display_description),
            text_cell(transaction.normalized_description),
            text_cell(transaction.statement_owner),
            text_cell(transaction.bank),
            text_cell(transaction.source_file),
            text_cell(transaction.matched_rule),
            text_cell(transaction.classification_note),
        ])

    return rows


def build_review_sheet_rows(transactions: list[Transaction]) -> list[list[Cell | object]]:
    rows: list[list[Cell | object]] = []
    rows.append([header_cell("Review Queue")])
    rows.append([text_cell("Transactions classified by heuristic or left uncategorized. These are the ones to review when refining rules.")])
    rows.append([])
    rows.append([
        header_cell("Normalized description"),
        header_cell("Category"),
        header_cell("Method"),
        header_cell("Tag"),
        header_cell("Count"),
        header_cell("Credits"),
        header_cell("Debits"),
        header_cell("Net"),
        header_cell("First date"),
        header_cell("Last date"),
        header_cell("Sample owner"),
        header_cell("Sample source"),
        header_cell("Sample note"),
    ])

    grouped: dict[tuple[str, str, str], list[Transaction]] = defaultdict(list)
    for transaction in transactions:
        if transaction.classification_method == "rule" and transaction.category != "Nao categorizado":
            continue
        key = (transaction.normalized_description, transaction.category, transaction.classification_method)
        grouped[key].append(transaction)

    for key, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][0])):
        group_sorted = sorted(group, key=lambda tx: (tx.transaction_date, tx.source_file, tx.raw_description))
        credits = sum((tx.amount for tx in group_sorted if tx.amount > 0), Decimal("0"))
        debits = sum((-tx.amount for tx in group_sorted if tx.amount < 0), Decimal("0"))
        rows.append([
            text_cell(key[0]),
            text_cell(key[1]),
            text_cell(key[2]),
            text_cell(group_sorted[0].transaction_tag),
            integer_cell(len(group_sorted)),
            money_cell(credits),
            money_cell(debits),
            money_cell(credits - debits),
            text_cell(group_sorted[0].transaction_date.isoformat()),
            text_cell(group_sorted[-1].transaction_date.isoformat()),
            text_cell(group_sorted[0].statement_owner),
            text_cell(group_sorted[0].source_file),
            text_cell(group_sorted[0].classification_note),
        ])

    if len(rows) == 4:
        rows.append([text_cell("All transactions were classified by explicit rules.")])

    return rows


def build_rules_sheet_rows(rules_config: RulesConfig) -> list[list[Cell | object]]:
    rows: list[list[Cell | object]] = []
    rows.append([header_cell("Rules")])
    rows.append([text_cell("Edit category_rules.json to change the classifier. Patterns are matched against normalized uppercase text.")])
    rows.append([])

    rows.append([header_cell("Household names")])
    rows.append([header_cell("Name")])
    for name in sorted(rules_config.household_names):
        rows.append([text_cell(name)])
    rows.append([])

    rows.append([header_cell("Skip patterns")])
    rows.append([header_cell("Pattern")])
    for pattern in rules_config.skip_patterns:
        rows.append([text_cell(pattern)])
    rows.append([])

    rows.append([header_cell("Rule table")])
    rows.append([
        header_cell("Priority"),
        header_cell("Pattern"),
        header_cell("Category"),
        header_cell("Nature"),
    ])
    for rule in rules_config.rules:
        rows.append([
            integer_cell(rule.priority),
            text_cell(rule.pattern),
            text_cell(rule.category),
            text_cell(rule.nature or ""),
        ])
    return rows


def summarize_statements(transactions: list[Transaction]) -> list[StatementInfo]:
    summary: dict[str, StatementInfo] = {}
    for transaction in transactions:
        key = transaction.source_file
        statement = summary.get(key)
        if statement is None:
            statement = StatementInfo(
                source_file=transaction.source_file,
                bank=transaction.bank,
                owner_name=transaction.statement_owner,
                statement_format=transaction.statement_format,
            )
            summary[key] = statement
        statement.transaction_count += 1
        if statement.first_date is None or transaction.transaction_date < statement.first_date:
            statement.first_date = transaction.transaction_date
        if statement.last_date is None or transaction.transaction_date > statement.last_date:
            statement.last_date = transaction.transaction_date
    return sorted(summary.values(), key=lambda item: item.source_file)


def summarize_transactions(transactions: list[Transaction]) -> dict[str, Decimal | int]:
    income = Decimal("0")
    spend = Decimal("0")
    transfer = Decimal("0")
    settlement = Decimal("0")
    net = Decimal("0")
    needs_review = 0
    for transaction in transactions:
        net += transaction.amount
        if transaction.classification_method != "rule" or transaction.category == "Nao categorizado":
            needs_review += 1
        if transaction.category == "Transferencias internas":
            transfer += abs(transaction.amount)
            continue
        if transaction.category == "Pagamentos financeiros" or transaction.category == "Saque em dinheiro":
            settlement += abs(transaction.amount)
            continue
        if transaction.amount > 0:
            income += transaction.amount
        elif transaction.amount < 0:
            spend += -transaction.amount
    return {
        "income": income,
        "spend": spend,
        "transfer": transfer,
        "settlement": settlement,
        "net": net,
        "needs_review": needs_review,
    }


def summarize_category_totals(transactions: list[Transaction], category_order: list[str]) -> dict[str, dict[str, Decimal | int]]:
    totals: dict[str, dict[str, Decimal | int]] = defaultdict(lambda: {"credits": Decimal("0"), "debits": Decimal("0"), "count": 0})
    for transaction in transactions:
        bucket = totals[transaction.category]
        bucket["count"] += 1
        if transaction.amount > 0:
            bucket["credits"] += transaction.amount
        elif transaction.amount < 0:
            bucket["debits"] += -transaction.amount
    for category in category_order:
        totals.setdefault(category, {"credits": Decimal("0"), "debits": Decimal("0"), "count": 0})
    return totals


def group_transactions_by_month(transactions: list[Transaction]) -> dict[str, list[Transaction]]:
    grouped: dict[str, list[Transaction]] = defaultdict(list)
    for transaction in transactions:
        month_key = transaction.transaction_date.strftime("%Y-%m")
        grouped[month_key].append(transaction)
    for month_key in grouped:
        grouped[month_key].sort(key=lambda tx: (tx.transaction_date, tx.source_file, tx.raw_description))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "pdftotext is required to parse the statement PDFs. Please install poppler-utils or an equivalent package."
        ) from exc
    return completed.stdout


def detect_statement_owner(lines: list[str], pdf_path: Path) -> str:
    for raw_line in lines[:20]:
        line = normalize_text(raw_line)
        if not line:
            continue
        match = re.search(r"\bCLIENTE\s+(.+)", line)
        if match:
            candidate = _trim_before_keywords(match.group(1))
            if candidate:
                return candidate
        match = re.match(
            r"^(?P<name>[A-Z0-9 .'\-]+?)\s+(?:AGENCIA(?: E CONTA)?|AG\.?:)",
            line,
        )
        if match:
            candidate = _trim_before_keywords(match.group("name"))
            if candidate:
                return candidate
    return pdf_path.stem.upper()


def detect_bank_name(lines: list[str], pdf_path: Path) -> str:
    filename = normalize_text(pdf_path.name)
    if "SANTANDER" in filename:
        return "Santander"
    if "ITAU" in filename:
        return "Itaú"
    for raw_line in lines[:20]:
        line = normalize_text(raw_line)
        if "SANTANDER" in line:
            return "Santander"
        if "ITAU" in line:
            return "Itaú"
    return "Unknown"


def detect_statement_format(lines: list[str], pdf_path: Path) -> str:
    filename = normalize_text(pdf_path.name)
    if "ITAU" in filename:
        return "itau"
    if "SANTANDER" in filename:
        return "santander"
    normalized_lines = [normalize_text(line) for line in lines[:40]]
    if any("DATA DESCRICAO DOCTO SITUACAO CREDITO DEBITO SALDO" in line for line in normalized_lines):
        return "santander"
    if any("DATA LANCAMENTOS VALOR SALDO" in line for line in normalized_lines):
        return "itau"
    if any("EXTRATO CONTA / LANCAMENTOS" in line for line in normalized_lines):
        return "itau"
    return "multiline"


def parse_amount(amount_text: str) -> Decimal | None:
    text = amount_text.strip().upper()
    if not text:
        return None
    text = text.replace("R$", "").replace(" ", "")
    sign = 1
    if text.endswith("(+)"):
        text = text[:-3]
    elif text.endswith("(-)"):
        sign = -1
        text = text[:-3]
    if text.startswith("-"):
        sign = -1
        text = text[1:]
    elif text.startswith("+"):
        text = text[1:]
    text = text.replace(".", "").replace(",", ".")
    text = text.strip()
    if not text:
        return None
    try:
        return Decimal(text) * sign
    except InvalidOperation:
        return None


def parse_date(date_text: str) -> date:
    return datetime.strptime(date_text.strip(), "%d/%m/%Y").date()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.upper()
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_display_description(raw_description: str) -> str:
    text = normalize_text(raw_description)
    text = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", " ", text)
    text = re.sub(r"\b\d{2}/\d{2}\b", " ", text)
    text = re.sub(r"\b\d{2}:\d{2}\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    for prefix in DISPLAY_PREFIXES:
        text = re.sub(prefix, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or normalize_text(raw_description)


def match_rule(normalized_text: str, rules: list[Rule]) -> Rule | None:
    for rule in rules:
        if rule.pattern in normalized_text:
            return rule
    return None


def infer_nature(transaction: Transaction, category: str) -> str:
    if category == "Transferencias internas":
        return "transfer"
    if category == "Pagamentos financeiros":
        return "settlement"
    if category == "Saque em dinheiro":
        return "withdrawal"
    if category == "Tarifas, juros e emprestimos":
        return "fee" if transaction.amount < 0 else "income"
    if transaction.amount > 0:
        return "income"
    return "expense"


def detect_transaction_tag(normalized_text: str) -> str:
    if _is_balance_like(normalized_text):
        return "balance"
    if "PAGTO A VISTA SALDO DEV" in normalized_text or "FATURA PAGA" in normalized_text:
        return "settlement"
    if "PAGAMENTO EMPRESTIMO CDC" in normalized_text or "PAGTO CDC" in normalized_text or "BB CREDITO AUTOMATICO" in normalized_text:
        return "loan"
    if "JUROS" in normalized_text or "TARIFA" in normalized_text or "SEGURO CARTAO" in normalized_text or "MENSALIDADE DE SEGURO" in normalized_text:
        return "fee"
    if "DAS MEI" in normalized_text or "DA DAS MEI" in normalized_text:
        return "tax"
    if "EST DAS MEI" in normalized_text:
        return "tax_refund"
    if "RECEITA FEDERAL" in normalized_text or "MUNICIPIO" in normalized_text:
        return "tax"
    if "SAQUE" in normalized_text:
        return "withdrawal"
    if "PIX RECEBIDO" in normalized_text:
        return "pix_received"
    if "PIX ENVIADO" in normalized_text or "PIX TRANSF" in normalized_text or "PIX QRS" in normalized_text:
        return "pix_sent"
    if "COMPRA COM CARTAO" in normalized_text or "DEBITO VISA ELECTRON BRASIL" in normalized_text or normalized_text.startswith("RSCSS") or normalized_text.startswith("RSCCS"):
        return "card_purchase"
    if "TED" in normalized_text or "DOC" in normalized_text:
        return "bank_transfer"
    return "generic"


def infer_merchant_category(normalized_text: str) -> str | None:
    transport_keywords = [
        "VIACAO PIRACICABANA",
        "AUTOPASS",
        "ECOVIAS",
        "SEM PARAR",
        "MOBIFACIL",
        "AUTO POSTO",
        "CENTRO D S AUTO NET",
        "SP20 LAV 60 MINUTO",
    ]
    health_keywords = [
        "DROGARIA SAO PAULO",
        "DROGARIA POPMED",
        "PLANO SANTA SAUDE",
    ]
    food_keywords = [
        "KRILL PERUIBE",
        "MERCADO EXTRA",
        "MINI MERCADO",
        "PANIFICADORA",
        "FRUTAS",
        "COCADASELIAS",
        "SABORES DA EMPADA",
        "BAR LOUNGE",
        "QUIOSQUE",
        "TUTTO BUONA",
        "KAOZ BEER",
        "PIRIQUITO",
        "COMPANHIA BRASILEIRA DE D",
        "ELECEBE COMERCIO DE ALIME",
        "GRSA VIA MUND",
    ]
    shopping_keywords = [
        "LOJAS AMERICANAS",
        "COMPRE FACIL",
        "V Z YAN COSMETICOS",
        "MP PARAISOCAPAS",
        "MP PARAISO CAPAS",
        "RZ EMBALAGENS",
        "PIX MARKETPLACE",
        "C PERNAMBUCA",
    ]
    education_keywords = ["COMPLEXO EDUCACIONAL CAST"]
    pet_keywords = ["PETSHOPGRIMALDI", "AGROPERUIBE"]
    digital_keywords = ["SPOTIFY", "DM SPOTIFY"]
    fixed_cost_keywords = ["SABESP", "NEOENERGIA", "CLARO"]
    business_keywords = ["ESCRITORIO DAS", "FIOTEC"]

    if _contains_any(normalized_text, transport_keywords):
        return "Transporte e veiculo"
    if _contains_any(normalized_text, health_keywords):
        return "Saude e seguros"
    if _contains_any(normalized_text, food_keywords):
        return "Mercado e alimentacao"
    if _contains_any(normalized_text, shopping_keywords):
        return "Compras e casa"
    if _contains_any(normalized_text, education_keywords):
        return "Educacao"
    if _contains_any(normalized_text, pet_keywords):
        return "Pets"
    if _contains_any(normalized_text, digital_keywords):
        return "Assinaturas e digital"
    if _contains_any(normalized_text, fixed_cost_keywords):
        return "Contas fixas"
    if _contains_any(normalized_text, business_keywords):
        return "Negocio e fornecedores"
    if "MENSALIDADE DE SEGURO" in normalized_text or "SEGURO CARTAO" in normalized_text:
        return "Tarifas, juros e emprestimos"
    return None


def _contains_any(text: str, patterns: set[str] | list[str] | tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _looks_like_person(normalized_text: str) -> bool:
    tokens = normalized_text.split()
    if len(tokens) < 2:
        return False
    if any(keyword in normalized_text for keyword in PERSON_NAME_BLOCKLIST):
        return False
    letter_tokens = [token for token in tokens if token.isalpha()]
    return len(letter_tokens) >= 2


def _is_balance_like(normalized_text: str) -> bool:
    return "SALDO DO DIA" in normalized_text or "SALDO ANTERIOR" in normalized_text or normalized_text == "SALDO"


def _looks_like_full_date_transaction(line: str) -> bool:
    return bool(re.match(r"^\d{2}/\d{2}/\d{4}\s+", line))


def _looks_like_continuation(line: str) -> bool:
    return bool(line and not _looks_like_full_date_transaction(line) and not re.match(r"^\d{2}/\d{2}/\d{4}$", line))


def _trim_before_keywords(text: str) -> str:
    for keyword in ["AGENCIA", "AG.:", "CONTA"]:
        index = text.find(keyword)
        if index > 0:
            text = text[:index].strip()
    return text.strip()


def _extract_owner_names(transactions: list[Transaction]) -> set[str]:
    names: set[str] = set()
    for transaction in transactions:
        owner = normalize_text(transaction.statement_owner)
        if owner:
            names.add(owner)
    return names


def _build_skip_patterns() -> list[str]:
    return [normalize_text(pattern) for pattern in DEFAULT_SKIP_LINE_PATTERNS]


def _build_stop_patterns() -> list[str]:
    return [normalize_text(pattern) for pattern in DEFAULT_STOP_PATTERNS]


def build_parser_patterns(rules_config: RulesConfig | None) -> tuple[list[str], list[str]]:
    skip_patterns = _build_skip_patterns()
    stop_patterns = _build_stop_patterns()
    if rules_config is not None:
        skip_patterns.extend(pattern for pattern in rules_config.skip_patterns if pattern not in skip_patterns)
        stop_patterns.extend(pattern for pattern in rules_config.stop_patterns if pattern not in stop_patterns)
    return skip_patterns, stop_patterns


DEFAULT_SKIP_LINE_PATTERNS = [
    "EXTRATO DE CONTA CORRENTE",
    "INTERNET BANKING",
    "PERIODO:",
    "LANCAMENTOS",
    "LANCAMENTOS FUTUROS",
    "SALDO DO DIA",
    "SALDO ANTERIOR",
    "SALDO DE CONTA CORRENTE",
    "SALDO BLOQUEADO",
    "PROVISAO DE ENCARGOS",
    "DATA DE DEBITO DE JUROS",
    "DATA DE DEBITO DE IOF",
    "TOTAL APLICACOES FINANCEIRAS",
    "AVISO!",
    "CONSULTAS, INFORMACOES E SERVICOS TRANSACIONAIS",
]

DEFAULT_STOP_PATTERNS = [
    "LANCAMENTOS FUTUROS",
]

DISPLAY_PREFIXES = [
    r"^PIX ENVIADO\s+",
    r"^PIX RECEBIDO\s+",
    r"^PIX TRANSF\s+",
    r"^PIX QRS\s+",
    r"^COMPRA COM CARTAO\s+",
    r"^DEBITO VISA ELECTRON BRASIL\s+\d{2}\s+\d{2}\s+",
    r"^DEBITO VISA ELECTRON BRASIL\s+\d{2}/\d{2}\s+",
    r"^TED[-\s]*PAG FORNECEDORES\s+",
    r"^PAGAMENTO DE BOLETO\s+",
    r"^FATURA PAGA\s+",
    r"^PAGTO A VISTA SALDO DEV\s+",
    r"^PAGTO CDC\s+",
    r"^PAGAMENTO EMPRESTIMO CDC\s+",
    r"^DA DAS MEI\s+",
    r"^EST DAS MEI\s+",
    r"^MENSALIDADE DE SEGURO\s+",
    r"^SEGURO CARTAO\s+",
    r"^JUROS SALDO UTILIZ.*\s+",
    r"^JUROS ATRASO LIM CONTA\s+",
    r"^TARIFA MSG\s+",
    r"^BB CREDITO AUTOMATICO\s+",
]

PERSON_NAME_BLOCKLIST = {
    "MERCADO",
    "SABESP",
    "CLARO",
    "NEOENERGIA",
    "DROGARIA",
    "LOJAS",
    "MOBIFACIL",
    "AUTOPASS",
    "ECOVIAS",
    "SEM PARAR",
    "PAGTO",
    "PAGAMENTO",
    "FATURA",
    "BOLETO",
    "SEGURO",
    "TARIFA",
    "JUROS",
    "DAS",
    "RECEITA",
    "MUNICIPIO",
    "POSTO",
    "SPOTIFY",
}


def format_money(value: Decimal | int | float) -> str:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    text = f"{value:,.2f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")
