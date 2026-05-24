from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

XL_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

ET.register_namespace("", XL_MAIN_NS)
ET.register_namespace("r", REL_NS)

STYLE_TEXT = 0
STYLE_HEADER = 1
STYLE_MONEY = 2
STYLE_INTEGER = 3


@dataclass(frozen=True)
class Cell:
    value: object
    style: int = STYLE_TEXT


@dataclass
class SheetData:
    name: str
    rows: list[list[Cell | object]] = field(default_factory=list)


def text_cell(value: object) -> Cell:
    return Cell(value=value, style=STYLE_TEXT)


def header_cell(value: object) -> Cell:
    return Cell(value=value, style=STYLE_HEADER)


def money_cell(value: object) -> Cell:
    return Cell(value=value, style=STYLE_MONEY)


def integer_cell(value: object) -> Cell:
    return Cell(value=value, style=STYLE_INTEGER)


def write_xlsx(path: Path, sheets: list[SheetData]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(sheets))
        archive.writestr("_rels/.rels", _package_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml(sheets))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(sheets))
        archive.writestr("xl/styles.xml", _styles_xml())
        for index, sheet in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(sheet))

    if path.exists():
        path.unlink()
    tmp_path.rename(path)


def _sanitize_sheet_name(name: str) -> str:
    cleaned = name.replace("\\", " ").replace("/", " ").replace("?", " ").replace("*", " ")
    cleaned = cleaned.replace("[", " ").replace("]", " ").replace(":", " ")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > 31:
        cleaned = cleaned[:31]
    return cleaned or "Sheet"


def _content_types_xml(sheets: list[SheetData]) -> bytes:
    root = ET.Element(
        "Types",
        {
            "xmlns": "http://schemas.openxmlformats.org/package/2006/content-types",
        },
    )
    ET.SubElement(
        root,
        "Default",
        {
            "Extension": "rels",
            "ContentType": "application/vnd.openxmlformats-package.relationships+xml",
        },
    )
    ET.SubElement(
        root,
        "Default",
        {
            "Extension": "xml",
            "ContentType": "application/xml",
        },
    )
    ET.SubElement(
        root,
        "Override",
        {
            "PartName": "/xl/workbook.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
        },
    )
    ET.SubElement(
        root,
        "Override",
        {
            "PartName": "/xl/styles.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml",
        },
    )
    for index, _sheet in enumerate(sheets, start=1):
        ET.SubElement(
            root,
            "Override",
            {
                "PartName": f"/xl/worksheets/sheet{index}.xml",
                "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
            },
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _package_rels_xml() -> bytes:
    root = ET.Element("Relationships", {"xmlns": PKG_REL_NS})
    ET.SubElement(
        root,
        "Relationship",
        {
            "Id": "rId1",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
            "Target": "xl/workbook.xml",
        },
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _workbook_xml(sheets: list[SheetData]) -> bytes:
    root = ET.Element(
        "workbook",
        {
            "xmlns": XL_MAIN_NS,
        },
    )
    sheets_element = ET.SubElement(root, "sheets")
    for index, sheet in enumerate(sheets, start=1):
        ET.SubElement(
            sheets_element,
            "sheet",
            {
                "name": _sanitize_sheet_name(sheet.name),
                "sheetId": str(index),
                f"{{{REL_NS}}}id": f"rId{index}",
            },
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _workbook_rels_xml(sheets: list[SheetData]) -> bytes:
    root = ET.Element("Relationships", {"xmlns": PKG_REL_NS})
    for index, _sheet in enumerate(sheets, start=1):
        ET.SubElement(
            root,
            "Relationship",
            {
                "Id": f"rId{index}",
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
                "Target": f"worksheets/sheet{index}.xml",
            },
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _styles_xml() -> bytes:
    root = ET.Element("styleSheet", {"xmlns": XL_MAIN_NS})

    num_fmts = ET.SubElement(root, "numFmts", {"count": "1"})
    ET.SubElement(num_fmts, "numFmt", {"numFmtId": "164", "formatCode": "#,##0.00"})

    fonts = ET.SubElement(root, "fonts", {"count": "2"})
    default_font = ET.SubElement(fonts, "font")
    ET.SubElement(default_font, "sz", {"val": "11"})
    ET.SubElement(default_font, "color", {"theme": "1"})
    ET.SubElement(default_font, "name", {"val": "Calibri"})
    ET.SubElement(default_font, "family", {"val": "2"})
    bold_font = ET.SubElement(fonts, "font")
    ET.SubElement(bold_font, "b")
    ET.SubElement(bold_font, "sz", {"val": "11"})
    ET.SubElement(bold_font, "color", {"theme": "1"})
    ET.SubElement(bold_font, "name", {"val": "Calibri"})
    ET.SubElement(bold_font, "family", {"val": "2"})

    fills = ET.SubElement(root, "fills", {"count": "2"})
    fill_none = ET.SubElement(fills, "fill")
    ET.SubElement(fill_none, "patternFill", {"patternType": "none"})
    fill_gray = ET.SubElement(fills, "fill")
    ET.SubElement(fill_gray, "patternFill", {"patternType": "gray125"})

    borders = ET.SubElement(root, "borders", {"count": "1"})
    border = ET.SubElement(borders, "border")
    ET.SubElement(border, "left")
    ET.SubElement(border, "right")
    ET.SubElement(border, "top")
    ET.SubElement(border, "bottom")
    ET.SubElement(border, "diagonal")

    cell_style_xfs = ET.SubElement(root, "cellStyleXfs", {"count": "1"})
    ET.SubElement(cell_style_xfs, "xf", {"numFmtId": "0", "fontId": "0", "fillId": "0", "borderId": "0"})

    cell_xfs = ET.SubElement(root, "cellXfs", {"count": "4"})
    ET.SubElement(
        cell_xfs,
        "xf",
        {"numFmtId": "0", "fontId": "0", "fillId": "0", "borderId": "0", "xfId": "0"},
    )
    ET.SubElement(
        cell_xfs,
        "xf",
        {"numFmtId": "0", "fontId": "1", "fillId": "0", "borderId": "0", "xfId": "0", "applyFont": "1"},
    )
    ET.SubElement(
        cell_xfs,
        "xf",
        {
            "numFmtId": "164",
            "fontId": "0",
            "fillId": "0",
            "borderId": "0",
            "xfId": "0",
            "applyNumberFormat": "1",
        },
    )
    ET.SubElement(
        cell_xfs,
        "xf",
        {
            "numFmtId": "1",
            "fontId": "0",
            "fillId": "0",
            "borderId": "0",
            "xfId": "0",
            "applyNumberFormat": "1",
        },
    )

    cell_styles = ET.SubElement(root, "cellStyles", {"count": "1"})
    ET.SubElement(cell_styles, "cellStyle", {"name": "Normal", "xfId": "0", "builtinId": "0"})

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _worksheet_xml(sheet: SheetData) -> bytes:
    root = ET.Element("worksheet", {"xmlns": XL_MAIN_NS})
    if sheet.rows:
        ET.SubElement(root, "dimension", {"ref": f"A1:{_cell_ref(len(sheet.rows), _max_row_width(sheet.rows))}"})
    sheet_data = ET.SubElement(root, "sheetData")

    for row_index, row in enumerate(sheet.rows, start=1):
        row_element = ET.SubElement(sheet_data, "row", {"r": str(row_index)})
        for column_index, raw_cell in enumerate(row, start=1):
            cell = raw_cell if isinstance(raw_cell, Cell) else Cell(raw_cell)
            if cell.value is None or cell.value == "":
                continue
            cell_ref = _cell_ref(row_index, column_index)
            cell_attributes = {"r": cell_ref}
            if cell.style == STYLE_HEADER:
                cell_attributes["s"] = "1"
            elif cell.style == STYLE_MONEY:
                cell_attributes["s"] = "2"
            elif cell.style == STYLE_INTEGER:
                cell_attributes["s"] = "3"
            c_element = ET.SubElement(row_element, "c", cell_attributes)
            _write_cell_value(c_element, cell.value)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _write_cell_value(cell_element: ET.Element, value: object) -> None:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float, Decimal)):
        v_element = ET.SubElement(cell_element, "v")
        v_element.text = _format_numeric(value)
        return

    text = "" if value is None else str(value)
    cell_element.set("t", "inlineStr")
    is_element = ET.SubElement(cell_element, "is")
    t_element = ET.SubElement(is_element, "t")
    if text.startswith(" ") or text.endswith(" "):
        t_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t_element.text = text


def _format_numeric(value: int | float | Decimal) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _cell_ref(row_index: int, column_index: int) -> str:
    return f"{_column_letter(column_index)}{row_index}"


def _column_letter(column_index: int) -> str:
    letters = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _max_row_width(rows: list[list[Cell | object]]) -> int:
    return max((len(row) for row in rows), default=1)
