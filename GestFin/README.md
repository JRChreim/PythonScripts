# GestFin

This folder contains the monthly expense-categorization workflow.

## Workflow

1. Drop new statement PDFs into `GestFin/Extratos/`.
2. Edit `GestFin/category_rules.json` as you discover new merchants.
3. Run:

```bash
gestfin-expenses --open
```

The script rebuilds `GestFin/gestfin_monthly_report.xlsx` from the PDFs each time, so the workbook stays in sync with the source files. Use `--open` if you want the workbook to open automatically after it is written.

## What To Expect

- Transactions are grouped by the actual transaction date, so each row lands in the month sheet that matches its date.
- The workbook is rebuilt from the PDFs on every run, rather than edited in place.
- If a month already exists in the output file, that month sheet is regenerated from scratch using the current PDFs and rules.
- The source of truth is the PDF folder plus `category_rules.json`; the Excel file is the derived report.

## Output

- `GestFin/gestfin_monthly_report.xlsx`: rolling workbook with one sheet per month, plus overview/review/rules sheets
- `GestFin/category_rules.json`: the editable rule file used to classify transactions
